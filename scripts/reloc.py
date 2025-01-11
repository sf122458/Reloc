#! /home/ros/miniconda3/envs/hloc/bin/python

import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, String
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from pathlib import Path
import pycolmap
from hloc import extract_features, match_features, pairs_from_retrieval
from hloc.utils.base_model import dynamic_load
from hloc import extractors
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
import torch

import time
from nav_msgs.msg import Odometry

from utils import quat2rot, inv_quat, quat_mul

"""
- reloc (root)
    - data
        - dataset_name
            - db    # database images
                - *.jpg
            - query # query images(在校准时用到，每次拍摄会覆盖)
                - *.jpg
    - scripts
        - reloc.py  # ros node
        - utils.py
    - outputs       # offline 运行hloc后的输出文件
        - dataset_name
            - *.h5
            - *.txt
"""

class Localizer:
    CAMERA_TOPIC = "/camera/color/image_raw"
    def __init__(self, dataset_name: str):
        np.set_printoptions(precision=3, suppress=True)
        print("Localizer starts initializing.")

        # 获取运行hloc所需要的文件路径
        self.root = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..'))
        self.images = self.root / f'data/{dataset_name}'
        if not os.path.exists(self.images):
            raise FileExistsError
        db_images = self.images / 'db'
        self.outputs = self.root / f"outputs/{dataset_name}"
        self.sfm_pairs = self.outputs / "pairs-netvlad.txt"
        sfm_dir = self.outputs / "sfm"

        self.feature_conf = extract_features.confs['disk']
        self.matcher_conf = match_features.confs['NN-mutual']
        self.retrieval_conf = extract_features.confs['netvlad']
        self.retrieval_path = self.outputs / 'global-feats-netvlad.h5'

        self.loc_conf = {
            'estimation': {'ransac': {'max_error': 12}},
            'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
        }

        db_image_list = [p.relative_to(self.images).as_posix() for p in (db_images).iterdir()]
        self.model = pycolmap.Reconstruction(sfm_dir)
        Model = dynamic_load(extractors, self.retrieval_conf["model"]["name"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retrieval_model = Model(self.retrieval_conf["model"]).eval().to(device)
        self.query = 'query/query.jpg'
        (self.db_desc, self.db_names, self.query_names) = pairs_from_retrieval.prepare(
            self.retrieval_path, db_list=db_image_list, query_list=[self.query])
        

        ######## Calibration ########

        self.R_point2local = None
        self.T_point2local = None
        self.S = None
        self.q_transform = None

        self.calibration = {
            "xyz_cam_in_point": [],
            "xyz_cam_in_local": [],
            "q_point": [],
            "q_local": []
        }

        ######## ROS topic ########

        # 订阅VINS的odometry信息作为校准时的ground truth
        rospy.Subscriber(
            # "vins_estimator/odometry",
            "vins_fusion/imu_propagate",
            Odometry,
            self.odom_cb
        )

        # 记录当前位置
        self.local_pos = PoseStamped()

        # 通过shfiles中的add_cali.sh拍摄校准图像并记录位置坐标
        rospy.Subscriber("add_cali",
                        data_class=Empty,
                        callback=self.cali_record_callback)
        
        # 通过shfiles中的calc_cali.sh计算校准参数
        rospy.Subscriber("calc_cali",
                        data_class=Empty,
                        callback=self.cali_calc_callback)
        
        # 通过shfiles中的reloc.sh [Path]重定位图像坐标
        rospy.Subscriber("reloc",
                        data_class=String,      # the path of the image to be relocated
                        callback=self.reloc_callback)

        # 重置校准参数
        rospy.Subscriber("reset_cali",
                         data_class=Empty,
                        callback=self.reset_callback)

        print("Localizer finishes initializing.")

    def odom_cb(self, vins_odom: Odometry):
        # 位置信息
        self.local_pos.pose.position.x = vins_odom.pose.pose.position.x
        self.local_pos.pose.position.y = vins_odom.pose.pose.position.y
        self.local_pos.pose.position.z = vins_odom.pose.pose.position.z
 
        # 姿态信息
        self.local_pos.pose.orientation.x = vins_odom.pose.pose.orientation.x
        self.local_pos.pose.orientation.y = vins_odom.pose.pose.orientation.y
        self.local_pos.pose.orientation.z = vins_odom.pose.pose.orientation.z
        self.local_pos.pose.orientation.w = vins_odom.pose.pose.orientation.w

    def hloc_reloc(self):
        """
            对query图像进行重定位，返回点云坐标系下的坐标
        """
        time_start = time.time()
        extract_features.extract_feature_from_query(
            self.retrieval_conf,
            self.images,
            query_name=self.query,
            model=self.retrieval_model,
            export_dir=self.outputs
        )
        pairs_from_retrieval.fast_retrieval(
            self.retrieval_path,
            self.sfm_pairs,
            num_matched=5,
            db_desc=self.db_desc,
            db_names=self.db_names,
            query_names=self.query_names
        )
        feature_path = extract_features.main(
            self.feature_conf, 
            self.images,
            self.outputs,
            image_list=[self.query],
            overwrite=True)
        match_path = match_features.main(
            self.matcher_conf,
            self.sfm_pairs,
            self.feature_conf["output"],
            self.outputs,
            overwrite=True
        )

        camera = pycolmap.Camera()
        camera.model = pycolmap.CameraModel.SIMPLE_RADIAL
        h, w = cv2.imread(os.path.join(self.images, self.query)).shape[:2]
        camera.width = w
        camera.height = h
        camera.params = [1.2 * max(h,w), w/2.0, h/2.0, 0.0]

        references_registered = self.get_pairs_info(self.sfm_pairs)
        ref_ids = [self.model.find_image_with_name(name).image_id for name in references_registered]
        localizer = QueryLocalizer(self.model, self.loc_conf)
        ret, log = pose_from_cluster(localizer, self.query, camera, ref_ids, feature_path, match_path)
        print(f"Inference time: {time.time() - time_start} s")

        # shape = (1, 3)
        return self.cam2world(ret["cam_from_world"]), ret["cam_from_world"].rotation.quat

    # utils
    def cam2world(self, rigid):
        """
            将相机坐标系下的坐标(hloc得出的结果)转换为点云坐标系下的坐标
        """
        assert isinstance(rigid, pycolmap.Rigid3d)
        return -np.linalg.inv(quat2rot(rigid.rotation.quat)) @ rigid.translation

    def get_pairs_info(self, file_path: Path):
        """
            读取pairs文件中的图像名,将图像对信息提取到数组中,跳过读取md5信息以节省时间
        """
        pairs_info = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    pairs_info.append(parts[1])
        return pairs_info
    
    def record_query_image(self):
        """
            记录当前图像并保存为query.jpg
        """
        msg = rospy.wait_for_message(self.CAMERA_TOPIC, Image, timeout=5.0)
        image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        name = os.path.join(self.images, "query/query.jpg")
        cv2.imwrite(name, image)

        
    def _calc_scale(self, xyz_a: np.ndarray, xyz_b: np.ndarray):
        """
            通过最小二乘法计算坐标系的缩放矩阵
            xyz_a: 3 x N
            xyz_b: 3 x N
        """
        matrix_x = np.zeros((xyz_a.shape[1] - 1, 1))
        matrix_y = np.zeros((xyz_a.shape[1] - 1, 1))
        for i in range(xyz_a.shape[1] - 1):
            matrix_x[i] = np.linalg.norm(xyz_a[:, i + 1] - xyz_a[:, i])
            matrix_y[i] = np.linalg.norm(xyz_b[:, i + 1] - xyz_b[:, i])
        return (np.linalg.inv(matrix_x.T @ matrix_x) @ matrix_x.T @ matrix_y).item()
    
    def _get_transform(self):
        """
            通过以获得的校准图像计算坐标系的变换矩阵
        """
        xyz_a = np.array(self.calibration["xyz_cam_in_point"]).T    # Final shape: (3, N)
        xyz_b = np.array(self.calibration["xyz_cam_in_local"]).T    # Final shape: (3, N)
        # scale = np.linalg.norm(xyz_b) / np.linalg.norm(xyz_a)

        # TODO: a robust method to calculate scale matrix
        # scale = np.sqrt(np.sum(np.square(xyz_b[:, 0] - xyz_b[:, 1])) / np.sum(np.square(xyz_a[:, 0] - xyz_a[:, 1])))
        scale = self._calc_scale(xyz_a, xyz_b)
        xyz_a = xyz_a * scale
        self.S = np.diag([scale, scale, scale])
        centroid_a = np.mean(xyz_a, axis=-1, keepdims=True)
        centroid_b = np.mean(xyz_b, axis=-1, keepdims=True)
        H = (xyz_a - centroid_a) @ (xyz_b - centroid_b).T
        U, _, V = np.linalg.svd(H)
        R = V.T @ U.T
        T = -R @ centroid_a + centroid_b
        self.R_point2local = R
        self.T_point2local = T
        print(f"Scale: {scale}\nRotation: {R}\nTranslation: {T}")

    def cali_record_callback(self, msg: Empty):
        # 记录校准图像
        print("Start recording...")
        self.record_query_image()

        # 记录当前位置
        coord_in_local = np.array([self.local_pos.pose.position.x,
                        self.local_pos.pose.position.y,
                        self.local_pos.pose.position.z])
        # FIXME: XYZW
        q_local = np.array([self.local_pos.pose.orientation.x, 
                            self.local_pos.pose.orientation.y,
                            self.local_pos.pose.orientation.z,
                            self.local_pos.pose.orientation.w,])

        coord_in_point, q_point = self.hloc_reloc() # 点云坐标系下的坐标
        
        self.calibration["xyz_cam_in_point"].append(coord_in_point)
        self.calibration["q_point"].append(q_point)

        # TODO: record the camera pose in the local coordinate
        self.calibration["xyz_cam_in_local"].append(coord_in_local)
        self.calibration["q_local"].append(q_local)


        self.q_transform = quat_mul(inv_quat(q_point), q_local)
        
        print("Coord in point-cloud coordinate: ", coord_in_point
              , "\nCoord in local coordinate: ", coord_in_local)

    def cali_calc_callback(self, msg: Empty):
        # 开始校准
        print("Start calibrating...")
        self._get_transform()

    def reloc_callback(self, msg: String):
        """
            对已有的图像进行重定位，最终获得目标图像在当前本地坐标系下的坐标
            msg: 需要重定位的图像路径
        """
        path = msg.data
        print("Start relocating...")
        if os.path.exists(path):
            os.system(f"cp {path} {self.images}/query/query.jpg")

        xyz_cam_in_point, q_point = self.hloc_reloc()
        # TODO: transform the camera pose in the point-cloud coordinate to the world coordinate
        xyz_cam_in_local = self.R_point2local @ self.S @ np.array([xyz_cam_in_point]).T + self.T_point2local
        try:
            q_local = quat_mul(q_point, self.q_transform)
        except:
            q_local = (0, 0, 0, 1)
        print(r"------------------------------------------")
        print(r"------------------------------------------")
        print(f"Predict: xyz: {xyz_cam_in_local[:,0]}, q: {q_local}")
        print(r"------------------------------------------")
        print(r"------------------------------------------")

        # TODO: 如何把目标位置通过offboard代码发送给px4，并控制无人机飞到目标位置

    def reset_callback(self, msg: Empty):
        self.calibration = {
            "xyz_cam_in_point": [],
            "xyz_cam_in_local": [],
            "q_point": [],
            "q_local": []
        }
        self.R_point2local = None
        self.T_point2local = None
        self.S = None
        self.q_transform = None
        print("Calibration data reset.")


if __name__ == "__main__":
    import sys
    rospy.init_node("camera_test")
    rate = rospy.Rate(20)
    # FIXME: dataset name
    localizer = Localizer(dataset_name=sys.argv[1])
    while not rospy.is_shutdown():
        rate.sleep()
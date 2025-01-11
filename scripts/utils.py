import numpy as np
import math
from typing import List, Tuple
from pycolmap import Rigid3d
from copy import deepcopy
import pycolmap
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from hloc import extract_features, match_features, pairs_from_exhaustive
from geometry_msgs.msg import PoseStamped

# utils for quaternion calculation and pose wrapping

def quat2rot(q):
    x, y, z, w = q
    return np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                    [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                    [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])

def quat_mul(q1, q2):
    """
        Multiply two quaternions
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = x1*w2 + y1*z2 - z1*y2 + w1*x2
    y = -x1*z2 + y1*w2 + z1*x2 + w1*y2
    z = x1*y2 - y1*x2 + z1*w2 + w1*z2
    w = -x1*x2 - y1*y2 - z1*z2 + w1*w2
    return (x, y, z, w)

def inv_quat(q):
    """
        Inverse the quaternion
    """
    x, y, z, w = q
    return (-x, -y, -z, w)

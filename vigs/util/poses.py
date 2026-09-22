import numpy as np
from scipy.spatial.transform import Rotation as R


def to_se3_vec(pose_mat):
    quat = R.from_matrix(pose_mat[:3, :3]).as_quat()
    return np.hstack((pose_mat[:3, 3], quat))

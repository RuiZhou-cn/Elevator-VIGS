import copy

import numpy as np
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D

def load_tum_to_pose_traj(filepath):
    timestamps = []
    positions_xyz = []
    orientations_quat_wxyz = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = list(map(float, line.strip().split()))
            if len(parts) != 8:
                continue
            t, tx, ty, tz, qx, qy, qz, qw = parts
            orientations_quat_wxyz.append([qw, qx, qy, qz])
            positions_xyz.append([tx, ty, tz])
            timestamps.append(t)
    return PoseTrajectory3D(positions_xyz=positions_xyz,
                            orientations_quat_wxyz=orientations_quat_wxyz,
                            timestamps=np.array(timestamps))

def compute_recall_from_file(gt_path, est_path, thresh_cm=10, max_diff=0.01):
    traj_ref = load_tum_to_pose_traj(gt_path)
    traj_est = load_tum_to_pose_traj(est_path)

    traj_ref_sync, traj_est_sync = sync.associate_trajectories(traj_ref, traj_est, max_diff=max_diff,)

    traj_est_aligned = copy.deepcopy(traj_est_sync)
    r_a, t_a, s = traj_est_aligned.align(traj_ref_sync, correct_scale=True)

    thresh_m = thresh_cm / 100.0
    matches = 0
    gt_positions = traj_ref_sync.positions_xyz
    est_positions = traj_est_aligned.positions_xyz

    for gt_pos in gt_positions:
        distances = np.linalg.norm(est_positions - gt_pos, axis=1)
        if np.min(distances) < thresh_m:
            matches += 1

    total_poses = len(gt_positions)
    recall_percentage = 100.0 * matches / total_poses if total_poses > 0 else 0.0

    return matches, total_poses, recall_percentage

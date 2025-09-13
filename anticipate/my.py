from .stable_virtual_camera.demo import svc_main, Model
import numpy as np
from scipy.spatial.transform import Rotation as R
import quaternion
from numpy import quaternion
import math


def build_trajectory(traj, pos, theta):
    trajectory = {
        'camera_pose': [],
        'camera_rotation': [],
        'camera_rotation_euler': [],
        'camera_extrinsic': [],
    }
    trajectory_json = {
        'camera_pose': [],
        'camera_rotation_euler': [],
    }
    def _append():
        # 欧拉角 -> 四元数
        r   = R.from_euler('xyz', theta, degrees=False)
        x, y, z, w = r.as_quat()
        quat = np.quaternion(w, x, y, z)
        trajectory['camera_pose'].append(np.round(pos, 4))
        # 保存四元数
        trajectory['camera_rotation'].append(quat)
        # 保存弧度制
        trajectory['camera_rotation_euler'].append(np.round(theta, 4))
        trajectory_json['camera_pose'].append(np.round(pos, 4).tolist())
        trajectory_json['camera_rotation_euler'].append(np.round(theta, 4).tolist())
        rot_mat = r.as_matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = rot_mat
        c2w[:3, 3] = pos
        trajectory['camera_extrinsic'].append(np.round(c2w, 6))
    _append()                                # first frame (no-op)

    prior = [0,0,0]
    for point in traj:
        degrees = np.degrees(point[2]) - np.degrees(prior[2])
        if degrees >=0:
            theta[1] -= np.radians(abs(degrees))
        else:
            theta[1] += np.radians(abs(degrees))
        forward_size = math.sqrt((point[0]-prior[0])**2 + (point[1]-prior[1])**2)
        dx = forward_size * np.sin(theta[1])
        dz = forward_size * np.cos(theta[1])
        pos += np.array([dx, 0.0, dz])
        prior = point
        _append()
    return trajectory, trajectory_json


def trajectory_prediction(trajectory, action_folder, model, imagine_resolution):

    pos   = np.zeros(3, dtype=float)
    theta = np.array([np.radians(-10.0), 0.0, 0.0], dtype=float)

    traj, _ = build_trajectory(trajectory, pos, theta)
    svc_main(
            model=model,
            data_path=action_folder,
            task="img2trajvid_s-prob",
            replace_or_include_input=True,
            traj_prior='',
            cfg=4.0,
            guider=1,
            L_short=imagine_resolution,
            num_targets=24,
            use_traj_prior=True,
            output_path=action_folder,
            chunk_strategy="interp",
            c2ws=traj['camera_extrinsic'],
        )
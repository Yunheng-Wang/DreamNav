import numpy as np
import random

def filtering_trajectories(all_trajectory, k=4):
    # input : all_trajectory -> [[traj1, traj2, ...]]

    all_trajectory = all_trajectory[0]

    def trajectory_distance(traj1, traj2):
        traj1 = np.array(traj1)[:,:-1]
        traj2 = np.array(traj2)[:,:-1]
        return np.mean(np.linalg.norm(traj1 - traj2, axis=1))

    # 构建轨迹间的距离矩阵
    num_traj = len(all_trajectory)
    Distance_Matrix = np.zeros((num_traj, num_traj))
    for i in range(num_traj):
        for j in range(i + 1, num_traj):
            d = trajectory_distance(all_trajectory[i], all_trajectory[j])
            Distance_Matrix[i, j] = Distance_Matrix[j, i] = d
    
    selected = [random.randint(0,23)]  
    while len(selected) < k:
        max_dist, max_idx = -np.inf, -1
        for i in range(len(Distance_Matrix)):
            if i in selected:
                continue
            min_dist_to_selected = min(Distance_Matrix[i, j] for j in selected)
            if min_dist_to_selected > max_dist:
                max_dist = min_dist_to_selected
                max_idx = i
        selected.append(max_idx)
    
    filter_all_trajectory = []
    for index in selected:
        filter_all_trajectory.append(all_trajectory[index])

    return np.expand_dims(filter_all_trajectory, axis=0)


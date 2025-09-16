import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import imageio.v2 as imageio
import cv2
import datetime
import time
    
    
def get_current_rgb_depth(simulator, save_path, save=True):
    
    observations = simulator.get_sensor_observations()

    rgb_image = observations["sensor_rgb"] 
    depth_image = observations["sensor_depth"]

    if save:
        rgb_img = Image.fromarray(rgb_image.astype(np.uint8), mode="RGBA")  # 确保是 uint8
        rgb_img.save(os.path.join(save_path, "egocentric_rgb.png"))

        depth_vis = (depth_image / np.nanmax(depth_image) * 255).astype(np.uint8)
        depth_img = Image.fromarray(depth_vis)
        depth_img.save(os.path.join(save_path, "egocentric_depth.png"))

    


    return rgb_image, depth_image



def visualize_trajectories(intrinsic,image,n_trajectories, save_path, save_name):
    trajectory_masks = []

    image = np.expand_dims(image[:, :, :3], axis=0)
    color = np.array([255, 0, 0]) 

    for i in range(image.shape[0]):
        trajectory_mask = np.array(image[i])
        n_trajectory = n_trajectories[i,:,:,0:2]

        for waypoints in n_trajectory:
            input_points = np.zeros((waypoints.shape[0],3)) - 0.2
            input_points[:,0:2] = waypoints
            input_points[:,1] = -input_points[:,1]

            camera_z = image[0].shape[0] - 1 - intrinsic[1][1] * input_points[:,2] / (input_points[:,0] + 1e-8) - intrinsic[1][2]
            camera_x = intrinsic[0][0] * input_points[:,1] / (input_points[:,0] + 1e-8) + intrinsic[0][2]
            
            for i in range(camera_x.shape[0]-1):
                try:
                    if camera_x[i] > 0 and camera_z[i] > 0 and camera_x[i+1] > 0 and camera_z[i+1] > 0:
                        trajectory_mask = cv2.line(trajectory_mask,(int(camera_x[i]),int(camera_z[i])),(int(camera_x[i+1]),int(camera_z[i+1])),color.astype(np.uint8).tolist(),5)
                except:
                    pass
            
        trajectory_masks.append(trajectory_mask)
    trajectory_masks = np.concatenate(trajectory_masks,axis=1)
    
    format_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
    save_name = f"{save_path}{format_time}_" + save_name + ".png"
    imageio.imwrite(save_name, trajectory_mask.astype(np.uint8))
    
    return trajectory_masks

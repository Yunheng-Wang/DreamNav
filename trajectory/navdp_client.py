import json
import numpy as np
import argparse
import cv2
import io
import time
import imageio
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .navdp_server import navdp_reset,navdp_reset_env,navdp_step_xy,navdp_step_nogoal
import matplotlib
matplotlib.use('Agg')
import math

def pointnav_reset(save_path, intrinsic=None, stop_threshold=-4.0,batch_size=1,env_id=None):
    if env_id is None:
        navdp_reset(intrinsic.tolist(),stop_threshold,batch_size, save_path)
    else:
        navdp_reset(env_id, save_path)


def pointnav_step(goals,rgb_images,depth_images):
    # 多 batch 图片拼接
    concat_images = np.concatenate([img for img in rgb_images],axis=0)
    concat_depths = np.concatenate([img for img in depth_images],axis=0)

    # 将rgb图像转化为png格式并保存
    _, rgb_image = cv2.imencode('.jpg', concat_images)
    image_bytes = io.BytesIO()
    image_bytes.write(rgb_image)
    
    # 将depth图像转化为png格式并保存
    depth_image = np.clip(concat_depths*10000.0,0,65535.0).astype(np.uint16)
    _, depth_image = cv2.imencode('.png', depth_image)
    depth_bytes = io.BytesIO()
    depth_bytes.write(depth_image)
    
    # 保存为请求可用的数据
    files = {
        'image': ('image.jpg', image_bytes.getvalue(), 'image/jpeg'),
        'depth': ('depth.png', depth_bytes.getvalue(), 'image/png'),
    }
    data = {
        'goal_data': json.dumps({
        'goal_x': goals[:,0].tolist(),
        'goal_y': goals[:,1].tolist()
        }),
        'depth_time':time.time(),
        'rgb_time':time.time(),
    }

    result = navdp_step_xy(files=files, data=data)
    
    trajectory = result['trajectory']
    all_trajectory = result['all_trajectory']
    all_value = result['all_values']
    return np.array(trajectory),np.array(all_trajectory),np.array(all_value)


def nogoal_step(rgb_images,depth_images):
    concat_images = np.concatenate([img for img in rgb_images],axis=0)
    concat_depths = np.concatenate([img for img in depth_images],axis=0)

    _, rgb_image = cv2.imencode('.jpg', concat_images)
    image_bytes = io.BytesIO()
    image_bytes.write(rgb_image)
    
    depth_image = np.clip(concat_depths*10000.0,0,65535.0).astype(np.uint16)
    _, depth_image = cv2.imencode('.png', depth_image)
    depth_bytes = io.BytesIO()
    depth_bytes.write(depth_image)
    
    files = {
        'image': ('image.jpg', image_bytes.getvalue(), 'image/jpeg'),
        'depth': ('depth.png', depth_bytes.getvalue(), 'image/png'),
    }
    data = {
        'goal_data': json.dumps({
        'goal_x': np.zeros((rgb_images.shape[0],)).tolist(),
        'goal_y': np.zeros((rgb_images.shape[0],)).tolist(),
        }),
        'depth_time':time.time(),
        'rgb_time':time.time(),
    }
    result = navdp_step_nogoal(files=files, data=data)

    trajectory = result['trajectory']
    all_trajectory = result['all_trajectory']
    all_value = result['all_values']
    return np.array(trajectory),np.array(all_trajectory),np.array(all_value)


def get_pointcloud_from_depth(rgb,depth,intrinsic):
    if len(depth.shape) == 3:
        depth = depth[:,:,0]
    filter_z,filter_x = np.where(depth>0)
    depth_values = depth[filter_z,filter_x]
    pixel_z = (-depth.shape[0] + filter_z  + intrinsic[1][2]) * depth_values / intrinsic[1][1]
    pixel_x = (filter_x - intrinsic[0][2])*depth_values / intrinsic[0][0]
    pixel_y = depth_values
    color_values = rgb[filter_z,filter_x]
    point_values = np.stack([pixel_y,-pixel_x,pixel_z],axis=-1)
    return filter_z,filter_x,depth_values,point_values,color_values
    

def bev_visualize(intrinsic,rgb_image,depth_image,exec_trajectory,trajectory_all,trajectory_values):
    vis_depth_image = (np.clip((depth_image/2000.0),0,1)*255).astype(np.uint8)
    vis_depth_image = np.tile(vis_depth_image[:,:,None],(1,1,3))
    rgbd_vis_image = np.concatenate((rgb_image,vis_depth_image),axis=1)
    _,_,_,points,_ = get_pointcloud_from_depth(rgb_image,depth_image/1000.0,intrinsic)
    points = points[(points[:,2]>points[:,2].min() + 0.4) & (points[:,2]<points[:,2].max() - 1.0)]
    
    plt.figure(figsize=(10,6))
    ax1 = plt.subplot(1,2,1)
    ax1.scatter(-points[:,1],points[:,0],s=2)
    ax1.plot(-exec_trajectory[:,1],exec_trajectory[:,0],color='b',label='exec')
    ax1.set_xlim(-6.0, 6.0)
    ax1.set_ylim(-6.0, 6.0)
    
    norm_values = np.clip(trajectory_values+1.0,0,1)
    colormap = cm.get_cmap('jet')
    ax2 = plt.subplot(1,2,2)
    ax2.scatter(-points[:,1],points[:,0],s=2)
    for i in range(trajectory_all.shape[0]):
        color = np.array(colormap(norm_values[i])) * 255.0
        ax2.plot(-trajectory_all[i,:,1],trajectory_all[i,:,0],color=color[:3]/255.0,linewidth=1.0)
    ax2.set_xlim(-6.0, 6.0)
    ax2.set_ylim(-6.0, 6.0)
    
    fig = plt.gcf()
    fig.canvas.draw()
    fig_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_array = fig_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig_array = cv2.cvtColor(fig_array, cv2.COLOR_RGB2BGR)
    fig_array = cv2.resize(fig_array, (rgbd_vis_image.shape[1], rgbd_vis_image.shape[0]))
    
    vis_image = np.concatenate((rgbd_vis_image,fig_array),axis=0)
    return vis_image    


def tra_gene(cfg, rgb, depth, save_path, seed = 45):

    # 设置随机种子
    import torch, random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

    # 构造相机内参矩阵
    def create_intrinsic(hfov_deg, width, height):
        hfov_rad = math.radians(hfov_deg)
        fx = width / (2 * math.tan(hfov_rad / 2))
        fy = fx * (height / width)
        cx = width / 2
        cy = height / 2

        K = np.array([
            [fx,  0, cx],
            [0,  fy, cy],
            [0,   0,  1]
        ])
        return K
    
    intrinsic = create_intrinsic(cfg["sensor_rgb"]["hfov"], cfg["sensor_rgb"]["resolution"][1], cfg["sensor_rgb"]["resolution"][0])
    pointnav_reset(save_path, intrinsic=intrinsic)
    rgb_images = np.array([rgb])
    depth_images = np.array([depth])

    # image_writer = imageio.get_writer(save_path + "current_trajectory.mp4", fps=10)

    trajectory,all_trajectory,all_value = nogoal_step(rgb_images,depth_images)
    # vis_image = bev_visualize(intrinsic,rgb,depth,trajectory[0],all_trajectory[0],all_value[0])

    # image_writer.append_data(cv2.cvtColor(vis_image,cv2.COLOR_BGR2RGB))

    # image_writer.close()
    
    return trajectory,all_trajectory,all_value, intrinsic

    



    
    
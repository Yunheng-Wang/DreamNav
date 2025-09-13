from PIL import Image
from .src.policy_agent import NavDP_Agent
import numpy as np
import cv2
import imageio
import time
import datetime
import json
import os
import io


from PIL import Image, ImageDraw, ImageFont
import argparse

navdp_navigator = None
navdp_fps_writer = None


#  图像可视化函数
def visualize(goal,image,image_step,depth,depth_step):
    vis_depth = np.tile(depth,(1,1,3))
    vis_depth = (vis_depth - vis_depth.min()) / (vis_depth.max() - vis_depth.min() + 1e-6)
    vis_depth = (vis_depth * 255).astype(np.uint8)
    vis_image = image.copy()
    vis_image[20:40,20:270] = 0
    
    concat_image = Image.fromarray(np.concatenate((vis_image,vis_depth),axis=1))
    image_draw = ImageDraw.Draw(concat_image)
    font = ImageFont.truetype('DejaVuSansMono.ttf',16)
    text1 = "PointGoal: {}".format(np.round(goal,decimals=1).tolist())
    image_draw.text((20,20),text1, font=font, fill=(255,255,255))
        
    text2 = str(image_step)
    image_draw.text((20,40),text2, font=font, fill=(255,255,255))
    
    text3 = str(depth_step)
    image_draw.text((20,60),text3, font=font, fill=(255,255,255))

    return_image = np.asarray(concat_image).copy()
    cv2.imwrite("navdp-pred.png",return_image)
    return return_image

def navdp_reset(intrinsic, threshold, batchsize, save_path):

    # 告诉 Python 要在函数里面使用和修改这两个全局变量
    global navdp_navigator,navdp_fps_writer

    # 获取请求中的 JSON 数据
    intrinsic = np.array(intrinsic)
    threshold = np.array(threshold)
    batchsize = np.array(batchsize)
    
    # 如果导航模型还没初始化，就创建一个新的导航模型 or 如果已经存在，就重置它
    if navdp_navigator is None:
        navdp_navigator = NavDP_Agent(intrinsic,
                                image_size=224,
                                memory_size=8,
                                predict_size=24,
                                temporal_depth=16,
                                heads=8,
                                token_dim=384,
                                stop_threshold=threshold,
                                navi_model="/home/dreams/Users/yunhengwang/vln/trajectory/checkpoints/navdp-weights.ckpt",
                                device='cuda:0')
        navdp_navigator.reset(batchsize)
    else:
        navdp_navigator.reset(batchsize)

    # 如果视频写入器还没有被创建，就创建一个新的 or 如果已经存在，就关闭它并创建一个新的
    if navdp_fps_writer is None:
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        navdp_fps_writer = imageio.get_writer(save_path + "{}_egocentric_trajectory.mp4".format(format_time),fps=7)
    else:
        navdp_fps_writer.close()
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        navdp_fps_writer = imageio.get_writer(save_path + "{}_egocentric_trajectory.mp4".format(format_time),fps=7)
    
    
# 当发送一个 POST 请求，路径是 /navdp_reset_env 运行 navdp_reset_env() 函数
def navdp_reset_env(env_id):
    global navdp_navigator
    navdp_navigator.reset_env(env_id)



def navdp_step_xy(files, data):
    global navdp_navigator,navdp_fps_writer
    start_time = time.time()

    # 获取请求数据流
    image_bytes = files['image'][1]
    depth_bytes = files['depth'][1]
    image = Image.open(io.BytesIO(image_bytes))
    depth = Image.open(io.BytesIO(depth_bytes))

    goal_data = json.loads(data['goal_data'])
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x,goal_y,np.ones_like(goal_x)),axis=1)
    batch_size = goal.shape[0]
    
    # 预处理rgb
    phase1_time = time.time()
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    # 预处理depth
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    # 
    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_pointgoal(goal,image,depth)
    phase3_time = time.time()
    navdp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))
    return {'trajectory': execute_trajectory.tolist(),
                    'all_trajectory': all_trajectory.tolist(),
                    'all_values': all_values.tolist()}


def navdp_step_nogoal(files, data):
    global navdp_navigator,navdp_fps_writer
    start_time = time.time()

    image_bytes = files['image'][1]
    depth_bytes = files['depth'][1]
    image = Image.open(io.BytesIO(image_bytes))
    depth = Image.open(io.BytesIO(depth_bytes))

    goal_data = json.loads(data['goal_data'])
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x,goal_y,np.ones_like(goal_x)),axis=1)
    batch_size = goal.shape[0]
    
    phase1_time = time.time()
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_nogoal(image,depth)
    phase3_time = time.time()
    navdp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))
    return {'trajectory': execute_trajectory.tolist(),
                    'all_trajectory': all_trajectory.tolist(),
                    'all_values': all_values.tolist()}


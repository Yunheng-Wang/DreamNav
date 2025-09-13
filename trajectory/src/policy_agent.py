import torch
import numpy as np
import cv2
from PIL import Image
from matplotlib import colormaps as cm
from .policy_network import NavDP_Policy_DPT

class NavDP_Agent:
    def __init__(self,
                 image_intrinsic,
                 image_size=224,
                 memory_size=8,
                 predict_size=24,
                 temporal_depth=16,             # 模型在每次推理时，处理“时间维度”上的多少帧图像
                 heads=8,                       # 模型的注意力头数
                 token_dim=384,
                 stop_threshold=-2.0,
                 navi_model = "./100.ckpt",
                 device='cuda:0'):
        self.image_intrinsic = image_intrinsic  # 相机内参
        self.device = device                    # 设备
        self.predict_size = predict_size        # 模型每次预测出的轨迹点数量
        self.image_size = image_size            # 所有输入的图像会被缩放成 224×224
        self.stop_threshold = stop_threshold    # 设定一个停止导航的门槛值
        self.memory_size = memory_size          # 模型在导航过程中最多记住多少帧图像作为历史记忆
        self.navi_former = NavDP_Policy_DPT(image_size,memory_size,predict_size,temporal_depth,heads,token_dim,device) # 初始化整个导航模型的架构
        self.navi_former.load_state_dict(torch.load(navi_model,map_location=self.device))                              # 加载预训练模型参数
        self.navi_former.to(self.device)                                                                               # 指定设备
        self.navi_former.eval()                                                                                        # 设置模型为评估模式
    
    def reset(self,batch_size):
        self.memory_queue = [[] for i in range(batch_size)]
    def reset_env(self,i):
        self.memory_queue[i] = []
    
    def project_trajectory(self,images,n_trajectories,n_values):
        trajectory_masks = []
        for i in range(images.shape[0]):
            trajectory_mask = np.array(images[i])
            n_trajectory = n_trajectories[i,:,:,0:2]
            n_value = n_values[i]
            for waypoints,value in zip(n_trajectory,n_value):
                norm_value = np.clip(-value*0.1,0,1)
                colormap = cm.get('jet')
                color = np.array(colormap(norm_value)[0:3]) * 255.0
                input_points = np.zeros((waypoints.shape[0],3)) - 0.2
                input_points[:,0:2] = waypoints
                input_points[:,1] = -input_points[:,1]
                camera_z = images[0].shape[0] - 1 - self.image_intrinsic[1][1] * input_points[:,2] / (input_points[:,0] + 1e-8) - self.image_intrinsic[1][2]
                camera_x = self.image_intrinsic[0][0] * input_points[:,1] / (input_points[:,0] + 1e-8) + self.image_intrinsic[0][2]
                
                for i in range(camera_x.shape[0]-1):
                    try:
                        if camera_x[i] > 0 and camera_z[i] > 0 and camera_x[i+1] > 0 and camera_z[i+1] > 0:
                            trajectory_mask = cv2.line(trajectory_mask,(int(camera_x[i]),int(camera_z[i])),(int(camera_x[i+1]),int(camera_z[i+1])),color.astype(np.uint8).tolist(),5)
                    except:
                        pass
                
            trajectory_masks.append(trajectory_mask)
        return np.concatenate(trajectory_masks,axis=1)

    def process_image(self,images):
        assert len(images.shape) == 4
        H,W,C = images.shape[1],images.shape[2],images.shape[3]
        prop = self.image_size/max(H,W)
        return_images = []
        for img in images:
            resize_image = cv2.resize(img,(-1,-1),fx=prop,fy=prop)
            pad_width = max((self.image_size - resize_image.shape[1])//2,0)
            pad_height = max((self.image_size - resize_image.shape[0])//2,0)
            pad_image = np.pad(resize_image,((pad_height,pad_height),(pad_width,pad_width),(0,0)),mode='constant',constant_values=0)
            resize_image = cv2.resize(pad_image,(self.image_size,self.image_size))
            resize_image = np.array(resize_image)
            resize_image = resize_image.astype(np.float32) / 255.0
            return_images.append(resize_image)
        return np.array(return_images)

    def process_depth(self,depths):
        assert len(depths.shape) == 4
        depths[depths==np.inf] = 0
        H,W,C = depths.shape[1],depths.shape[2],depths.shape[3]
        prop = self.image_size/max(H,W)
        return_depths = []
        for depth in depths:
            resize_depth = cv2.resize(depth,(-1,-1),fx=prop,fy=prop)
            pad_width = max((self.image_size - resize_depth.shape[1])//2,0)
            pad_height = max((self.image_size - resize_depth.shape[0])//2,0)
            pad_depth = np.pad(resize_depth,((pad_height,pad_height),(pad_width,pad_width)),mode='constant',constant_values=0)
            resize_depth = cv2.resize(pad_depth,(self.image_size,self.image_size))
            resize_depth[resize_depth>5.0] = 0
            resize_depth[resize_depth<0.1] = 0
            return_depths.append(resize_depth[:,:,np.newaxis])
        return np.array(return_depths)
    
    # batch_inference
    def process_pointgoal(self,goals):
        clip_goals = goals.clip(-10,10)
        clip_goals[:,0] = np.clip(clip_goals[:,0],0,10)
        return clip_goals
    
    def step_pointgoal(self,goals,images,depths):
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        input_images = []
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size:
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
                input_image = np.pad(input_image,((self.memory_size - input_image.shape[0],0),(0,0),(0,0),(0,0)))
            else:
                del self.memory_queue[i][0]
                self.memory_queue[i].append(process_images[i])    
                input_image = np.array(self.memory_queue[i])
                
            input_images.append(input_image)
        input_image = np.array(input_images)
        
        cv2.imwrite("input_image.jpg",np.concatenate(self.memory_queue[0],axis=0)*255)
        input_depth = process_depths
        input_goals = self.process_pointgoal(goals)
        all_trajectory, all_values, good_trajectory, bad_trajectory = self.navi_former.predict_pointgoal_action(input_goals,input_image,input_depth)
        
        if all_values.max() < self.stop_threshold:
            good_trajectory[:,:,:,0:2] = good_trajectory[:,:,:,0:2] * 0.0
        
        trajectory_mask = self.project_trajectory(images,all_trajectory,all_values) 
        return good_trajectory[:,0], all_trajectory, all_values, trajectory_mask
    
    def step_nogoal(self,images,depths):
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        input_images = []
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size:
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
                input_image = np.pad(input_image,((self.memory_size - input_image.shape[0],0),(0,0),(0,0),(0,0)))
            else:
                del self.memory_queue[i][0]
                self.memory_queue[i].append(process_images[i])    
                input_image = np.array(self.memory_queue[i])
                
            input_images.append(input_image)
        input_image = np.array(input_images)
        
        # cv2.imwrite("input_image.jpg",np.concatenate(self.memory_queue[0],axis=0)*255)
        input_depth = process_depths
        all_trajectory, all_values, good_trajectory, bad_trajectory = self.navi_former.predict_nogoal_action(input_image,input_depth)
        
    

        if all_values.max() < self.stop_threshold:
            good_trajectory[:,:,:,0:2] = good_trajectory[:,:,:,0:2] * 0.0
        
        # ###### 修改 #######
        trajectory_mask = self.project_trajectory(images,all_trajectory,all_values) 
        return good_trajectory[:,0], all_trajectory, all_values, trajectory_mask
    
  
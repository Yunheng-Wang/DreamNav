import re
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import cv2
import imageio
import datetime
import time
import json
import torch
import gc

from util.loader import load_config, load_task, load_video_as_frames
from util.image import get_current_rgb_depth, visualize_trajectories
from sim.init import environment
from sim.action import action_trajectory, action_point
from trajectory.navdp_client import tra_gene
from anticipate.my import trajectory_prediction
from foundation_model.api import reasoning_mllm,reasoning_llm, reasoning_video
from foundation_model.preprocess import instruction_prompt,navigator_prompt, progress_prompt, observation_prompt
from foundation_model.prompt import IMAGEINATIVE_DESCRIBLE
from anticipate.stable_virtual_camera.demo import Model
from trajectory.filter.diversity import filtering_trajectories
from util.tool import compute_passable_proportions, compute_passable_proportions_mask, video_extraction
from util.save import save_imagine, content_to_txt
from semantic_segmentation.inference_samples import semantic_infer
from Inference import segmentation_mask



if __name__ == "__main__": 
    
    task_type = "opennav"
    cfg = load_config("config.yaml")
    task = load_task(cfg, task_type)

    record = {} # 记录每个任务的导航过程
    
    for sig_task in task:
        if sig_task["episode_id"] != 546:
            continue

        # --------------------- 重复跳过 ---------------------
        save_path = "picture/" + task_type + "/" +  os.path.splitext(os.path.basename(sig_task["scene_id"]))[0] + "_" + str(sig_task["episode_id"]) + "/" 
        record_path = save_path  + "trajectory.json"
        if os.path.exists(record_path):
            continue
        


        # --------------------- 初始化 ---------------------
        record[sig_task["episode_id"]] = []
        nav_step = 0            # 记录导航步数
        history = {}            # 记录每一步的图像 {0：(rgb, trajectory_rgb), 1：(rgb, trajectory_rgb), ...}
        imagine = {}            # 记录每一步的想象 {0：{0：([rgb, ...], trajetory describle), 1：([rgb, ...], trajetory describle), 2：([rgb, ...], trajetory describle), 3：([rgb, ...], trajetory describle)}, ....}
        os.makedirs(save_path, exist_ok=True) 
        # 记录大模型思考
        with open(save_path + "thinking.txt", "w", encoding="utf-8") as f:
            pass
        content_to_txt("--------------- Task Overview ---------------", save_path)
        content_to_txt("Instruction: " + sig_task["instruction"]["instruction_text"], save_path)
        content_to_txt("Scene: " + str(os.path.splitext(os.path.basename(sig_task["scene_id"]))[0]), save_path)
        content_to_txt("episode_id: " + str(sig_task["episode_id"]), save_path)



        # --------------------- 环境加载 ---------------------
        sim, agent, agent_cfg = environment(cfg, sig_task["scene_id"], 0, sig_task["start_position"],sig_task["start_rotation"])
        


        action_point(sim, agent_cfg, 0, 1.57, observation = False)

        action_point(sim, agent_cfg, 1.9, 0, observation = False)
        action_point(sim, agent_cfg, 1, 0.87, observation = False)
        action_point(sim, agent_cfg, 0, 3.6, observation = False)
        action_point(sim, agent_cfg, 0.62, 0, observation = False)
        action_point(sim, agent_cfg, 0, -0.35, observation = False)


        # example 1
        """
        action_point(sim, agent_cfg, 0.3, -0.1, observation = False)
        action_point(sim, agent_cfg, 0.3, -0.1, observation = False)
        action_point(sim, agent_cfg, 0.3, -0.1, observation = False)
        action_point(sim, agent_cfg, 0.3, -0.1, observation = False)
        action_point(sim, agent_cfg, 0.3, -0.1, observation = False)
        action_point(sim, agent_cfg, 0.3, -0.1, observation = False)
        action_point(sim, agent_cfg, 0.3, -0.1, observation = False)
        action_point(sim, agent_cfg, 0.3, -0.1, observation = False)
        action_point(sim, agent_cfg, 0.3, -0.1, observation = False)
        action_point(sim, agent_cfg, 0.3, -0.1, observation = False)
        """


        # example 2
        action_point(sim, agent_cfg, 0.3, 0.05, observation = False)
        action_point(sim, agent_cfg, 0.3, 0.05, observation = False)
        action_point(sim, agent_cfg, 0.3, 0.05, observation = False)
        action_point(sim, agent_cfg, 0.3, 0.05, observation = False)
        action_point(sim, agent_cfg, 0.3, 0.05, observation = False)
        action_point(sim, agent_cfg, 0.3, 0.05, observation = False)
        action_point(sim, agent_cfg, 0.3, 0.05, observation = False)
        action_point(sim, agent_cfg, 0.3, 0.05, observation = False)
        action_point(sim, agent_cfg, 0.3, 0.05, observation = False)
        action_point(sim, agent_cfg, 0.3, 0.05, observation = False)
        
        rgb,depth,_, _ , _ = get_current_rgb_depth(sim, save_path, save=True)

        
        
        """
        import matplotlib.pyplot as plt
        rgb,depth,_, _ , _ = get_current_rgb_depth(sim, save_path, save=True)
        accessible_area_mask = segmentation_mask(rgb, save_path)
        mask = accessible_area_mask.astype(int)
        plt.imshow(mask, cmap = "gray", interpolation = "nearest")
        plt.axis("off")
        plt.savefig("mask.png", bbox_inches="tight",pad_inches=0)
        plt.close()
        """
        

  


        












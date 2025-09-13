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
import random

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
    
    task_type = "dreamnav_ablation"
    cfg = load_config("config.yaml")

    task = load_task(cfg, task_type)

    record = {} # 记录每个任务的导航过程

    for sig_task in task:

        # --------------------- 重复跳过 ---------------------
        save_path = "cache/" + task_type + "/" +  os.path.splitext(os.path.basename(sig_task["scene_id"]))[0] + "_" + str(sig_task["episode_id"]) + "/" 
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
        rgb,depth,_, _ , _ = get_current_rgb_depth(sim, save_path, save=True)
        accessible_area_mask = segmentation_mask(rgb, save_path)
        # EMSFormer 
        # semantic,semantic_category_map = semantic_infer(rgb[..., :3], depth)
        record[sig_task["episode_id"]].append(sim.get_agent(0).get_state().position) # 记录初始位置
        

        # --------------------- 指令拆解 ---------------------
        subtasks_index = 0
        ins_prompt = instruction_prompt(sig_task["instruction"]["instruction_text"])
        res_ins = reasoning_llm(cfg, ins_prompt)
        subtasks = [task.strip() for task in res_ins.split(';') if task.strip()]
        content_to_txt("Instruction Decomposition Result: " + res_ins, save_path)
        max_step = len(subtasks)+2           # 最大步数 -> 可以容忍 2 次进度评估为0
        content_to_txt("Max Steps of Current Navigation: " + str(max_step), save_path)
        

        # --------------------- 导航开始 ---------------------
        while(True):
            if (nav_step >= max_step) or (subtasks_index >= len(subtasks)):
                break

            history[nav_step] = [None,None]
            imagine[nav_step] = {}
            content_to_txt("--------------- Navigation Step: {} ---------------".format(nav_step), save_path)


            # --------------------- 视角微调矫正模块 ---------------------
            content_to_txt("@@@@@ Adjusting the perspective @@@@@", save_path)
            
            content_to_txt("Initial view correction:", save_path)
            if nav_step == 0:
                n = 0
                while(True):
                    if n >=4:
                        break
                    obs_prompt = observation_prompt(sig_task["instruction"]["instruction_text"])
                    adjust_view = reasoning_mllm(cfg, [rgb], obs_prompt)
                    adjust_view_matches = list(re.finditer(r'decision', adjust_view))
                    adjust_view_last_match = adjust_view_matches[-1]
                    adjust_view_choose = adjust_view[adjust_view_last_match.end():].strip()
                    content_to_txt("Thinking: " + adjust_view, save_path)
                    if "1" in adjust_view_choose:
                        action_point(sim, agent_cfg, 0, 1.57, observation = False) # 90度
                        content_to_txt("choose: turn 90 degree", save_path)
                        rgb,depth,_, _ , _ = get_current_rgb_depth(sim, save_path, save=True)
                        accessible_area_mask = segmentation_mask(rgb, save_path)
                        n += 1
                    else: 
                        content_to_txt("choose: the current perspective is perfect.", save_path)
                        break
            
            # if nav_step != 0:
            #     content_to_txt("Adjusting the perspective:", save_path)
            #     total_ratio, left_ratio, right_ratio = compute_passable_proportions_mask(accessible_area_mask)
            #     if total_ratio >= 0.1:
            #         rgb,depth,_, _ , _ = get_current_rgb_depth(sim, save_path, save=True)
            #         accessible_area_mask = segmentation_mask(rgb, save_path)
            #         # semantic,semantic_category_map = semantic_infer(rgb[..., :3], depth)
            #     else:
            #         if left_ratio <= right_ratio:
            #             count = 0
            #             while(count < 2):
            #                 action_point(sim, agent_cfg, 0, -0.523, observation = False)
            #                 content_to_txt("right - 30 degrees", save_path)
            #                 rgb,depth,_, _ , _ = get_current_rgb_depth(sim, save_path, save=True)
            #                 accessible_area_mask = segmentation_mask(rgb, save_path)
            #                 # semantic,semantic_category_map = semantic_infer(rgb[..., :3], depth)
            #                 total_ratio, left_ratio, right_ratio = compute_passable_proportions_mask(accessible_area_mask)
            #                 count += 1
            #                 if total_ratio >= 0.1:
            #                     break
            #         else:
            #             count = 0
            #             while(count < 2):
            #                 action_point(sim, agent_cfg, 0, 0.523, observation = False)
            #                 content_to_txt("left - 30 degrees", save_path)
            #                 rgb,depth,_, _ , _ = get_current_rgb_depth(sim, save_path, save=True)
            #                 accessible_area_mask = segmentation_mask(rgb, save_path)
            #                 # semantic,semantic_category_map = semantic_infer(rgb[..., :3], depth)
            #                 total_ratio, left_ratio, right_ratio = compute_passable_proportions_mask(accessible_area_mask)
            #                 count += 1
            #                 if total_ratio >= 0.1:
            #                     break
            
            # 记录当前视角
            history[nav_step][0] = rgb


            # --------------------- 完成度判定模块 ---------------------
            content_to_txt("@@@@@ Progress Evaluation @@@@@", save_path)
            if nav_step != 0:
                # 第一步不做判断
                if subtasks_index + 1 < len(subtasks):
                    # 判断执行上一步的轨迹的完成度
                    estimate_progress_prompt = progress_prompt([subtasks[subtasks_index], subtasks[subtasks_index + 1]])
                    while (True):
                        try:
                            estimate_progress = reasoning_mllm(cfg, [history[nav_step-1][1], history[nav_step][0]], estimate_progress_prompt)
                            completion_matches = list(re.finditer(r'completion_status', estimate_progress))
                            completion_last_match = completion_matches[-1]
                            completeness = estimate_progress[completion_last_match.end():].strip()
                            if ("0" in completeness) or ("2" in completeness) or ("1" in completeness):
                                content_to_txt("Thought on Task Completion Rate: " + estimate_progress, save_path)
                                break
                            else:
                                raise ValueError(f"Invalid completion_status value: {completeness}")
                    
                        except Exception as e:
                            print(f"[Navigator Retry] Error occurred: {e}")
                            continue  # 重新尝试

                    if "0" in completeness:
                        content_to_txt("This step has not completed the sub-tasks.", save_path)
                        pass
                    elif "2" in completeness:
                        content_to_txt("This step involves completing two sub-tasks.", save_path)
                        subtasks_index += 2
                        # 两个都完成直接结束任务
                        if subtasks_index >= len(subtasks):
                            break
                    else:
                        content_to_txt("This step involves completing one sub-tasks.", save_path)
                        subtasks_index += 1
                else:
                    # 如果上一步已经执行最后一个子任务，不做任何判断 直接跳出
                    break
            else:
                content_to_txt("Currently, it is the initial position of the navigation. No completion check is required.", save_path)
            content_to_txt("The currently executing subtask: " +  subtasks[subtasks_index], save_path)


            # --------------------- 轨迹生成模块 ---------------------
            _,all_trajectory_local,_, intrinsic = tra_gene(cfg, rgb[..., :3], depth, save_path)
            all_trajectory_local = filtering_trajectories(all_trajectory_local, cfg["hyper_parameter"]["num_trajectory"])                     # 基于距离矩阵的最大散度贪心选择过滤
            filter_trajectory_image = visualize_trajectories(intrinsic, rgb , all_trajectory_local, save_path, "filter_egocentric_trajectory")
            

            
            # --------------------- 轨迹想象模块 ---------------------
            content_to_txt("@@@@@ Imagine @@@@@", save_path)
            traj_idx = 0
            model = Model()
            for sig_trajectory in all_trajectory_local[0]:
                # 执行想象
                trajectory_prediction(sig_trajectory, save_path, model, cfg["hyper_parameter"]["imagine_resolution"])
                frames = load_video_as_frames(save_path)
                # 想象描述
                inageinative_describle_prompt = IMAGEINATIVE_DESCRIBLE
                frames_for_mllm = video_extraction(frames, cfg["hyper_parameter"]["imagine_step"])
                while(True):
                    try:
                        traj_desc = reasoning_video(cfg, frames_for_mllm, inageinative_describle_prompt)
                        break
                    except Exception as e:
                        print(f"[Navigator Retry] Error occurred: {e}")
                        continue  # 重新尝试
                # 保存
                content_to_txt("trajectory {} : {}".format(traj_idx, traj_desc), save_path)
                save_imagine(nav_step, traj_idx, frames, frames_for_mllm, save_path)
                # 缓存
                imagine[nav_step][traj_idx] = (frames,traj_desc)

                traj_idx += 1


            # --------------------- 导航推理模块 ---------------------
            content_to_txt("@@@@@ Navigator @@@@@", save_path)
            navigator_prompt_input = navigator_prompt(subtasks[subtasks_index], imagine[nav_step])
            while (True):
                try:
                    navigator_decision = reasoning_llm(cfg, navigator_prompt_input)
                    matches = list(re.finditer(r'choose', navigator_decision))
                    last_match = matches[-1]
                    choose = navigator_decision[last_match.end():].strip()
                    if ("0" in choose) or ("1" in choose) or ("2" in choose) or ("3" in choose):
                        content_to_txt("reasoning: {}".format(navigator_decision), save_path)
                        content_to_txt("choose trajectory id: {}".format(choose), save_path)
                        break
                    else:
                        raise ValueError(f"Invalid choose value: {choose}")
                except Exception as e:
                    print(f"[Navigator Retry] Error occurred: {e}")
                    continue  # 重新尝试


            # --------------------- 导航执行模块 ---------------------
            if "0" in choose:
                st = (cfg["hyper_parameter"]["imagine_step"])*2 - 1
                selected_trajectory_image = visualize_trajectories(intrinsic, rgb , all_trajectory_local[:, 0:1, :st, :], save_path, "selected_egocentric_trajectory")
                history[nav_step][1] = selected_trajectory_image
                record = action_trajectory(sim, agent_cfg, cfg, all_trajectory_local[0][0][:st], save_path, record, sig_task, video=True)
            elif "1" in choose:
                st = (cfg["hyper_parameter"]["imagine_step"])*2 - 1
                selected_trajectory_image = visualize_trajectories(intrinsic, rgb , all_trajectory_local[:, 1:2, :st, :], save_path, "selected_egocentric_trajectory")
                history[nav_step][1] = selected_trajectory_image
                record = action_trajectory(sim, agent_cfg, cfg, all_trajectory_local[0][1][:st], save_path, record, sig_task, video=True)
            elif "2" in choose:
                st = (cfg["hyper_parameter"]["imagine_step"])*2 - 1
                selected_trajectory_image = visualize_trajectories(intrinsic, rgb , all_trajectory_local[:, 2:3, :st, :], save_path, "selected_egocentric_trajectory")
                history[nav_step][1] = selected_trajectory_image
                record = action_trajectory(sim, agent_cfg, cfg, all_trajectory_local[0][2][:st], save_path, record, sig_task, video=True)
            else:
                st = (cfg["hyper_parameter"]["imagine_step"])*2 - 1
                selected_trajectory_image = visualize_trajectories(intrinsic, rgb , all_trajectory_local[:, 3:4, :st, :], save_path, "selected_egocentric_trajectory")
                history[nav_step][1] = selected_trajectory_image
                record = action_trajectory(sim, agent_cfg, cfg, all_trajectory_local[0][3][:st], save_path, record, sig_task, video=True)
            rgb,depth,_, _ , _ = get_current_rgb_depth(sim, save_path, save=True)
            accessible_area_mask = segmentation_mask(rgb, save_path)

            nav_step += 1


        # --------------------- 保存路径结果 ---------------------
        episode_id = sig_task["episode_id"]
        episode_record = record[episode_id]
        episode_record_serializable = [pos.tolist() for pos in episode_record]
        record_path = os.path.join(save_path, "trajectory.json")
        with open(record_path, "w") as f:
            json.dump({
                "episode_id": episode_id,
                "trajectory": episode_record_serializable
            }, f, indent=2)
        

        sim.close()
        del sim
        del model
        torch.cuda.empty_cache()
        gc.collect()


        












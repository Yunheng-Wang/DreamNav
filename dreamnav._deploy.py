import subprocess
import os
from PIL import Image
import numpy as np
import cv2
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
import math

from util.save import content_to_txt
from util.loader import load_config, load_task, load_video_as_frames
from util.image import get_current_rgb_depth, visualize_trajectories
from sim.init import environment
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

import threading
import time
import traceback

def _obs_worker(stop_event, robot_user, robot_ip, local_save_dir, interval_sec=0.5):
    """
    后台循环抓取一帧（RGB/Depth），直到 stop_event 被置位。
    interval_sec: 控制抓取频率，避免IO/网络过载；0.3~1.0s 都可。
    """
    while not stop_event.is_set():
        try:
            # 抓1帧并落盘（你已有时间戳保存逻辑）
            get_obervations(robot_user, robot_ip, local_save_dir)
        except Exception as e:
            # 打印但不中断循环，避免偶发失败导致线程退出
            print("[obs_worker] error:", repr(e))
            traceback.print_exc()
        finally:
            # 控制抓取频率
            stop_event.wait(interval_sec)

def run_action_with_observer(
    trajectory,
    dis_speed,
    ang_speed,
    user,
    host,
    script,
    python_bin,
    robot_user,
    robot_ip,
    local_save_dir,
    grab_interval=0.5
):
    """
    在 action_trajectory 执行期间，并行持续抓取第一视角，结束后自动关闭。
    """
    stop_event = threading.Event()
    t = threading.Thread(
        target=_obs_worker,
        args=(stop_event, robot_user, robot_ip, local_save_dir, grab_interval),
        daemon=True,
    )
    t.start()
    try:
        # 前台：执行你的轨迹
        action_trajectory(trajectory, dis_speed, ang_speed, user, host, script, python_bin=python_bin)
    finally:
        # 无论成功或异常，都确保停止抓取并回收线程
        stop_event.set()
        t.join(timeout=5.0)

def get_obervations(robot_user,robot_ip,local_save_dir):
    # 定义脚本路径
    script_dir = "/home/dreams/Users/yunhengwang/vln/deploy"
    script_path = os.path.join(script_dir, "get_rgb_depth.sh")
    subprocess.run(["chmod", "+x", script_path], check=True)
    subprocess.run([script_path, robot_user, robot_ip, local_save_dir],cwd=script_dir,check=True)

    rgb = Image.open(local_save_dir + "current_rgb.png").convert("RGB")
    rgb = np.array(rgb)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(local_save_dir, "his")
    os.makedirs(save_dir, exist_ok=True)  #
    save_path = os.path.join(save_dir, f"rgb_{timestamp}.png")
    Image.fromarray(rgb).save(save_path)

    depth = cv2.imread(local_save_dir + "current_depth.png", cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 1000.0
    return rgb, depth

import subprocess
import shlex
import json
import math

import subprocess, shlex, json, math

def action_trajectory(trajectory, dis_speed, ang_speed, user, host, script, python_bin="python"):
    plan = []
    prior = (0.0, 0.0, 0.0)
    for x, y, yaw in trajectory:
        dist = math.hypot(x - prior[0], y - prior[1])
        dyaw = yaw - prior[2]
        plan.append([dist, dyaw])
        prior = (x, y, yaw)

    plan_str = json.dumps(plan, separators=(",", ":"))

    remote = (
            f"source /opt/ros/melodic/setup.bash >/dev/null 2>&1 && "
            f"{shlex.quote(python_bin)} {shlex.quote(script)} "
            f'--plan "{plan_str}" '
            f"--lin {dis_speed} "
            f"--ang_speed {ang_speed}"
        )

    # 最终 ssh 命令（与手动那条一致）
    cmd = f"ssh {shlex.quote(user)}@{shlex.quote(host)} '{remote}'"


    # 直接运行，无需 shell=True
    res = subprocess.run(cmd, shell=True, text=True, capture_output=True)






def rotate(angle, ang_speed,user, host,script):
    cmd = (
        f'ssh {user}@{host} '
        f'"source {"/opt/ros/melodic/setup.bash"} && {"python"} {script} --angle {angle} --ang_speed {ang_speed}"'
    )

    result = subprocess.run(cmd, shell=True, text=True,
                            capture_output=True)




if __name__ == "__main__": 
    save_name = "classroom_0" 
    instruction = "Stop beside the potted plant in the black vase"

    ssh_host = "10.12.143.140"
    ssh_user = "agilex"
    rotate_in_place_path = "/home/agilex/Desktop/rotate_in_place.py"
    run_trajectory_path = "/home/agilex/Desktop/run_trajectory.py"
    angle_speed = 0.5236 
    distance_speed = 0.2

    # --------------------- 初始化 ---------------------
    save_path = "/home/dreams/Users/yunhengwang/vln/deploy/cache/" + save_name + "/"
    record = []
    nav_step = 0            # 记录导航步数
    history = {}            # 记录每一步的图像 {0：(rgb, trajectory_rgb), 1：(rgb, trajectory_rgb), ...}
    imagine = {}            # 记录每一步的想象 {0：{0：([rgb, ...], trajetory describle), 1：([rgb, ...], trajetory describle), 2：([rgb, ...], trajetory describle), 3：([rgb, ...], trajetory describle)}, ....}
    cfg = load_config("config.yaml")
    os.makedirs(save_path, exist_ok=True) 
    # 记录大模型思考
    with open(save_path + "thinking.txt", "w", encoding="utf-8") as f:
        pass
    content_to_txt("--------------- Task Overview ---------------", save_path)
    content_to_txt("Instruction: " + instruction, save_path)

    rgb, depth = get_obervations(ssh_user,ssh_host,save_path)
    accessible_area_mask = segmentation_mask(rgb, save_path)

    # --------------------- 指令拆解 ---------------------
    subtasks_index = 0
    ins_prompt = instruction_prompt(instruction)
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
                obs_prompt = observation_prompt(instruction)
                adjust_view = reasoning_mllm(cfg, [rgb], obs_prompt)
                adjust_view_matches = list(re.finditer(r'decision', adjust_view))
                adjust_view_last_match = adjust_view_matches[-1]
                adjust_view_choose = adjust_view[adjust_view_last_match.end():].strip()
                content_to_txt("Thinking: " + adjust_view, save_path)
                if "1" in adjust_view_choose:

                    rotate(1.57, angle_speed, ssh_user, ssh_host, rotate_in_place_path) 

                    content_to_txt("choose: turn 90 degree", save_path)
                    rgb, depth = get_obervations(ssh_user,ssh_host,save_path)
                    accessible_area_mask = segmentation_mask(rgb, save_path)
                    n += 1
                else: 
                    content_to_txt("choose: the current perspective is perfect.", save_path)
                    break
        
        if nav_step != 0:
            content_to_txt("Adjusting the perspective:", save_path)
            total_ratio, left_ratio, right_ratio = compute_passable_proportions_mask(accessible_area_mask)
            if total_ratio >= 0.1:
                rgb, depth = get_obervations(ssh_user,ssh_host,save_path)
                accessible_area_mask = segmentation_mask(rgb, save_path)
                # semantic,semantic_category_map = semantic_infer(rgb[..., :3], depth)
            else:
                if left_ratio <= right_ratio:
                    count = 0
                    while(count < 2):
                        rotate(-0.523, angle_speed, ssh_user, ssh_host, rotate_in_place_path) 
                        content_to_txt("right - 30 degrees", save_path)
                        rgb, depth = get_obervations(ssh_user,ssh_host,save_path)
                        accessible_area_mask = segmentation_mask(rgb, save_path)
                        # semantic,semantic_category_map = semantic_infer(rgb[..., :3], depth)
                        total_ratio, left_ratio, right_ratio = compute_passable_proportions_mask(accessible_area_mask)
                        count += 1
                        if total_ratio >= 0.1:
                            break
                else:
                    count = 0
                    while(count < 2):
                        rotate(-0.523, angle_speed, ssh_user, ssh_host, rotate_in_place_path)
                        content_to_txt("left - 30 degrees", save_path)
                        rgb, depth = get_obervations(ssh_user,ssh_host,save_path)
                        accessible_area_mask = segmentation_mask(rgb, save_path)
                        # semantic,semantic_category_map = semantic_infer(rgb[..., :3], depth)
                        total_ratio, left_ratio, right_ratio = compute_passable_proportions_mask(accessible_area_mask)
                        count += 1
                        if total_ratio >= 0.1:
                            break
        
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
            run_action_with_observer(
            trajectory=all_trajectory_local[0][0],
            dis_speed=distance_speed,
            ang_speed=angle_speed,
            user=ssh_user,
            host=ssh_host,
            script=run_trajectory_path,
            python_bin="python",
            robot_user=ssh_user,
            robot_ip=ssh_host,
            local_save_dir=save_path,     # 这里就是你 cache/<save_name>/ 的目录
            grab_interval=0.5             # 可调：0.3~1.0s
        )
        elif "1" in choose:
            st = (cfg["hyper_parameter"]["imagine_step"])*2 - 1
            selected_trajectory_image = visualize_trajectories(intrinsic, rgb , all_trajectory_local[:, 1:2, :st, :], save_path, "selected_egocentric_trajectory")
            history[nav_step][1] = selected_trajectory_image
            run_action_with_observer(
            trajectory=all_trajectory_local[0][1],
            dis_speed=distance_speed,
            ang_speed=angle_speed,
            user=ssh_user,
            host=ssh_host,
            script=run_trajectory_path,
            python_bin="python",
            robot_user=ssh_user,
            robot_ip=ssh_host,
            local_save_dir=save_path,     # 这里就是你 cache/<save_name>/ 的目录
            grab_interval=0.5             # 可调：0.3~1.0s
        )
        elif "2" in choose:
            st = (cfg["hyper_parameter"]["imagine_step"])*2 - 1
            selected_trajectory_image = visualize_trajectories(intrinsic, rgb , all_trajectory_local[:, 2:3, :st, :], save_path, "selected_egocentric_trajectory")
            history[nav_step][1] = selected_trajectory_image
            run_action_with_observer(
            trajectory=all_trajectory_local[0][2],
            dis_speed=distance_speed,
            ang_speed=angle_speed,
            user=ssh_user,
            host=ssh_host,
            script=run_trajectory_path,
            python_bin="python",
            robot_user=ssh_user,
            robot_ip=ssh_host,
            local_save_dir=save_path,     # 这里就是你 cache/<save_name>/ 的目录
            grab_interval=0.5             # 可调：0.3~1.0s
        )
        else:
            st = (cfg["hyper_parameter"]["imagine_step"])*2 - 1
            selected_trajectory_image = visualize_trajectories(intrinsic, rgb , all_trajectory_local[:, 3:4, :st, :], save_path, "selected_egocentric_trajectory")
            history[nav_step][1] = selected_trajectory_image
            run_action_with_observer(
            trajectory=all_trajectory_local[0][3],
            dis_speed=distance_speed,
            ang_speed=angle_speed,
            user=ssh_user,
            host=ssh_host,
            script=run_trajectory_path,
            python_bin="python",
            robot_user=ssh_user,
            robot_ip=ssh_host,
            local_save_dir=save_path,     # 这里就是你 cache/<save_name>/ 的目录
            grab_interval=0.5             # 可调：0.3~1.0s
        )

        rgb, depth = get_obervations(ssh_user,ssh_host,save_path)
        accessible_area_mask = segmentation_mask(rgb, save_path)

        nav_step += 1

    

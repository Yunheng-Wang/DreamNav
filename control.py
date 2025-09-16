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
    

    plan_str = json.dumps(trajectory, separators=(",", ":"))

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
    instruction = "Pass by the chair in front of you, then stop beside the red box."

    ssh_host = "10.12.143.140"
    ssh_user = "agilex"
    rotate_in_place_path = "/home/agilex/Desktop/rotate_in_place.py"
    run_trajectory_path = "/home/agilex/Desktop/run_trajectory.py"
    angle_speed = 0.5236 
    distance_speed = 0.2

    save_path = "/home/dreams/Users/yunhengwang/vln/deploy/cache/" + save_name + "/"

    # rotate(0.523, angle_speed, ssh_user, ssh_host, rotate_in_place_path) 
    rgb, depth = get_obervations(ssh_user,ssh_host,save_path)
    accessible_area_mask = segmentation_mask(rgb, save_path)

    run_action_with_observer(
            trajectory=[[0.2-0]],
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
    

    
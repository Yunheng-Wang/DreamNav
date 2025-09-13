import yaml
import json
import cv2


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_task(cfg, task_type):

    if task_type == "r2r":
        path = cfg["task"]["r2r_val_unseen"]
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)["episodes"]
         
    if task_type == "opennav":
        path = cfg["task"]["opennav_100"]
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)["episodes"]
    if task_type == "dreamnav_ablation":
        path = cfg["task"]["dreamnav_ablation"]
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)["episodes"]
            
    return data


def load_video_as_frames(video_path):
    cap = cv2.VideoCapture(video_path + "pred.mp4")
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 转换为 RGB 格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb) 

    cap.release()
    return frames

def load_trajectory_json(task_name, file_name):

    pass

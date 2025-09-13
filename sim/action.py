import math
import cv2
import time
import datetime
import os


def action_point(simulator, agent_cfg, distance, angle, observation = False):
    # angle : 弧度 (左转正数，右转负数)
    agent_cfg.action_space["move_forward"].actuation.amount = distance

    if angle >= 0: # turn left
        agent_cfg.action_space["turn_left"].actuation.amount = math.degrees(angle)
        simulator.step("turn_left")
    else:           # turn right
        agent_cfg.action_space["turn_right"].actuation.amount = -math.degrees(angle)
        simulator.step("turn_right")
    simulator.step("move_forward")

    if observation:

        obs_rgb = simulator.get_sensor_observations()["sensor_rgb"]

        return obs_rgb
    


def action_trajectory(simulator, agent_cfg, cfg, trajectory, save_path, record, sig_task, video = False):
    if video == True:
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d_%H-%M-%S") 
        os.makedirs(save_path + "action_video", exist_ok=True)
        vid = cv2.VideoWriter(save_path + "action_video/" + "{}.mp4".format(format_time), cv2.VideoWriter_fourcc(*'mp4v'), 1, (cfg["sensor_rgb"]["resolution"][1], cfg["sensor_rgb"]["resolution"][0]))
    
    sum = 0
    prior = [0,0,0]
    for point in trajectory:
        if video == True:
            obs_rgb = action_point(simulator, agent_cfg, math.sqrt((point[0]-prior[0])**2 + (point[1]-prior[1])**2), point[2] - prior[2], observation=True)
            sum += math.sqrt((point[0]-prior[0])**2 + (point[1]-prior[1])**2)
            record[sig_task["episode_id"]].append(simulator.get_agent(0).get_state().position) 
            prior = point
            vid.write(cv2.cvtColor(obs_rgb, cv2.COLOR_RGB2BGR))
        else:
            action_point(simulator, agent_cfg, math.sqrt((point[0]-prior[0])**2 + (point[1]-prior[1])**2), point[2] - prior[2], observation=False)
            sum += math.sqrt((point[0]-prior[0])**2 + (point[1]-prior[1])**2)
            record[sig_task["episode_id"]].append(simulator.get_agent(0).get_state().position) 
            prior = point
        
    if video == True:
        vid.release()
    print(sum)
    return record
    


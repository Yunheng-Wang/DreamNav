import os
import json
import argparse
from util.loader import load_config, load_task
import math
from sim.init import environment
import numpy as np
import habitat_sim

def geodesic_distance(sim, pos1, pos2):
    pf = sim.pathfinder
    p1 = np.array(pos1, dtype=np.float32)
    p2 = np.array(pos2, dtype=np.float32)

    if pf.is_navigable(p1) and pf.is_navigable(p2):
        sp = habitat_sim.ShortestPath()
        sp.requested_start = p1
        sp.requested_end = p2
        if pf.find_path(sp):
            return float(sp.geodesic_distance)

        # 失败后再试 snap
        snap_p1 = pf.snap_point(p1)
        snap_p2 = pf.snap_point(p2)
        snap_sp = habitat_sim.ShortestPath()
        snap_sp.requested_start = snap_p1
        snap_sp.requested_end = snap_p2
        found2 = pf.find_path(snap_sp)  
        if found2:
            return float(snap_sp.geodesic_distance)
        return float(np.linalg.norm(snap_p1 - snap_p2))  # 兜底用 snapped 欧氏

    else:
        snap_p1 = pf.snap_point(p1)
        snap_p2 = pf.snap_point(p2)
        sp = habitat_sim.ShortestPath()
        sp.requested_start = snap_p1
        sp.requested_end = snap_p2
        if pf.find_path(sp):
            return float(sp.geodesic_distance)
        return float(np.linalg.norm(snap_p1 - snap_p2))  # 兜底用 snapped 欧氏



def SR_count(simulator, goals_position, trajectory, thresh=3.0):
    if not trajectory:
        return 0
    dis = geodesic_distance(simulator, goals_position, trajectory[-1])
    return 1 if dis <= thresh else 0


def OSR_count(simulator, goals_position, trajectory, thresh=3.0):
    for pt in trajectory or []:
        dis = geodesic_distance(simulator, goals_position, pt)
        if dis <= thresh:
            return 1
    return 0


def NE_count(simulator, goal_position, trajectory):
    final_pos = trajectory[-1]
    return geodesic_distance(simulator, goal_position, final_pos)


def TL_count(trajectory):
    TL = 0
    for pt in range(len(trajectory)):
        if pt == 0:
            continue
        a = np.array(trajectory[pt], dtype=np.float32)
        b = np.array(trajectory[pt-1], dtype=np.float32)

        TL += float(np.linalg.norm(a - b))
    return TL


def SPL_count(reference_length, real_length, success):
    denom = max(reference_length, real_length)
    return success*(reference_length/denom)



def evaluate(result_path, config_path="config.yaml"):
    cfg = load_config(config_path)
    task = load_task(cfg)

    folder_list = [name for name in os.listdir(result_path) if os.path.isdir(os.path.join(result_path, name))]
    folders_with_traj = []
    for folder in folder_list:
        traj_path = os.path.join(result_path, folder, "trajectory.json")
        if os.path.exists(traj_path):
            folders_with_traj.append(folder)
   
    SR = 0
    OSR = 0
    NE = 0
    TL = 0
    SPL = 0
    result = {}
    total_num = len(folders_with_traj)
    if total_num == 0:
        raise RuntimeError(f"No trajectory.json files found under: {result_path}")
    for folder in folders_with_traj:
        scene, episode_id = folder.split("_", 1)
        episode_id = episode_id.rsplit("_",1)[0]
        
        result[folder] = {}
        current_task = None
        for sig_task in task:
            if (str(sig_task["episode_id"]) == str(episode_id)) and (str(sig_task["scene_id"]) == str('mp3d/' + scene + "/" + scene + ".glb")):
                current_task = sig_task

        traj_path = os.path.join(result_path, folder, "trajectory.json")
        with open(traj_path, "r") as f:
            traj_data = json.load(f)

        traj = traj_data["trajectory"]
        sim, _, _ = environment(cfg, current_task["scene_id"], 0, current_task["start_position"],current_task["start_rotation"])
        
        now_TL = TL_count(traj)
        result[folder]["TL"] = now_TL
        TL += now_TL
        
        now_NE = NE_count(sim,current_task["goals"][0]["position"],traj)
        result[folder]["NE"] = now_NE
        NE += now_NE

        now_OSR = OSR_count(sim,current_task["goals"][0]["position"],traj)
        result[folder]["OSR"] = now_OSR
        OSR += now_OSR

        now_SR = SR_count(sim,current_task["goals"][0]["position"],traj)
        result[folder]["SR"] = now_SR
        SR += now_SR     

        now_SPL = SPL_count(current_task["info"]["geodesic_distance"], now_TL, now_SR)
        result[folder]["SPL"] = now_SPL
        SPL += now_SPL
        sim.close()

    SR = (SR / total_num) *100
    OSR = (OSR / total_num) *100
    NE = (NE / total_num) 
    TL = (TL / total_num) 
    SPL = (SPL / total_num) *100

    
    print(f"SR:  {SR:.3f}%")
    print(f"OSR: {OSR:.3f}%")
    print(f"NE:  {NE:.3f} m")
    print(f"TL:  {TL:.3f} m")
    print(f"SPL: {SPL:.3f}%")

    result["summary"] = {
        "SR_percent": SR,
        "OSR_percent": OSR,
        "NE_mean_m": NE,
        "TL_mean_m": TL,
        "SPL_percent": SPL,
        "number": total_num
    }

    output_path = os.path.join(result_path, "eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved detailed evaluation to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DreamNav trajectories on R2R-CE.")
    parser.add_argument("--result-path", required=True, help="Directory containing episode result folders.")
    parser.add_argument("--config", default="config.yaml", help="Path to the DreamNav YAML configuration.")
    args = parser.parse_args()
    evaluate(args.result_path, args.config)

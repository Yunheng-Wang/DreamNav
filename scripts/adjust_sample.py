import json

def adjust(path):
    with open(path, "r") as f:
        traj_data = json.load(f)
    traj_data["trajectory"] = traj_data["trajectory"][:-17]

    save_path = path
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(traj_data, f, ensure_ascii=False, indent=2)



path = "/media/dreams/yunhengwang/DreamNav/temp/x8F5xyUWy9e_1665/trajectory.json"

adjust(path)
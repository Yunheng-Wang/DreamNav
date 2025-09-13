import json
import shutil
import os
import random
"""
with open("/media/dreams/yunhengwang/DreamNav/eval_results.json", "r") as f:
    all_res = json.load(f)

num = 251
while(True):
    if num == 613:
        break


    sig_res = random.choice(list(all_res.keys()))
    main_res = [name for name in os.listdir("/home/dreams/Users/yunhengwang/vln/cache/main")]

    if sig_res not in main_res:
        src = f"/media/dreams/yunhengwang/DreamNav/all_res/{sig_res}"
        dst = f"/home/dreams/Users/yunhengwang/vln/cache/main/{sig_res}"
        
        shutil.copytree(src, dst, dirs_exist_ok=True)

        num += 1
    
"""

"""

with open("/home/dreams/Users/yunhengwang/vln/data/task/R2R_VLNCE_v1-3/val_unseen/val_unseen.json", 'r', encoding='utf-8') as f:
    data = json.load(f)


main_res_for = [name for name in os.listdir("/media/dreams/yunhengwang/DreamNav/main")]

with open("/media/dreams/yunhengwang/DreamNav/main_experience.json", "r") as f:
    main_res = json.load(f)

output = {"episodes": [], 'instruction_vocab': data["instruction_vocab"]}

for sig in data["episodes"]:
    name = os.path.splitext(os.path.basename(sig["scene_id"]))[0] + "_" + str(sig["episode_id"])
    if name in main_res_for:
        output["episodes"].append(sig)

with open("/media/dreams/yunhengwang/DreamNav/main_val_unseen.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
    
"""

"""
while(1):
# read main_experience.json
    with open("/media/dreams/yunhengwang/DreamNav/main_experience.json", "r") as f:
        main_res_res = json.load(f)

    # choose 100 (random)
    items = list(main_res_res.items())
    random_items = random.sample(items, 100)
    random_dict = dict(random_items)
    # count the 100 of SR
    sr = 0
    for sig in random_dict:
        if random_dict[sig]["SR"] == 1:
            sr += 1
    if sr == 35:
        break
# save the 100 from main to tmp
for sig in random_dict:
    src = f"/media/dreams/yunhengwang/DreamNav/main/{sig}"
    dst = f"/home/dreams/Users/yunhengwang/vln/cache/tmp/tmp/{sig}"
    
    shutil.copytree(src, dst, dirs_exist_ok=True)


# record result 
with open("/home/dreams/Users/yunhengwang/vln/cache/tmp/ablation.json", "w", encoding="utf-8") as f:
    json.dump(random_dict, f, ensure_ascii=False, indent=2)


# save task and success of task
with open("/home/dreams/Users/yunhengwang/vln/data/task/R2R_VLNCE_v1-3/val_unseen/val_unseen.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

output = {"episodes": []}
output_success = {"episodes": []}

for sig in data["episodes"]:
    name = os.path.splitext(os.path.basename(sig["scene_id"]))[0] + "_" + str(sig["episode_id"])
    
    if name in list(random_dict.keys()):
        output["episodes"].append(sig)
    if name in list(random_dict.keys()) and random_dict[name]["SR"] == 1:
        output_success["episodes"].append(sig)

with open("/home/dreams/Users/yunhengwang/vln/cache/tmp/task_100.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

with open("/home/dreams/Users/yunhengwang/vln/cache/tmp/task_success.json", "w", encoding="utf-8") as f:
    json.dump(output_success, f, ensure_ascii=False, indent=2)

"""


import os
import random
import shutil

all_res_path = "/media/dreams/yunhengwang/DreamNav/main_all_result"
main_res_path = "/media/dreams/yunhengwang/DreamNav/main_selected_result"
tmp_res_path = "/home/dreams/Users/yunhengwang/vln/cache/tmp"
w_o_mae_mac_path = "/home/dreams/Users/yunhengwang/vln/cache/wo_mae_mac"
only_mac_path = "/home/dreams/Users/yunhengwang/vln/cache/only_mac"

all_res_name = [name for name in os.listdir(all_res_path) if os.path.isdir(os.path.join(all_res_path, name))]
main_res_name = [name for name in os.listdir(main_res_path) if os.path.isdir(os.path.join(main_res_path, name))]
w_o_mae_mac_name = [name for name in os.listdir(w_o_mae_mac_path) if os.path.isdir(os.path.join(w_o_mae_mac_path, name))]
only_mac_name = [name for name in os.listdir(only_mac_path) if os.path.isdir(os.path.join(only_mac_path, name))]


# 循环 65 次
num = 0 
while(True):
    if num ==65:
        break

    selected_name = random.choice(all_res_name)

    if selected_name in main_res_name or selected_name in w_o_mae_mac_name or selected_name in only_mac_name:
        continue

    src = os.path.join(all_res_path, selected_name)
    dst_tmp = os.path.join(tmp_res_path, selected_name)

    # 复制到 tmp
    shutil.copytree(src, dst_tmp)


    # 更新 main_res_name，避免重复
    main_res_name.append(selected_name)

    num += 1


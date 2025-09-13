from PIL import Image
import os


def save_imagine(nav_step, trajctory_id, all_frames, part_frames, save_path):

    os.makedirs(save_path + "imagine", exist_ok=True)
    os.makedirs(save_path + "imagine/" + "all", exist_ok=True)
    os.makedirs(save_path + "imagine/" + "mllm", exist_ok=True)

    for idx, img_array in enumerate(all_frames):
        img = Image.fromarray(img_array.astype('uint8'))  # 转为PIL格式
        img.save(os.path.join(save_path + "imagine/" + "all/", f"{nav_step}_{trajctory_id}_{idx}.png"))

    for idx, img_array in enumerate(part_frames):
        img = Image.fromarray(img_array.astype('uint8'))  # 转为PIL格式
        img.save(os.path.join(save_path + "imagine/" + "mllm/", f"{nav_step}_{trajctory_id}_{idx}.png"))


def content_to_txt(content, save_path):
    escaped_content = content.encode('unicode_escape').decode()
    with open(save_path + "thinking.txt", "a", encoding="utf-8") as f:
        f.write(escaped_content.strip() + "\n\n")
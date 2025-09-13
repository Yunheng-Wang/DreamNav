import numpy as np

def compute_passable_proportions(semantic_img: np.ndarray, semantic_map: dict):
    # 设定可通行类别
    # passable_labels = {'floor', 'stairs'}
    passable_labels = {'floor', 'floor map'}

    # 获取所有可通行的类别ID
    passable_ids = {cat_id for cat_id, name in semantic_map.items() if name in passable_labels}

    # 图像切分
    h, w = semantic_img.shape
    left = semantic_img[:, :w // 2]
    right = semantic_img[:, w // 2:]

    # 构造掩码
    full_mask = np.isin(semantic_img, list(passable_ids))
    left_mask = np.isin(left, list(passable_ids))
    right_mask = np.isin(right, list(passable_ids))

    # 比例计算
    total_ratio = np.sum(full_mask) / semantic_img.size
    left_ratio = np.sum(left_mask) / left.size
    right_ratio = np.sum(right_mask) / right.size

    return total_ratio, left_ratio, right_ratio


import numpy as np

def compute_passable_proportions_mask(accessible_area_mask: np.ndarray):

    mask = accessible_area_mask


    H, W = mask.shape

    # 全图比例
    total_ratio = mask.sum() / mask.size if mask.size > 0 else np.nan

    mid = W // 2

    left  = mask[:, :mid]
    right = mask[:, mid:]


    left_ratio  = left.sum()  / left.size  if left.size  > 0 else np.nan
    right_ratio = right.sum() / right.size if right.size > 0 else np.nan

    return float(total_ratio), float(left_ratio), float(right_ratio)



def video_extraction(video, num):
    if num == 12:
        return [video[1], video[5], video[9], video[13], video[17], video[21], video[25], video[29], video[33], video[37], video[41], video[45]] # 23 step
    if num == 9:
        return [video[1], video[5], video[9], video[13], video[17], video[21], video[25], video[29], video[33]] # 17 step
    if num == 6:
        return [video[1], video[5], video[9], video[13], video[17], video[21]] # 11 step
    if num == 3:
        return [video[1], video[3], video[5], video[7], video[9]] # 5 step 
    

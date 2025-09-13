import numpy as np
import base64
import cv2

def transform_base64(image_np: np.ndarray) -> str:
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]
    _, buffer = cv2.imencode('.png', image_np[:, :, ::-1])  # 用PNG代替JPG
    return base64.b64encode(buffer).decode('utf-8')

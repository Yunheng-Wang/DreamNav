import argparse
from FastSAM.fastsam import FastSAM, FastSAMPrompt 
import ast
import torch
from PIL import Image
from FastSAM.utils.fastsamtools import convert_box_xywh_to_xyxy
import numpy as np


def segmentation_mask(rgb,save_path):
    model = FastSAM("./FastSAM/weights/FastSAM-x.pt")
    input = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
    everything_results = model(
        input,
        device="cuda",
        retina_masks=True,
        imgsz=640,
        conf=0.3,
        iou=0.6   
        )

    prompt_process = FastSAMPrompt(input, everything_results, device="cuda")


    ann = prompt_process.text_prompt(text="walkable indoor area: floor, hallway, corridor, aisle, ramp, stairs (not walls, ceiling, furniture, beds, sofas, tables, cabinets, windows)")
    
    prompt_process.plot(
        annotations=ann,
        output_path= save_path + "egocentric_semantic.png",
        bboxes = None,
        points = None,
        point_label = None,
        withContours=False,
        better_quality=False,
    )
    
    return ann[0]


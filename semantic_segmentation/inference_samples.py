from glob import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from nicr_mt_scene_analysis.data import move_batch_to_device
from nicr_mt_scene_analysis.data import mt_collate

from .emsaformer.args import ArgParserEMSAFormer
from .emsaformer.data import get_datahelper
from .emsaformer.model import EMSAFormer
from .emsaformer.preprocessing import get_preprocessor
from .emsaformer.visualization import visualize_predictions
from .emsaformer.weights import load_weights
import sys







def _get_args():
    parser = ArgParserEMSAFormer()

    # add additional arguments
    group = parser.add_argument_group('Inference')
    group.add_argument(    # useful for appm context module
        '--inference-input-height',
        type=int,
        default=480,
        dest='validation_input_height',    # used in test phase
        help="Network input height for predicting on inference data."
    )
    group.add_argument(    # useful for appm context module
        '--inference-input-width',
        type=int,
        default=640,
        dest='validation_input_width',    # used in test phase
        help="Network input width for predicting on inference data."
    )
    group.add_argument(
        '--depth-max',
        type=float,
        default=None,
        help="Additional max depth values. Values above are set to zero as "
             "they are most likely not valid. Note, this clipping is applied "
             "before scaling the depth values."
    )
    group.add_argument(
        '--depth-scale',
        type=float,
        default=1.0,
        help="Additional depth scaling factor to apply."
    )

    default_samples_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'samples'
    )
    group.add_argument(
        '--samples-path',
        type=str,
        default=default_samples_dir,
        help="Directory containing the samples."
    )
    group.add_argument(
        '--output-path',
        type=str,
        default=None,
        help="Directory to save the results."
    )
    group.add_argument(
        '--show-results',
        action='store_true',
        default=False,
        help="Show results in a window."
    )

    return parser.parse_args()


def _load_img(fp):
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def semantic_infer(img_rgb: np.ndarray, img_depth: np.ndarray, identifier: str = "habitat_frame"):
    # 写死参数：相当于你在命令行输入以下内容
    sys.argv = [
        'inference_samples.py',
        '--dataset', 'scannet',
        '--tasks', 'semantic', 
        '--raw-depth',
        '--depth-max', '8.0',
        '--depth-scale', '1000.0',
        '--weights-filepath', './semantic_segmentation/trained_models/scannet/scannet_swin_multi_t_v2_128_segformermlp_decoder.pth',
    ]
    args = _get_args()
    device = torch.device(args.device)

    data = get_datahelper(args)
    dataset_config = data.dataset_config
    model = EMSAFormer(args, dataset_config=dataset_config)
    checkpoint = torch.load(args.weights_filepath,map_location=torch.device('cpu'))
    load_weights(args, model, checkpoint['state_dict'], verbose=True)
    torch.set_grad_enabled(False) 
    model.eval()
    model.to(device)

    # build preprocessor
    preprocessor = get_preprocessor(
        args,
        dataset=data.datasets_valid[0],
        phase='test',
        multiscale_downscales=None 
    )

    # 深度图裁剪 + 缩放
    if args.depth_max is not None:
        img_depth[img_depth > args.depth_max] = 0
    img_depth *= args.depth_scale

    # preprocess sample
    sample = preprocessor({
        'rgb': img_rgb,
        'depth': img_depth,
        'identifier': identifier
    })

    # add batch axis as there is no dataloader
    batch = mt_collate([sample])
    batch = move_batch_to_device(batch, device=device)

    # apply model
    predictions = model(batch, do_postprocessing=True)

    # visualize predictions
    preds_viz = visualize_predictions(
        predictions=predictions,
        batch=batch,
        dataset_config=dataset_config
    )
    """
    # 正确读取模型输出的语义分割 label map
    semantic_map = np.array(preds_viz['semantic_segmentation_idx'][0])
    walkable_class_ids = [1, 19]
    occupancy_map = np.ones_like(semantic_map, dtype=np.uint8)
    occupancy_map[np.isin(semantic_map, walkable_class_ids)] = 0
    """
    semantic = preds_viz['semantic_segmentation_idx'][0]  # 已是 PIL 图像

    semantic.save(f"{identifier}_semantic_map.png")

    sll = data.dataset_config.semantic_label_list
    class_names = list(sll.class_names)
    semantic_category_map = {i: name for i, name in enumerate(class_names)}
    return np.array(semantic), semantic_category_map
    

    
    



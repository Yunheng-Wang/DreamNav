<div align="center">

# DreamNav

### A Trajectory-Based Imaginative Framework for Zero-Shot Vision-and-Language Navigation

**ICRA 2026**

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2509.11197)
[![Project Page](https://img.shields.io/badge/Project-Page-2b657a.svg)](https://yunheng-wang.github.io/dreamnav-public.github.io/)
[![Conference](https://img.shields.io/badge/ICRA-2026-f06f52.svg)](#citation)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab.svg)](https://www.python.org/)

Yunheng Wang<sup>1,*</sup>, Yuetong Fang<sup>1,*</sup>, Taowen Wang<sup>1,*</sup>,
Yixiao Feng<sup>1,*</sup>, Yawen Tan<sup>2</sup>, Shuning Zhang<sup>1</sup>,
Peiran Liu<sup>1</sup>, Yiding Ji<sup>1</sup>, Renjing Xu<sup>1,✉</sup>

<sup>1</sup> The Hong Kong University of Science and Technology (Guangzhou)  
<sup>2</sup> Zhejiang Normal University  
<sup>*</sup> Equal contribution

</div>

---

## Overview

DreamNav is a zero-shot Vision-and-Language Navigation (VLN) framework that performs long-horizon navigation from low-cost **egocentric RGB-D observations**. Instead of selecting isolated point-level waypoints from panoramic views, DreamNav predicts complete candidate trajectories, imagines their prospective outcomes, and executes the trajectory that best matches the language instruction.

The framework contains four coupled modules:

1. **EgoView Corrector** — aligns the limited egocentric view with the instruction-relevant direction.
2. **Trajectory Predictor** — generates multiple collision-aware candidate trajectories.
3. **Imagination Predictor** — rolls each trajectory forward and summarizes its long-horizon outcome.
4. **Navigation Manager** — compares candidates, selects the best trajectory, executes it, and monitors progress.

<p align="center">
  <img src="https://raw.githubusercontent.com/Yunheng-Wang/dreamnav-public.github.io/main/assets/images/method-main.webp" width="100%" alt="DreamNav framework overview">
</p>

## Highlights

- **Egocentric-only perception:** does not require panoramic observations or repeated full-view rotations.
- **Trajectory-level decision making:** represents actions as executable trajectories rather than isolated waypoints.
- **Active imagination:** predicts future visual observations before committing to an action.
- **Closed-loop correction:** handles both initial viewpoint errors and post-action orientation drift.
- **Simulation and real-world evaluation:** supports R2R-CE evaluation in Habitat-Sim and deployment on an RGB-D mobile robot.

## Results

### R2R-CE Val-Unseen

| Observation | TL ↓ | NE ↓ | OSR ↑ | SR ↑ | SPL ↑ |
|:--|--:|--:|--:|--:|--:|
| Egocentric RGB-D | **8.20** | **7.06** | **40.95** | **32.79** | **28.95** |

DreamNav achieves the strongest SR and SPL among the compared zero-shot egocentric methods while using no panoramic observations.

### Real-World Navigation

DreamNav succeeds in **12 of 20 trials** across four unseen indoor environments: office, corridor, classroom, and auditorium.

<p align="center">
  <img src="https://raw.githubusercontent.com/Yunheng-Wang/dreamnav-public.github.io/main/assets/images/real-world.webp" width="100%" alt="DreamNav real-world navigation results">
</p>

## Repository Structure

```text
DreamNav/
├── dreamnav_r2r.py              # Main R2R-CE simulation entry point
├── dreamnav._deploy.py          # Real-robot deployment entry point
├── eval.py                      # SR / OSR / NE / TL / SPL evaluation
├── config.yaml                  # Sensor, dataset, model, and hyperparameter config
├── foundation_model/            # LLM / MLLM / video-model API wrappers and prompts
├── sim/                         # Habitat-Sim initialization, state, and action execution
├── trajectory/                  # NavDP trajectory generation and filtering
├── anticipate/                  # Stable Virtual Camera imagination module
├── FastSAM/                     # Walkable-area segmentation module
├── ultralytics/                 # Vendored vision model implementation
├── util/                        # Loading, visualization, logging, and saving utilities
├── deploy/                      # RGB-D acquisition scripts for the physical robot
└── docs/                        # Legacy project-page assets
```

## Requirements

The main pipeline requires:

- Linux
- Python 3.10 or newer
- A CUDA-capable NVIDIA GPU
- A PyTorch build compatible with the installed CUDA version
- Habitat-Sim with RGB-D rendering support
- Matterport3D scenes and R2R-CE task annotations
- API access for the configured language and vision-language models

The imagination and trajectory modules are GPU intensive. A high-memory GPU is recommended, especially when increasing `num_trajectory`, `imagine_resolution`, or `imagine_step`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Yunheng-Wang/DreamNav.git
cd DreamNav
```

### 2. Create a Python environment

```bash
conda create -n dreamnav python=3.10 -y
conda activate dreamnav
```

### 3. Install PyTorch

Install the PyTorch version matching your CUDA driver. For example:

```bash
pip install torch torchvision
```

For CUDA-specific commands, follow the official PyTorch installation selector.

### 4. Install Habitat-Sim

Install a Habitat-Sim build compatible with your system and Matterport3D assets. A typical Conda installation is:

```bash
conda install habitat-sim -c conda-forge -c aihabitat
```

If you need a specific CUDA-enabled build, follow the Habitat-Sim installation documentation instead of relying on the generic command above.

### 5. Install DreamNav dependencies

```bash
pip install -r trajectory/requirements.txt
pip install openai pyyaml scipy numpy-quaternion safetensors huggingface-hub
pip install -e anticipate/stable_virtual_camera --no-deps
```

Then install any missing Stable Virtual Camera dependencies required by your environment:

```bash
pip install roma viser tyro fire ninja einops colorama splines kornia open-clip-torch
```

> The vendored trajectory and imagination components specify different optional Gradio versions. Gradio is not needed for the headless DreamNav navigation pipeline. If the dependency resolver reports a Gradio conflict, keep the version required by your local visualization workflow and install the imagination package with `--no-deps` as shown above.

## Data Preparation

DreamNav expects the R2R-CE episode annotations and Matterport3D scene assets under `data/`.

```text
data/
├── task/
│   └── R2R-CE_val_unseen.json
└── scene_datasets/
    └── mp3d/
        └── <scene_id>/
            ├── <scene_id>.glb
            └── <scene_id>.navmesh
```

The default locations are configured in `config.yaml`:

```yaml
task:
  path: "./data/task/R2R-CE_val_unseen.json"

scene: "./data/scene_datasets/"
```

Matterport3D data is distributed under its own license and is not included in this repository. Obtain access from the official Matterport3D project and prepare the Habitat-compatible scene files before running DreamNav.

## Model Weights

The following weights are required but are not committed to the repository:

| Module | Expected location | Notes |
|:--|:--|:--|
| FastSAM-X | `FastSAM/weights/FastSAM-x.pt` | Used for walkable-area segmentation and EgoView correction. |
| NavDP | `trajectory/checkpoints/navdp-weights.ckpt` | Used by the Trajectory Predictor. Override with `DREAMNAV_NAVDP_CHECKPOINT`. |
| Stable Virtual Camera | Hugging Face cache | Downloaded automatically from `stabilityai/stable-virtual-camera` on first use. |

Create the local weight directories if necessary:

```bash
mkdir -p FastSAM/weights trajectory/checkpoints
```

To use a NavDP checkpoint stored elsewhere:

```bash
export DREAMNAV_NAVDP_CHECKPOINT=/absolute/path/to/navdp-weights.ckpt
```

## Foundation-Model Credentials

DreamNav uses OpenAI-compatible API clients for instruction decomposition, viewpoint reasoning, trajectory selection, and imagined-outcome narration.

Set credentials through environment variables:

```bash
export DEEPBRICKS_API_KEY="your_deepbricks_api_key"
export DASHSCOPE_API_KEY="your_dashscope_api_key"
```

The default model configuration is:

```yaml
foundation_model:
  llm_model_type: "gpt-4o-2024-08-06"
  mllm_model_type: "gpt-4o-2024-08-06"
  video_model_type: "qwen-vl-max-latest"
```

Do not commit API keys to `config.yaml`, shell scripts, notebooks, or Git history.

## Configuration

Edit `config.yaml` before running an experiment.

### Sensor configuration

The default simulator setup uses one egocentric RGB camera and one aligned depth camera:

```yaml
sensor_rgb:
  resolution: [480, 640]
  position: [0.0, 1.25, 0.0]
  hfov: 69.0

sensor_depth:
  resolution: [480, 640]
  position: [0.0, 1.25, 0.0]
  hfov: 69.0
```

### Core hyperparameters

```yaml
hyper_parameter:
  num_trajectory: 4
  imagine_resolution: 320
  imagine_step: 9
```

| Parameter | Description |
|:--|:--|
| `num_trajectory` | Number of candidate trajectories generated at each navigation step. |
| `imagine_resolution` | Spatial resolution used by the imagination model. |
| `imagine_step` | Number of future steps imagined for each candidate trajectory. |
| `gpu_device_id` | Habitat-Sim rendering GPU in `config_env`. |

Increasing trajectory count or imagination length improves candidate coverage but increases model latency, API cost, GPU memory usage, and saved artifacts.

## Running R2R-CE Simulation

After preparing data, weights, credentials, and `config.yaml`, run:

```bash
python dreamnav_r2r.py
```

The script iterates through episodes in the configured task JSON. Episodes with an existing `trajectory.json` are skipped, which allows interrupted runs to resume from the existing `cache/` directory.

For each episode, DreamNav performs:

1. Habitat-Sim environment initialization.
2. Initial egocentric viewpoint correction.
3. Natural-language instruction decomposition.
4. Candidate trajectory generation with NavDP.
5. Candidate filtering and future-view imagination.
6. Long-horizon outcome narration and trajectory selection.
7. Closed-loop trajectory execution and progress monitoring.
8. Trajectory and reasoning artifact serialization.

## Output Structure

Each episode is saved under:

```text
cache/<scene_id>_<episode_id>/
├── trajectory.json                 # Executed agent positions
├── thinking.txt                    # Foundation-model reasoning trace
├── current_rgb.png                 # Current egocentric RGB observation
├── current_depth.png               # Current depth observation
├── egocentric_semantic.png         # Walkable-area segmentation
├── selected_egocentric_trajectory* # Selected trajectory visualization/video
└── ...                             # Candidate and imagination artifacts
```

The exact set of files depends on the enabled visualization and saving options.

## Evaluation

Evaluate all episode folders containing `trajectory.json`:

```bash
python eval.py --result-path cache/
```

To use a non-default configuration:

```bash
python eval.py --result-path cache/ --config path/to/config.yaml
```

The evaluator reports:

- **SR — Success Rate:** final position is within 3 m geodesic distance of the goal.
- **OSR — Oracle Success Rate:** any visited position is within 3 m geodesic distance of the goal.
- **NE — Navigation Error:** final geodesic distance to the goal.
- **TL — Trajectory Length:** total executed path length.
- **SPL — Success weighted by Path Length:** success normalized by path efficiency.

Detailed per-episode metrics and a summary are written to `eval_results.json` in the result directory.

## Real-Robot Deployment

The deployment path is implemented in `dreamnav._deploy.py` and `deploy/get_rgb_depth.sh`. The current implementation assumes:

- A mobile robot reachable over SSH.
- ROS Melodic on the robot.
- RGB and depth image-saver ROS services.
- Robot-side scripts for in-place rotation and trajectory execution.
- RGB depth values saved in millimeters and converted to meters on the compute server.

Before deployment, update the robot-specific settings near the main entry point of `dreamnav._deploy.py`:

```python
ssh_host = "<robot-ip>"
ssh_user = "<robot-user>"
rotate_in_place_path = "/path/on/robot/rotate_in_place.py"
run_trajectory_path = "/path/on/robot/run_trajectory.py"
angle_speed = 0.5236
distance_speed = 0.2
```

Also update the local deployment cache path and the script path used by `get_obervations`. Verify SSH key authentication and ROS services before starting a navigation trial.

Run the deployment script only after reviewing all robot parameters:

```bash
python dreamnav._deploy.py
```

> Real-robot execution can move physical hardware. Test observation capture, emergency stopping, rotation, and short trajectories independently before running the complete policy.

## Troubleshooting

### `FastSAM-x.pt` is missing

Place the FastSAM-X checkpoint at:

```text
FastSAM/weights/FastSAM-x.pt
```

### NavDP checkpoint cannot be found

Either place the checkpoint at the default path:

```text
trajectory/checkpoints/navdp-weights.ckpt
```

or set:

```bash
export DREAMNAV_NAVDP_CHECKPOINT=/absolute/path/to/navdp-weights.ckpt
```

### Stable Virtual Camera downloads fail

The imagination module downloads its checkpoint from Hugging Face on first use. Confirm network access and that the Hugging Face cache directory is writable.

### Habitat-Sim cannot load a scene

Check that `config.yaml` points to the parent scene directory and that the episode `scene_id` resolves to a valid `.glb` file. Also ensure the corresponding navigation mesh exists.

### CUDA out of memory

Reduce one or more of:

- `num_trajectory`
- `imagine_resolution`
- `imagine_step`

You can also run trajectory generation and imagination on separate GPUs by adapting the hard-coded device assignments in the corresponding modules.

### API authentication fails

Confirm that `DEEPBRICKS_API_KEY` and `DASHSCOPE_API_KEY` are exported in the shell that launches DreamNav.

## Citation

If DreamNav is useful in your research, please cite:

```bibtex
@inproceedings{wang2026dreamnav,
  title     = {DreamNav: A Trajectory-Based Imaginative Framework
               for Zero-Shot Vision-and-Language Navigation},
  author    = {Wang, Yunheng and Fang, Yuetong and Wang, Taowen and
               Feng, Yixiao and Tan, Yawen and Zhang, Shuning and
               Liu, Peiran and Ji, Yiding and Xu, Renjing},
  booktitle = {2026 IEEE International Conference on Robotics and
               Automation (ICRA)},
  year      = {2026}
}
```

## Acknowledgements

DreamNav builds on several open-source projects and research resources, including Habitat-Sim, Matterport3D, R2R-CE, NavDP, FastSAM, Stable Virtual Camera, Depth Anything, PyTorch, and the OpenAI-compatible API ecosystem. We thank their authors for making their work available to the community.

## License

This repository does not currently include a license file. Please contact the authors before redistributing or using the code in commercial products. Third-party components and datasets remain subject to their original licenses and terms of use.


import habitat_sim
import numpy as np


def config_simulator(cfg, scene):

    sim_cfg = habitat_sim.SimulatorConfiguration()

    sim_cfg.allow_sliding = cfg["config_env"]["allow_sliding"]
    sim_cfg.create_renderer = cfg["config_env"]["create_renderer"]
    sim_cfg.enable_gfx_replay_save = cfg["config_env"]["enable_gfx_replay_save"]
    sim_cfg.enable_physics = cfg["config_env"]["enable_physics"]
    sim_cfg.force_separate_semantic_scene_graph = cfg["config_env"]["force_separate_semantic_scene_graph"]
    sim_cfg.frustum_culling = cfg["config_env"]["frustum_culling"]
    sim_cfg.gpu_device_id = cfg["config_env"]["gpu_device_id"]
    sim_cfg.requires_textures = cfg["config_env"]["requires_textures"]
    sim_cfg.scene_id = cfg["scene"] + scene

    return sim_cfg


def main_config_agent_and_sensor(cfg):

    # 创建 agent 配置
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = cfg["config_agent"]["height"]
    agent_cfg.radius = cfg["config_agent"]["radius"]
    agent_cfg.body_type = cfg["config_agent"]["body_type"]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=0)
        ),
    }

    # 创建 RGB 传感器
    rgb_sensor_spec = habitat_sim.SensorSpec()
    rgb_sensor_spec.uuid = cfg["sensor_rgb"]["uuid"]
    rgb_sensor_spec.sensor_type = getattr(habitat_sim.SensorType, cfg["sensor_rgb"]["sensor_type"])
    rgb_sensor_spec.sensor_subtype = getattr(habitat_sim.SensorSubType, cfg["sensor_rgb"]["sensor_subtype"])
    rgb_sensor_spec.position = cfg["sensor_rgb"]["position"]
    rgb_sensor_spec.resolution = cfg["sensor_rgb"]["resolution"]
    rgb_sensor_spec.orientation = cfg["sensor_rgb"]["orientation"]
    rgb_sensor_spec.hfov = cfg["sensor_rgb"]["hfov"]

    # 创建 Depth 传感器
    depth_sensor_spec = habitat_sim.SensorSpec()
    depth_sensor_spec.uuid = cfg["sensor_depth"]["uuid"]
    depth_sensor_spec.sensor_type = getattr(habitat_sim.SensorType, cfg["sensor_depth"]["sensor_type"])
    depth_sensor_spec.sensor_subtype = getattr(habitat_sim.SensorSubType, cfg["sensor_depth"]["sensor_subtype"])
    depth_sensor_spec.position = cfg["sensor_depth"]["position"]
    depth_sensor_spec.resolution = cfg["sensor_depth"]["resolution"]
    depth_sensor_spec.orientation = cfg["sensor_depth"]["orientation"]
    depth_sensor_spec.hfov = cfg["sensor_depth"]["hfov"]



    # 绑定两个传感器
    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]

    return agent_cfg

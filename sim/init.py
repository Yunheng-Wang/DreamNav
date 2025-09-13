from . import config
import habitat_sim


# configuration environment (only one agent)
def environment(cfg, scene, agent_id, init_pos, init_rot):
    # configuration simulator & agent
    sim_cfg = config.config_simulator(cfg, scene)
    main_agent_cfg = config.main_config_agent_and_sensor(cfg)

    # initialize habitat simulator
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [main_agent_cfg]))
    agent = sim.initialize_agent(agent_id)

    # setting agent's initial state
    initial_state = habitat_sim.AgentState()
    initial_state.position = init_pos
    initial_state.rotation = init_rot
    agent.set_state(initial_state)

    return sim, agent, main_agent_cfg
    




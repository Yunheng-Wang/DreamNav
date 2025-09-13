

def cur_pos_rot(simulator):
    state = simulator.get_agent(0).get_state()
    pos = list(state.position)
    rot = [state.rotation.x, state.rotation.y, state.rotation.z, state.rotation.w]
    return pos, rot



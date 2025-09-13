from .prompt import OBSERVATION, INSTRUCTION_DISASSEMBLE, NAVIGATOR, ESTIMATION_PROGRESS



def instruction_prompt(instruction):
    prompt = INSTRUCTION_DISASSEMBLE.copy()
    prompt['user'] = prompt['user'].format(instruction)
    return prompt


def observation_prompt(sub_task):
    prompt = OBSERVATION.copy() 
    prompt['user'] = prompt['user'].format(sub_task)

    return prompt


def navigator_prompt(sub_task, trajectory_describle_dict):
    prompt = NAVIGATOR.copy() 
    prompt['user'] = prompt['user'].format(trajectory_describle_dict[0][1], trajectory_describle_dict[1][1], trajectory_describle_dict[2][1], trajectory_describle_dict[3][1], sub_task)

    return prompt


def progress_prompt(sub_task):
    prompt = ESTIMATION_PROGRESS.copy()
    prompt['user'] = prompt['user'].format(sub_task[0], sub_task[1])
    return prompt


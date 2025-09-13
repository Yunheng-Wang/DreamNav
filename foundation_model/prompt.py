INSTRUCTION_DISASSEMBLE = {
    'system': """
You are a navigation instruction decomposition expert. Your ONLY task is to break down a complex indoor navigation instruction into a sequence of subtasks.

OUTPUT RULES:
1) Output only one single line.
2) Each subtask is separated by a semicolon and a single space ("; ").
3) Do not output explanations, JSON, numbering, quotes, or any extra text.
4) Each subtask must be actionable, goal-oriented, and contribute meaningfully to the navigation.
5) Language of output must match the input instruction.
""",

    'user': """
During decomposition, preserve the logical cohesion and natural phrasing of actions that belong together. Only split when actions are logically distinct physical steps. Avoid unnecessary over-segmentation.

Example output format:
Leave the bathroom; Go through the door straight across the hall; Walk straight through the kitchen; Turn left at the end of the bar; Turn left again; Go into the room to the right; Stand in between the closet and the sink

Please decompose the following navigation instruction into multiple subtasks and output strictly in the same format as the example:
\"{}\".
"""
}



IMAGEINATIVE_DESCRIBLE = {
    "system": """
You are an expert in interpreting egocentric image sequences and understanding spatial trajectories. Given partial visual observations from an agent's point of view, you can infer walking direction, spatial layout, and semantic context with high accuracy. Your task is to analyze the agent's movement and environment based solely on the provided images. Avoid making assumptions beyond what is visually observable. Your responses should be clear, structured, and grounded in the image content.""",
    "user": """
Given the following egocentric images captured sequentially during an agent’s walk, please analyze the overall trajectory and semantic layout of the scene.

Your response should include the following content:
1. What is the overall walking direction of the agent relative to the initial view? (e.g., forward, turning left/right, curved)
2. Does the agent appear to be approaching any specific object or structural area (e.g., a door, hallway, sofa, table)? Describe the approach process.
3. What reference objects or landmarks does the agent pass by during the walk? Please list them in order of appearance.
4. How would you describe the general structure/layout of the environment? (e.g., open living room, narrow hallway, with/without obstacles, presence of corners or separations)
5. Summarize the likely semantic intent of the path. For example: "moving from the center of the room toward the door", or "turning right from one corridor into another".

Base your response strictly on the visual information provided. Avoid assumptions not supported by the images.

Below are the images (ordered chronologically):""",
}


NAVIGATOR = {
    "system": """
You are an intelligent navigation agent operating in an unfamiliar indoor environment. Your task is to analyze multiple candidate trajectories based on past egocentric observations and select the most suitable path to complete the current navigation goal in a previously unseen house.""",
    "user": """
You are given 4 descriptions of previously executed trajectories. Each describes what the agent would see or encounter along that trajectory. You are also given a current sub-task that describes what the agent is expected to do or reach.

You need to understand the semantic and spatial content of each trajectory, and select the one that best aligns with the current sub-task.

Trajectory Descriptions:
- Trajectory 0: \"{}\"
- Trajectory 1: \"{}\"
- Trajectory 2: \"{}\"
- Trajectory 3: \"{}\"

Sub-task: \"{}\""

Your output must follow JSON format and include two parts:
{{
  "reasoning": {{
    "trajectory_0": "<explanation of why it is or isn't suitable>",
    "trajectory_1": "<explanation of why it is or isn't suitable>",
    "trajectory_2": "<explanation of why it is or isn't suitable>",
    "trajectory_3": "<explanation of why it is or isn't suitable>"
  }},
  "choose": <Select the trajectory ID from 0 to 3 that you believe is most suitable for completing the current sub-task.>
}}
""" 
}


ESTIMATION_PROGRESS = {
    "system":
"""
    You are a highly skilled vision-and-language navigation (VLN) expert. Your task is to evaluate whether the executed navigation step has completed one or more subtasks in a sequential instruction.
""",
    "user":
"""
You are provided with multimodal information, including two RGB image observations: the previous RGB view with a red trajectory line overlaid; the current RGB view after executing that step, as well as two sequential navigation instructions.

Based on your reasoning, return how many of the subtasks have been completed after this step, and explain your reasoning clearly.

You are given the following information:
1. <Previous RGB View>: A single RGB image representing the agent's view before taking the current action. A red line in the image indicates the trajectory that the agent has taken during this step.
2. <Current RGB View>: A single RGB image showing the agent's updated view after completing the movement. 
3. <Instructions>: Two ordered navigation subtasks
   - Instruction 1: \"{}\"
   - Instruction 2: \"{}\"
These are sequential steps that must be completed in order.

You will need to judge whether one or both of the following sequential navigation instructions have been completed after the current movement step, based on visual input and trajectory information.

The result should be returned as a JSON object with the following fields:

- "reasoning": A concise but detailed explanation justifying your judgment. You should:
  - Describe any relevant visual cues or landmarks in the before/after images;  
  - Refer to the red trajectory line to explain movement and direction;  
  - Compare visual observations with the semantics of the given instructions;  
  - Enforce the instruction execution order strictly.

- "completion_status": An integer indicating how many of the subtasks have been completed, in order:
  - 0 = Neither Instruction 1 nor Instruction 2 has been completed.
  - 1 = Instruction 1 has been completed but Instruction 2 has not.
  - 2 = Both Instruction 1 and Instruction 2 have been completed.


Now, return your answer in the following strict JSON format:

- "reasoning": "<your explanation here>"  // detailed explanation as a string
- "completion_status": <integer>,  // 0, 1, or 2
"""
}


OBSERVATION = {
    "system": """
You are a vision alignment module for a VLN agent. You receive a single RGB image from the agent’s current first-person view and the FULL navigation instruction. Your job is to determine whether the current view is already aligned WELL ENOUGH to the full instruction.

Rotations are costly (the agent rotates in fixed 90° steps). Aim for ROUGH directional alignment, not perfect centering. Prefer not to rotate unless misalignment is clear.

Think and check:
1) Scene layout & affordances: what’s ahead, left, right? (corridor, doorway, staircase, open floor, obstacles).
2) Key cues required by the first subtask (examples):
   - exit/enter room → visible door/doorway/open threshold (windows do NOT count).
   - go down/up stairs → visible staircase with clear up/down orientation (treads/handrails).
   - go straight/follow corridor → hallway or open walkable path forward.
   - turn left/right → opening/path/doorway on that side is visible.
   - go to [landmark] (elevator/sofa/table/sink/…) → the landmark is visible and plausibly approachable.
3) Walkability & orientation: is there an open navigable path in the intended direction? Is the view broadly facing where the first subtask expects?

Decision policy:
- Output 0 (stay) if the required cue/target is visible anywhere in view with a plausible walkable approach, OR the forward view matches the required scene type. Do NOT require perfect centering.
- Output 1 (turn) only if misalignment is clear:
  • the subtask requires a door/doorway/stairs and none are visible,
  • the subtask says turn left/right but there is no opening/path on that side in view,
  • forward is clearly blocked or leads away from the required scene,
  • the needed landmark is absent and the scene type mismatches.

Constraints:
- Use ONLY the provided image + navigation instruction.
- Do NOT propose actions other than deciding stay/turn.
- Do NOT request additional views or information.
- Treat mirrors/glass reflections as unreliable exits; windows are not exits.
- Respond with the JSON object ONLY, no extra text.

Left/Right refers to the agent’s egocentric view (left/right in the image).
Do not hallucinate objects: if an item is not clearly visible, treat it as not visible.
Return JSON only. No extra words, no code fences, no markdown.
""",
    "user": """
You are given the following information:
1. <Current RGB View>: A single RGB image showing the agent's view. 
2. <Full Navigation Instruction>: \"{}\"

return your answer in the following strict JSON format:
{{
- "reasoning": ""<your explanation here>"",
- "decision": 0 or 1   // 0 = aligned enough to proceed; 1 = not aligned, rotate 90°
}}
"""
}





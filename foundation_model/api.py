from openai import OpenAI
from .encode import transform_base64



def reasoning_mllm(cfg, image_list, prompt):
    if cfg["foundation_model"]["mllm_model_type"] == "gpt-4o-2024-08-06":
        MLLM = OpenAI(api_key=cfg["foundation_model"]["deepbricks_api_key"], base_url="https://api.deepbricks.ai/v1/")
        content = [{"type": "text", "text": prompt["user"]}]
        for index, image in enumerate(image_list):
            base64_image = transform_base64(image)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": content}
        ]
        # 用于存储流式的所有字符
        full_output = ""

        completion = MLLM.chat.completions.create(
            model=cfg["foundation_model"]["mllm_model_type"],
            messages=messages,
            temperature=0.5,
            stream=True
        )
        for chunk in completion:
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_output += chunk.choices[0].delta.content
        
        return full_output


def reasoning_llm(cfg, prompt):
        
    LLM = OpenAI(api_key=cfg["foundation_model"]["deepbricks_api_key"], base_url="https://api.deepbricks.ai/v1/")

    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]}
    ]
    
    # 用于存储流式的所有字符
    full_output = ""

    response = LLM.chat.completions.create(
        model = cfg["foundation_model"]["llm_model_type"],
        messages = messages,
        temperature = 0.5,
        stream=True
    )
    for chunk in response:
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_output += chunk.choices[0].delta.content

    return full_output
        
        

def reasoning_video(cfg, image_list, prompt):
    if cfg["foundation_model"]["video_model_type"] == "qwen-vl-max-latest":
        client = OpenAI(api_key=cfg["foundation_model"]["qwen_api_key"], base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        video_base64_list = [f"data:image/jpeg;base64,{transform_base64(image)}" for image in image_list]
        content = [
        {
            "type": "video",
            "video": video_base64_list
        },
        {
            "type": "text",
            "text": prompt["user"]
        }
        ]
        messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": prompt["system"]}]
        },
        {
            "role": "user",
            "content": content
        }
        ]

        response = client.chat.completions.create(
            model="qwen-vl-max-latest",
            messages=messages,
            temperature=0.5,
            stream=False
        )
    return response.choices[0].message.content

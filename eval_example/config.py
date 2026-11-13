llm_config=dict(
    base_url="http://localhost:18008/v1",#"https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="YOUR_API_KEY",
    model="Qwen3-0.6B-finetune", #"deepseek-r1-0528"
    temperature=0.0,
)

judge_llm_config=dict(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="YOUR_API_KEY",
    model="deepseek-v3",
    temperature=0.0,
)


llm_config_list=[
    dict(
        base_url="http://localhost:18008/v1",#"https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="YOUR_API_KEY",
        model="Qwen3-0.6B-finetune", #"deepseek-r1-0528"
        temperature=0.1,
    ),
    dict(
        base_url="http://localhost:19008/v1",#"https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="YOUR_API_KEY",
        model="Qwen3-0.6B", #"deepseek-r1-0528"
        temperature=0.1,
    ),
    dict(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="YOUR_API_KEY",
        model="qwen3-4b",
        temperature=0.1,
    ),
    dict(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="YOUR_API_KEY",
        model="qwen3-8b",
        temperature=0.1,
    ),
    dict(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="YOUR_API_KEY",
        model="qwen-plus",
        temperature=0.1,
    ),
]


import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import FlexibleDataLoader, parse_opanai_response2messages

data_loader = FlexibleDataLoader(
    dataset_name="general_qa",
    dataset_path="/workspace/llm/dataset/self_build/origin_data/202509/gesture.jsonl"
)
data = data_loader.load_data()
assert len(data) > 0
print(data[0])


from client.llm_client import OpenAIClient

llm = OpenAIClient(
    model="qwen-plus",
    api_key="YOUR_API_KEY",
    api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

indexs_assistant = [i for i, msg in enumerate(data[0].messages) if msg["role"] == "assistant"]
messages = data[0].messages[:indexs_assistant[-1]]
print(f"messages: {messages}")
tools = data[0].tools
# print(f"tools: {tools}")
import json
# 修复tool_calls中的arguments参数格式，将字典转换为JSON字符串


print(f"messages: {messages}")

response = llm.client.chat.completions.create(
    model=llm.model,
    messages=messages,
    tools=tools,
)


print(f"response: {response.model_dump()}")

print(f"response: {parse_opanai_response2messages(response)}")   

response = llm.async_client.chat.completions.create(
    model=llm.model,
    messages=messages,
    tools=tools,
)

print(f"response: {response.model_dump()}")

print(f"response: {parse_opanai_response2messages(response)}")   
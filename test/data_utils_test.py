
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import load_and_convert
import json

if __name__ == "__main__":
    test_cases = [
        # 含工具调用的ShareGPT格式
        {
            "system": "你可以调用天气工具获取实时天气",
            "tools": [
                {
                    "name": "get_weather",
                    "description": "获取指定城市的天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "城市名称"}
                        },
                        "required": ["city"]
                    }
                }
            ],
            "conversations": [
                {"from": "human", "value": "上海今天的天气怎么样？"},
                {"from": "function_call", "value": "1111"},
                {"from": "observation" '{"temperature": 28, "condition": "晴天"}', "name": "get_weather"},
                {"from": "gpt", "value": "上海今天28度，晴天"}
            ]
        },
        # 普通ShareGPT格式（无工具调用）
        {
            "conversations": [
                {"from": "human", "value": "你好"},
                {"from": "gpt", "value": "你好！有什么可以帮你？"}
            ]
        }
    ]
    print("=== 工具调用格式转换测试 ===")
    # 创建临时测试文件
    test_file = "test_data.json"
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)
    
    # 测试load_and_convert函数

    converted_datasets = load_and_convert(test_file)
    print(converted_datasets)
    if os.path.exists(test_file):
        os.remove(test_file)
        
# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入要测试的函数
from response_messages import del_last_assistant, response_messages

def test_del_last_assistant():
    """
    测试del_last_assistant函数的功能
    """
    print("开始测试del_last_assistant函数...")
    
    # 测试用例1: 空消息列表
    test1 = []
    result1 = del_last_assistant(test1)
    print(f"测试用例1 - 空消息列表:")
    print(f"输入: {test1}")
    print(f"输出: {result1}")
    print(f"结果: {'通过' if result1 == [] else '失败'}\n")
    
    # 测试用例2: 没有assistant消息
    test2 = [
        {"role": "user", "content": "你好"},
        {"role": "system", "content": "你是一个AI助手"}
    ]
    result2 = del_last_assistant(test2)
    print(f"测试用例2 - 没有assistant消息:")
    print(f"输入: {test2}")
    print(f"输出: {result2}")
    print(f"结果: {'通过' if result2 == test2 else '失败'}\n")
    
    # 测试用例3: 最后一条消息是assistant
    test3 = [
        {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！我是AI助手。"}
    ]
    expected3 = [
        {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": "你好"}
    ]
    result3 = del_last_assistant(test3)
    print(f"测试用例3 - 最后一条消息是assistant:")
    print(f"输入: {test3}")
    print(f"期望输出: {expected3}")
    print(f"实际输出: {result3}")
    print(f"结果: {'通过' if result3 == expected3 else '失败'}\n")
    
    # 测试用例4: 有多个assistant消息，删除最后一个
    test4 = [
        {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": "问题1"},
        {"role": "assistant", "content": "回答1"},
        {"role": "user", "content": "问题2"},
        {"role": "assistant", "content": "回答2"}
    ]
    expected4 = [
        {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": "问题1"},
        {"role": "assistant", "content": "回答1"},
        {"role": "user", "content": "问题2"}
    ]
    result4 = del_last_assistant(test4)
    print(f"测试用例4 - 有多个assistant消息，删除最后一个:")
    print(f"输入: {test4}")
    print(f"期望输出: {expected4}")
    print(f"实际输出: {result4}")
    print(f"结果: {'通过' if result4 == expected4 else '失败'}\n")
    
    # 测试用例5: 包含tools的情况
    test5 = {
        "messages": [
            {"role": "system", "content": "你是一个AI助手"},
            {"role": "user", "content": "计算1+1"},
            {"role": "assistant", "content": "让我为您计算"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"}
                        }
                    }
                }
            }
        ]
    }
    print(f"测试用例5 - 包含tools的情况:")
    print(f"输入messages: {test5['messages']}")
    print(f"输入tools: {test5['tools']}")
    result5 = del_last_assistant(test5['messages'])
    print(f"处理后messages: {result5}")
    print(f"结果: {'通过' if len(result5) == 2 and result5[-1]['role'] == 'user' else '失败'}\n")

def test_response_messages_function():
    """
    测试response_messages函数（使用模拟响应）
    """
    print("\n开始测试response_messages函数（使用模拟响应）...")
    
    # 准备测试数据
    test_dataset = [
        {
            "messages": [
                {"role": "system", "content": "你是一个AI助手"},
                {"role": "user", "content": "测试问题"},
                {"role": "assistant", "content": "这是要被删除的助手回复"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "param": {"type": "string"}
                            }
                        }
                    }
                }
            ]
        }
    ]
    
    # 调用函数（使用模拟响应）
    responses = response_messages(test_dataset, mock_response=True)
    
    # 检查结果
    print("测试结果:")
    print(f"响应数量: {len(responses)}")
    if responses:
        response = responses[0]
        print(f"处理后的messages数量: {len(response['processed_messages'])}")
        print(f"最后一条消息的role: {response['processed_messages'][-1]['role'] if response['processed_messages'] else '无'}")
        print(f"API响应包含assistant: {'assistant' in str(response['api_response']['choices'][0]['message']['role'])}")
        
        # 验证最后一个assistant消息是否被删除
        original_has_assistant = any(msg['role'] == 'assistant' for msg in test_dataset[0]['messages'])
        processed_has_assistant = any(msg['role'] == 'assistant' for msg in response['processed_messages'])
        
        print(f"\n验证:")
        print(f"原始数据包含assistant消息: {original_has_assistant}")
        print(f"处理后数据是否移除了最后一个assistant: {'通过' if (original_has_assistant and not processed_has_assistant) or (not original_has_assistant and not processed_has_assistant) else '失败'}")

if __name__ == "__main__":
    print("===== 测试response_messages.py功能 =====")
    
    # 测试del_last_assistant函数
    test_del_last_assistant()
    
    # 测试response_messages函数
    test_response_messages_function()
    
    print("\n===== 测试完成 =====")
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import json
import random
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import openai
from tqdm import tqdm
from loguru import logger
try:
    from .llm_client import OpenAIClient
except ImportError:
    print("llm_client模块未找到，将使用openai库直接调用")

# 尝试导入异步版本的openai
try:
    import aiohttp
except ImportError:
    logger.warning("aiohttp未安装，异步功能可能无法正常使用")

# logger.add("logs/response_messages.log", rotation="100 MB", encoding="utf-8")

def del_last_assistant(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    删除messages列表中最后一个role为assistant的消息
    
    Args:
        messages: 消息列表
    
    Returns:
        处理后的消息列表
    """
    if not messages:
        return messages
    if messages[-1].get('role') == 'assistant':
        return messages[:-1]
    else:
        raise ValueError("最后一条消息必须是assistant角色")
    

def mock_openai_response(messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """
    生成模拟的OpenAI API响应
    
    Args:
        messages: 消息列表
        **kwargs: 其他参数
    
    Returns:
        模拟的响应字典
    """
    # 获取最后一条用户消息的内容
    last_user_message = None
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            last_user_message = msg.get('content', '')
            break
    
    # 生成模拟响应内容
    mock_contents = [
        "这是一个模拟的响应。",
        "我是一个AI助手，可以帮助您解答问题。",
        "感谢您的提问，让我为您提供一些信息。",
        "根据您的问题，我认为...",
        f"我已收到您的消息：{last_user_message[:50]}..." if last_user_message else "收到您的请求。"
    ]
    
    return {
                    "role": "assistant",
                    "content": random.choice(mock_contents)
                }
    # {
    #     "id": f"chatcmpl-{random.randint(1000, 9999)}",
    #     "object": "chat.completion",
    #     "created": int(time.time()),
    #     "model": kwargs.get("model", "gpt-3.5-turbo"),
    #     "choices": [
    #         {
    #             "index": 0,
    #             "message": {
    #                 "role": "assistant",
    #                 "content": random.choice(mock_contents)
    #             },
    #             "finish_reason": "stop"
    #         }
    #     ],
    #     "usage": {
    #         "prompt_tokens": random.randint(50, 200),
    #         "completion_tokens": random.randint(20, 100),
    #         "total_tokens": random.randint(70, 300)
    #     }
    # }

def rename_tool_response(messages: List[Dict[str, Any]],origin_role: str = 'tool_response',target_role: str = 'tool') -> List[Dict[str, Any]]:
    """
    将messages列表中的tool_response角色重命名为assistant
    
    Args:
        messages: 消息列表
    
    Returns:
        处理后的消息列表
    """
    out = []
    for i in messages:
        if i.get('role') == origin_role:
            i.update({'role': target_role})
        out.append(i)
    return out

def response_messages(
    dataset: List[Dict[str, Any]],
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    base_url: Optional[str] = "https://api.openai.com/v1",
    temperature: float = 0.05,
    mock_response: bool = False
) -> List[Dict[str, Any]]:
    """
    处理数据集并调用OpenAI API获取响应
    
    Args:
        dataset: 数据集列表，每个元素包含messages和tools
        model: 使用的模型名称
        api_key: OpenAI API密钥
        base_url: OpenAI API基础URL
        mock_response: 是否使用模拟响应
    
    Returns:
        包含原始数据和响应的列表
    """
    responses = []
    
    # 处理每条数据
    for idx, item in enumerate(tqdm(dataset, desc="处理数据")):
        # 获取原始messages和tools
        original_messages = item.get('messages', [])
        tools = item.get('tools', [])
        # print(f"original_messages: {original_messages}")
        # 处理messages，删除最后一个assistant消息
        if len(original_messages) > 0 and original_messages[-1].get('role') == 'assistant':
            processed_messages = del_last_assistant(original_messages)

            processed_messages = rename_tool_response(processed_messages)
        else:
            logger.warning(f"第{idx}条数据长度为{len(original_messages)},最后一条不是assistant消息，跳过")
            continue
        # print(f"processed_messages: {processed_messages}")
        
        # 准备API调用参数
        api_params = {
            "model": model,
            "messages": processed_messages,
            "temperature": temperature,
        }
        
        # 如果有tools，添加到API参数中
        if tools:
            api_params["tools"] = tools
        try:
            # 获取响应
            start_time = time.time()
            if 'OpenAIClient' in globals():
                client = OpenAIClient(api_key=api_key, base_url=base_url)
                
                response = client.chat_completion(**api_params)
            else:
                # 回退到直接使用openai库
                client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
                response = client.chat.completions.create(**api_params)
            # 将OpenAI响应对象转换为字典
        except Exception as e:
            logger.error(f"第{idx}条数据调用OpenAI API失败: {e}\nbase_url: {base_url} \n api_params: {api_params}")
            raise
        end_time = time.time()
        use_time = round(end_time - start_time, 4)
        response_message = response.choices[0].message.model_dump()
        response_message_simple = {
            "role": response_message.get('role', 'assistant'),
            "content": response_message.get('content', ''),
            "tool_calls": response_message.get('tool_calls', None),
        }
        
        usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        try:
            use_now = response.usage.model_dump()
            usage.update(dict(filter(lambda item:item[0] in usage,use_now.items())))
        except:
            pass
        # 构建响应项
        response_item = {
            "input_messages": processed_messages,
            "tools": tools,
            "ref_message": original_messages[-1],
            "res_message": response_message_simple,
            "use_time": use_time,
        }
        response_item.update(usage)
        
        responses.append(response_item)
    
    return responses


async def response_messages_async(
    dataset: List[Dict[str, Any]],
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    base_url: Optional[str] = "https://api.openai.com/v1",
    temperature: float = 0.05,
    max_concurrency: int = 10,
    mock_response: bool = False
) -> List[Dict[str, Any]]:
    """
    异步处理数据集并调用OpenAI API获取响应
    
    Args:
        dataset: 数据集列表，每个元素包含messages和tools
        model: 使用的模型名称
        api_key: OpenAI API密钥
        base_url: OpenAI API基础URL
        temperature: 采样温度
        max_concurrency: 最大并发数
        mock_response: 是否使用模拟响应
    
    Returns:
        包含原始数据和响应的列表
    """
    responses = []
    
    # 创建一个信号量来控制并发数
    semaphore = asyncio.Semaphore(max_concurrency)
    
    # 异步处理单个数据项的函数
    async def process_item(idx, item):
        async with semaphore:
            try:
                # 获取原始messages和tools
                original_messages = item.get('messages', [])
                tools = item.get('tools', [])
                
                # 处理messages，删除最后一个assistant消息
                if len(original_messages) > 0 and original_messages[-1].get('role') == 'assistant':
                    processed_messages = del_last_assistant(original_messages)
                    processed_messages = rename_tool_response(processed_messages)
                else:
                    logger.warning(f"第{idx}条数据长度为{len(original_messages)},最后一条不是assistant消息，跳过")
                    return None
                
                # 准备API调用参数
                api_params = {
                    "model": model,
                    "messages": processed_messages,
                    "temperature": temperature,
                }
                
                # 如果有tools，添加到API参数中
                if tools:
                    api_params["tools"] = tools
                
                start_time = time.time()
                
                # 使用线程池执行同步的API调用
                loop = asyncio.get_event_loop()
                try:
                    if 'OpenAIClient' in globals():
                        client = OpenAIClient(api_key=api_key, base_url=base_url)
                        response = await loop.run_in_executor(
                            None,
                            lambda: client.chat_completion(**api_params)
                        )
                    else:
                        # 回退到直接使用openai库
                        client = openai.OpenAI(
                            api_key=api_key,
                            base_url=base_url
                        )
                        response = await loop.run_in_executor(
                            None,
                            lambda: client.chat.completions.create(**api_params)
                        )
                except Exception as e:
                    logger.error(f"第{idx}条数据调用OpenAI API失败: {e}\nbase_url: {base_url} \n api_params: {api_params}")
                    raise
                
                end_time = time.time()
                use_time = round(end_time - start_time, 4)
                
                response_message = response.choices[0].message.model_dump()
                response_message_simple = {
                    "role": response_message.get('role', 'assistant'),
                    "content": response_message.get('content', ''),
                    "tool_calls": response_message.get('tool_calls', None),
                }
                
                usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
                try:
                    use_now = response.usage.model_dump()
                    usage.update(dict(filter(lambda item:item[0] in usage, use_now.items())))
                except:
                    pass
                
                # 构建响应项
                response_item = {
                    "input_messages": processed_messages,
                    "tools": tools,
                    "ref_message": original_messages[-1],
                    "res_message": response_message_simple,
                    "use_time": use_time,
                    "index": idx,  # 保留索引以便后续排序
                }
                response_item.update(usage)
                
                return response_item
            except Exception as e:
                logger.error(f"处理第{idx}条数据时出错: {e}")
                raise
    
    # 创建所有任务
    tasks = [process_item(idx, item) for idx, item in enumerate(dataset)]
    
    # 使用tqdm显示进度
    logger.info(f"开始异步处理{len(tasks)}条数据，最大并发数: {max_concurrency}")
    
    # 收集结果
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理结果
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"第{idx}条数据处理失败: {result}")
        elif result is not None:
            responses.append(result)
    
    # 按原始顺序排序
    responses.sort(key=lambda x: x.get('index', 0))
    
    logger.info(f"异步处理完成，成功处理{len(responses)}条数据")
    
    return responses


def response_messages_concurrent(
    dataset: List[Dict[str, Any]],
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    base_url: Optional[str] = "https://api.openai.com/v1",
    temperature: float = 0.05,
    max_workers: int = 10,
    mock_response: bool = False
) -> List[Dict[str, Any]]:
    """
    使用线程池并发处理数据集并调用OpenAI API获取响应
    
    Args:
        dataset: 数据集列表，每个元素包含messages和tools
        model: 使用的模型名称
        api_key: OpenAI API密钥
        base_url: OpenAI API基础URL
        temperature: 采样温度
        max_workers: 线程池中的最大工作线程数
        mock_response: 是否使用模拟响应
    
    Returns:
        包含原始数据和响应的列表
    """
    responses = []
    
    # 处理单个数据项的函数
    def process_item(idx, item):
        try:
            # 获取原始messages和tools
            original_messages = item.get('messages', [])
            tools = item.get('tools', [])
            
            # 处理messages，删除最后一个assistant消息
            if len(original_messages) > 0 and original_messages[-1].get('role') == 'assistant':
                processed_messages = del_last_assistant(original_messages)
                processed_messages = rename_tool_response(processed_messages)
            else:
                logger.warning(f"第{idx}条数据长度为{len(original_messages)},最后一条不是assistant消息，跳过")
                return None
            
            # 准备API调用参数
            api_params = {
                "model": model,
                "messages": processed_messages,
                "temperature": temperature,
            }
            
            # 如果有tools，添加到API参数中
            if tools:
                api_params["tools"] = tools
            
            start_time = time.time()
            
            try:
                if 'OpenAIClient' in globals():
                    client = OpenAIClient(api_key=api_key, base_url=base_url)
                    response = client.chat_completion(**api_params)
                else:
                    # 回退到直接使用openai库
                    client = openai.OpenAI(
                        api_key=api_key,
                        base_url=base_url
                    )
                    response = client.chat.completions.create(**api_params)
            except Exception as e:
                logger.error(f"第{idx}条数据调用OpenAI API失败: {e}\nbase_url: {base_url} \n api_params: {api_params}")
                raise
            
            end_time = time.time()
            use_time = round(end_time - start_time, 4)
            
            response_message = response.choices[0].message.model_dump()
            response_message_simple = {
                "role": response_message.get('role', 'assistant'),
                "content": response_message.get('content', ''),
                "tool_calls": response_message.get('tool_calls', None),
            }
            
            usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
            try:
                use_now = response.usage.model_dump()
                usage.update(dict(filter(lambda item:item[0] in usage, use_now.items())))
            except:
                pass
            
            # 构建响应项
            response_item = {
                "input_messages": processed_messages,
                "tools": tools,
                "ref_message": original_messages[-1],
                "res_message": response_message_simple,
                "use_time": use_time,
                "index": idx,  # 保留索引以便后续排序
            }
            response_item.update(usage)
            
            return response_item
        except Exception as e:
            logger.error(f"处理第{idx}条数据时出错: {e}")
            raise
    
    logger.info(f"开始并发处理{len(dataset)}条数据，最大线程数: {max_workers}")
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {executor.submit(process_item, idx, item): idx for idx, item in enumerate(dataset)}
        
        # 使用tqdm显示进度
        with tqdm(total=len(dataset)) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result is not None:
                        responses.append(result)
                except Exception as e:
                    logger.error(f"第{idx}条数据处理失败: {e}")
                finally:
                    pbar.update(1)
    
    # 按原始顺序排序
    responses.sort(key=lambda x: x.get('index', 0))
    
    logger.info(f"并发处理完成，成功处理{len(responses)}条数据")
    
    return responses


def save_responses(responses: List[Dict[str, Any]], output_file: str) -> None:
    """
    保存响应结果到文件
    
    Args:
        responses: 响应列表
        output_file: 输出文件路径
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # 保存到JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)
        
        print(f"响应结果已成功保存到: {output_file}")
    except Exception as e:
        print(f"保存响应结果失败: {e}")




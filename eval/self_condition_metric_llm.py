"""自我认知一致性指标评估模块"""

import pandas as pd
import json
import asyncio
import logging
from typing import List, Dict, Any
from loguru import logger

from .llm_client import OpenAIClient
# from config import judge_llm_config
from .prompt import build_evaluation_messages

# 配置日志


logger.add("./logs/self_condition_metric_llm.log", rotation="10 MB", encoding="utf-8")
def self_condition_metric_llm(responses: List[Dict[str, Any]],judge_llm_config, model_name: str = "", 
                             dataset_name: str = "", subset_name: str = "", 
                             channel: str = "chat", max_concurrency: int = 5) -> pd.DataFrame:
    """
    使用LLM模型对响应进行自我认知一致性评分，支持并发请求
    评分维度：
    - SelfCondition: 模型输出和参考输出对自我认知的一致性（0分或1分）
    
    Args:
        responses: 响应列表，每个元素包含res_message和ref_message
        judge_llm_config: 评估LLM配置base_url,api_key,model,temperature...
        model_name: 模型名称
        dataset_name: 数据集名称
        subset_name: 子集名称
        channel: 通道名称
        max_concurrency: 最大并发数
        
    Returns:
        pandas DataFrame: 包含评分结果的DataFrame
    """
    # 初始化OpenAI客户端
    client = OpenAIClient(
        base_url=judge_llm_config["base_url"],
        api_key=judge_llm_config["api_key"],
        model=judge_llm_config["model"]
    )
    
    async def process_responses():
        """异步处理所有响应"""
        all_results = []
        total_responses = len(responses)
        
        # 使用信号量控制并发
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def bounded_process(idx, response_dict):
            async with semaphore:
                return await _process_single_response(
                    response_dict, idx, total_responses, model_name, 
                    dataset_name, subset_name, channel, client
                )
        
        # 创建所有任务
        tasks = [bounded_process(idx, response_dict) 
                for idx, response_dict in enumerate(responses)]
        
        # 并发执行所有任务
        results = await asyncio.gather(*tasks)
        
        # 合并所有结果
        for result in results:
            all_results.extend(result)
        
        return all_results
    
    # 运行异步处理
    try:
        # 检查当前事件循环是否正在运行
        try:
            loop = asyncio.get_running_loop()
            # 如果事件循环正在运行，使用create_task
            all_results = loop.run_until_complete(process_responses())
        except RuntimeError:
            # 如果没有正在运行的事件循环，使用new_event_loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            all_results = loop.run_until_complete(process_responses())
            loop.close()
    except Exception as e:
        logger.error(f"Error in async processing: {str(e)}")
        all_results = []
    
    # 创建DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        # 调整列顺序
        df = df[['Model', 'Dataset', 'Metric', 'Subset', 'Sample/Num', 'Score', 'Channel']]
        return df
    else:
        # 如果没有结果，返回空的DataFrame
        columns = ['Model', 'Dataset', 'Metric', 'Subset', 'Sample/Num', 'Score', 'Channel']
        return pd.DataFrame(columns=columns)


async def _process_single_response(response_dict: Dict[str, Any], idx: int, total: int, 
                                  model_name: str, dataset_name: str, subset_name: str, 
                                  channel: str, client: OpenAIClient) -> List[Dict[str, Any]]:
    """
    异步处理单个响应，获取自我认知一致性评分
    
    Args:
        response_dict: 包含res_message和ref_message的字典
        idx: 当前响应的索引
        total: 总响应数
        model_name: 模型名称
        dataset_name: 数据集名称
        subset_name: 子集名称
        channel: 通道名称
        client: OpenAI客户端实例
        
    Returns:
        List[Dict[str, Any]]: 包含评分结果的列表
    """
    results = []
    
    try:
        # 提取参考消息和模型消息
        ref_message = response_dict.get('ref_message', '')
        res_message = response_dict.get('res_message', '')
        
        # 构建评估消息
        messages = build_evaluation_messages(ref_message, res_message, metric_type="self_condition")
        
        # 调用LLM进行评估
        response = await client.achat_completion(
            messages=messages,
            temperature=0.0,  # 确定性输出
            max_tokens=100
        )
        
        # 解析评估结果
        # print(response)
        # 将ChatCompletion对象转换为字典
        response_dict = response.choices[0].message.model_dump()
        evaluation_content = response_dict.get('content', '')
        
        # 尝试解析JSON
        try:
            eval_result = json.loads(evaluation_content)
            
            # 获取SelfCondition评分
            self_condition_score = eval_result.get('score', 0)
            
            # 添加评分结果
            results.append({
                'Model': model_name,
                'Dataset': dataset_name,
                'Metric': 'SelfCondition',
                'Subset': subset_name,
                'Sample/Num': idx + 1,
                'Score': float(self_condition_score),  # 转换为float类型
                'Channel': channel
            })
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response for sample {idx + 1}: {evaluation_content}")
            # 解析失败时，添加默认评分0
            results.append({
                'Model': model_name,
                'Dataset': dataset_name,
                'Metric': 'SelfCondition',
                'Subset': subset_name,
                'Sample/Num': idx + 1,
                'Score': 0.0,
                'Channel': channel
            })
    
    except Exception as e:
        logger.error(f"Error processing sample {idx + 1}/{total}: {str(e)}")
        # 发生异常时，添加默认评分0
        results.append({
            'Model': model_name,
            'Dataset': dataset_name,
            'Metric': 'SelfCondition',
            'Subset': subset_name,
            'Sample/Num': idx + 1,
            'Score': 0.0,
            'Channel': channel
        })
    
    # 记录进度
    if (idx + 1) % 10 == 0 or (idx + 1) == total:
        logger.info(f"Processed {idx + 1}/{total} samples")
    
    return results
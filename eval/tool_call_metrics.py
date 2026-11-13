

import pandas as pd
import json
import uuid
import asyncio
import logging
from typing import List, Dict, Any
from loguru import logger

# 导入LLM相关模块和配置
from .llm_client import OpenAIClient
# from config import judge_llm_config
from .prompt import build_evaluation_messages
from .utils import contant2toolcall

# 配置日志


logger.add("./logs/tool_call_metrics.log", rotation="10 MB", encoding="utf-8")


def tool_call_metrics_parse(responses,judge_llm_config={},model_name="", dataset_name="", subset_name="", channel="chat"):
    """
    从响应字典中提取工具调用指标。
    评分维度：
    - ToolCallUseAccuracy: 是否调用工具判断的准确性（0分或1分）
    - ToolFormatAccuracy: 调用工具格式准确性（0分或1分）
    - ToolCategoryNameAccuracy: 工具调用类别和名称准确性（0分或1分）
    - ToolParamValueAccuracy: 工具调用参数值准确性（0分或1分）
    
    Args:
        responses: 响应列表，每个元素包含res_message和ref_message
        model_name: 模型名称
        dataset_name: 数据集名称
        subset_name: 子集名称
        channel: 通道名称
        
    Returns:
        pandas DataFrame: 包含评分结果的DataFrame
    """
    
    
    # 准备结果数据
    results = []
    
    # 遍历每个响应
    for idx, response_dict in enumerate(responses):
        res_message = contant2toolcall(response_dict.get('res_message', {}))
        ref_message = contant2toolcall(response_dict.get('ref_message', {}))


        # 获取参考消息和回复消息中的工具调用
        ref_tool_calls = ref_message.get('tool_calls', [])
        res_tool_calls = res_message.get('tool_calls', [])
        
        # 初始化分数
        metrics = {
            'Model': model_name,
            'Dataset': dataset_name,
            'Subset': subset_name,
            'Sample/Num': f"{idx+1}/{len(responses)}",
            'Channel': channel
        }
        # print(f"response_dict: {response_dict}")
        # print(f"ref_tool_calls: {ref_tool_calls}")
        # print(f"res_tool_calls: {res_tool_calls}")
        # 情况1：参考消息中没有工具调用
        if not ref_tool_calls or len(ref_tool_calls) == 0:
            # 如果回复中调用了工具，则调用判断准确性为0
            if res_tool_calls and len(res_tool_calls) > 0:
                metrics['Metric'] = 'ToolCallUseAccuracy'
                metrics['Score'] = 0
                results.append(metrics.copy())
            else:
                metrics['Metric'] = 'ToolCallUseAccuracy'
                metrics['Score'] = 1
                results.append(metrics.copy())
            # 其他指标为None
            for metric_name in ['ToolFormatAccuracy', 'ToolCategoryNameAccuracy', 'ToolParamValueAccuracy']:
                metrics_copy = metrics.copy()
                metrics_copy['Metric'] = metric_name
                metrics_copy['Score'] = None
                results.append(metrics_copy)
        
        # 情况2：参考消息中有工具调用，但回复中没有
        elif not res_tool_calls or len(res_tool_calls) == 0:
            metrics['Metric'] = 'ToolCallUseAccuracy'
            metrics['Score'] = 0
            results.append(metrics.copy())
            # 其他指标为None
            for metric_name in ['ToolFormatAccuracy', 'ToolCategoryNameAccuracy', 'ToolParamValueAccuracy']:
                metrics_copy = metrics.copy()
                metrics_copy['Metric'] = metric_name
                metrics_copy['Score'] = None
                results.append(metrics_copy)
        
        # 情况3：双方都有工具调用，进行详细评分
        else:
            # 1. ToolCallUseAccuracy：是否调用工具判断的准确性
            metrics['Metric'] = 'ToolCallUseAccuracy'
            metrics['Score'] = 1
            results.append(metrics.copy())
            
            # 2. ToolFormatAccuracy：调用工具格式准确性
            # 检查是否包含必要的字段：id, type, function
            format_accurate = 1
            # 首先检查所有回复工具调用的基本格式
            for tool_call in res_tool_calls:
                if not all(key in tool_call for key in ['id', 'type', 'function']):
                    format_accurate = 0
                    break
                # 检查function中是否包含name和arguments
                if not all(key in tool_call['function'] for key in ['name', 'arguments']):
                    format_accurate = 0
                    break
            
            # 然后检查参数数量是否匹配
            # 为每个回复的工具调用找到对应的参考工具调用并比较参数键
            if format_accurate == 1 and len(ref_tool_calls) == len(res_tool_calls):
                # 创建参考工具调用名称到参数键的映射
                ref_name_to_args = {}
                for ref_call in ref_tool_calls:
                    ref_func = ref_call.get('function', {})
                    ref_args = json.loads(ref_func.get('arguments', '{}')).keys()
                    ref_name_to_args[ref_func.get('name', '')] = set(ref_args)
                
                # 检查每个回复工具调用的参数键是否匹配对应的参考工具调用
                for res_call in res_tool_calls:
                    res_func = res_call.get('function', {})
                    res_func_name = res_func.get('name', '')
                    if res_func_name not in ref_name_to_args:
                        # format_accurate = 0
                        # break
                        continue # 在参考回复中没有的函数不考虑
                    res_args = json.loads(res_func.get('arguments', '{}')).keys()
                    if set(res_args) != ref_name_to_args[res_func_name]:
                        format_accurate = 0
                        break
            elif len(ref_tool_calls) != len(res_tool_calls):
                format_accurate = 0
            
            metrics['Metric'] = 'ToolFormatAccuracy'
            metrics['Score'] = format_accurate
            results.append(metrics.copy())
            
            # 3. ToolCategoryNameAccuracy：工具调用类别和名称准确性
            name_accurate = 0
            ref_func_names = [ref_tool_call['function'].get('name', '') for ref_tool_call in ref_tool_calls]
            res_func_names = [res_tool_call['function'].get('name', '') for res_tool_call in res_tool_calls]
            if set(ref_func_names) == set(res_func_names):
                name_accurate = 1
            metrics['Metric'] = 'ToolCategoryNameAccuracy'
            metrics['Score'] = name_accurate
            results.append(metrics.copy())
            
            # 4. ToolParamValueAccuracy：工具调用参数值准确性
            param_accurate = 0
            # 创建参考工具调用名称到参数值的映射
            ref_name_to_args = {}
            for ref_call in ref_tool_calls:
                ref_func = ref_call.get('function', {})
                ref_args = json.loads(ref_func.get('arguments', '{}'))
                ref_name_to_args[ref_func.get('name', '')] = ref_args
            
            # 检查所有回复工具调用的参数值是否都匹配对应的参考工具调用
            all_params_match = True
            for res_call in res_tool_calls:
                res_func = res_call.get('function', {})
                res_func_name = res_func.get('name', '')
                if res_func_name not in ref_name_to_args:
                    # all_params_match = False
                    # break 
                    continue # 在参考回复中没有的函数不考虑
                res_args = json.loads(res_func.get('arguments', '{}'))
                if res_args != ref_name_to_args[res_func_name]:
                    all_params_match = False
                    break
            
            if all_params_match:
                param_accurate = 1

            
            metrics['Metric'] = 'ToolParamValueAccuracy'
            metrics['Score'] = param_accurate
            results.append(metrics.copy())
    
    # 创建DataFrame
    if results:
        df = pd.DataFrame(results)
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
    处理单个响应，调用LLM进行评分
    
    Args:
        response_dict: 响应字典
        idx: 索引
        total: 总数
        model_name: 模型名称
        dataset_name: 数据集名称
        subset_name: 子集名称
        channel: 通道名称
        client: OpenAI客户端
        
    Returns:
        list: 评分结果列表
    """
    results = []
    metrics = {
        'Model': model_name,
        'Dataset': dataset_name,
        'Subset': subset_name,
        'Sample/Num': f"{idx+1}/{total}",
        'Channel': channel
    }
    
    try:
        # 获取响应消息和参考消息
        res_message = response_dict.get('res_message', {})
        ref_message = response_dict.get('ref_message', {})
        
        # 构建评估消息
        messages = build_evaluation_messages(ref_message, res_message)
        
        # 调用LLM进行评估
        response = await client.achat_completion(
            messages=messages,
            temperature=0.0,  # 使用确定性输出
            max_tokens=500
        )
        
        # 解析LLM响应
        # 将ChatCompletion对象转换为字典
        response_dict = response.choices[0].message.model_dump()
        llm_content = response_dict.get('content', '').strip()
        if llm_content.startswith('{') and llm_content.endswith('}'):
            try:
                scores = json.loads(llm_content)
                
                # 生成结果
                for metric_name, score in scores.items():
                    metrics_copy = metrics.copy()
                    metrics_copy['Metric'] = metric_name
                    metrics_copy['Score'] = score
                    results.append(metrics_copy)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {llm_content}")
                # 添加默认错误结果
                for metric_name in ['ToolCallUseAccuracy', 'ToolFormatAccuracy', 
                                  'ToolCategoryNameAccuracy', 'ToolParamValueAccuracy']:
                    metrics_copy = metrics.copy()
                    metrics_copy['Metric'] = metric_name
                    metrics_copy['Score'] = None
                    results.append(metrics_copy)
        else:
            logger.error(f"LLM response not in expected JSON format: {llm_content}")
            # 添加默认错误结果
            for metric_name in ['ToolCallUseAccuracy', 'ToolFormatAccuracy', 
                              'ToolCategoryNameAccuracy', 'ToolParamValueAccuracy']:
                metrics_copy = metrics.copy()
                metrics_copy['Metric'] = metric_name
                metrics_copy['Score'] = None
                results.append(metrics_copy)
                
    except Exception as e:
        logger.error(f"Error processing response {idx+1}: {str(e)}")
        # 添加默认错误结果
        for metric_name in ['ToolCallUseAccuracy', 'ToolFormatAccuracy', 
                          'ToolCategoryNameAccuracy', 'ToolParamValueAccuracy']:
            metrics_copy = metrics.copy()
            metrics_copy['Metric'] = metric_name
            metrics_copy['Score'] = None
            results.append(metrics_copy)
    
    return results


def tool_call_metrics_llm(responses: List[Dict[str, Any]],judge_llm_config, model_name: str = "", 
                         dataset_name: str = "", subset_name: str = "", 
                         channel: str = "chat", max_concurrency: int = 5) -> pd.DataFrame:
    """
    使用LLM模型对响应进行工具调用指标评分，支持并发请求
    评分维度：
    - ToolCallUseAccuracy: 是否调用工具判断的准确性（0分或1分）
    - ToolFormatAccuracy: 调用工具格式准确性（0分或1分）
    - ToolCategoryNameAccuracy: 工具调用类别和名称准确性（0分或1分）
    - ToolParamValueAccuracy: 工具调用参数值准确性（0分或1分）
    
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
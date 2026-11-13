# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional
import json  # 需要添加json模块
import pandas as pd
import uuid
import sys
import os
from loguru import logger
import datetime
import random
import uuid
import copy

from dataset_deal import load_dataset

try:
    from response_messages import response_messages, response_messages_async, response_messages_concurrent
    from utils import (toolcall2contant,
                        contant2toolcall,
                        calculate_metric_averages,
                        normalize_column_name,
                        safe_json_dumps,
                        dataset_messages_rebuild)
    from cut_messages import cut_datasets_message
    from tool_call_metrics import tool_call_metrics_parse,tool_call_metrics_llm
    from self_condition_metric_llm import self_condition_metric_llm
    from register_data_custom import register_datasets
    from swanlablog import swanlab_log_df,swanlab_log_radar
except:
    from .response_messages import response_messages, response_messages_async, response_messages_concurrent
    from .utils import (toolcall2contant,
                        contant2toolcall,
                        calculate_metric_averages,
                        calculate_response_averages,
                        generate_radar_chart_data,
                        normalize_column_name,
                        safe_json_dumps,
                        dataset_messages_rebuild)
    from .cut_messages import cut_datasets_message
    from .tool_call_metrics import tool_call_metrics_parse,tool_call_metrics_llm
    from .self_condition_metric_llm import self_condition_metric_llm
    from .register_data_custom import register_datasets
    from .swanlablog import swanlab_log_df,swanlab_log_radar

# 导入必要的异步支持库
try:
    import asyncio
except ImportError:
    logger.warning("asyncio未安装，异步功能可能无法正常使用")



logger.add("./logs/eval.log", rotation="100 MB", encoding="utf-8")

def parse_response_dict_compare(response_dict:Dict[str,Any]):
    """
    从响应字典中提取对比指标。
    """
    res_message = response_dict['res_message']
    ref_message = response_dict['ref_message']
    # print(f"res_message: {res_message}")
    # print(f"ref_message: {ref_message}")
    hypothese = toolcall2contant(res_message).get('content','')
    reference = toolcall2contant(ref_message).get('content','')
    return hypothese,reference

def common_metric_from_responses(responses,model_name="",dataset_name="",subset_name="",channel=""):
    """
    从响应字典中提取通用指标。
    responses = list[response_dict]
    response_dict key: res_message, ref_message, input_messages, tools
    """
    from eval.metrics.common_score import evaluate_texts
    
    hypotheses = []
    references = []
    for response_dict in responses:
        hypothese,reference = parse_response_dict_compare(response_dict)
        hypotheses.append(hypothese)
        references.append(reference)
        # print(f"response_dict: {response_dict}")
    # print(f"hypotheses: {hypotheses}")
    # print(f"references: {references}")
    metrics = evaluate_texts(
        hypotheses=hypotheses,
        references=references,
        model_name=model_name,
        dataset_name=dataset_name, 
        subset_name=subset_name,
        channel=channel,
        per_sample=True,
        # show_table=True
    )
    return metrics


def eval_from_data_config_swanlab(jsonpath, 
                                eval_llm_config, 
                                judge_llm_config={},
                                max_per_eval_num=100,
                                cutmode="assistant", 
                                keepmode="all", 
                                save_path= "./output",
                                use_swanlab=True,       
                                swanlab_tags=[],
                                swanlab_token='YOUR_SWANLAB_TOKEN',
                                swanlab_project='model-eval',
                                swanlab_workspace='demo',
                                swanlab_exp_name=None,
                                swanlab_mode='cloud',
                                swanlab_lark_webhook_url=None,
                                swanlab_lark_secret=None,
                                **kwargs):
    """
    封装eval_from_data_config函数，添加SwanLab日志记录功能
    
    Args:
        jsonpath: 配置文件路径
        eval_llm_config: 评估模型配置
        cutmode: 切割模式（"tool_call" 或 "assistant"）
        keepmode: 保留模式（"first"、"last" 或 "all"）
        max_per_eval_num: 每个评估的最大样本数
        swanlab_token: SwanLab API令牌
        swanlab_project: SwanLab项目名称
        swanlab_workspace: SwanLab工作空间名称
        swanlab_exp_name: SwanLab实验名称，默认使用模型名称
        swanlab_mode: SwanLab模式
        swanlab_lark_webhook_url: 飞书Webhook URL
        swanlab_lark_secret: 飞书Webhook密钥
    
    Returns:
        tuple: (all_metric, all_responses) - 评估结果和响应数据
    """
    # 如果未指定实验名称，使用模型名称
    if swanlab_exp_name is None:
        swanlab_exp_name = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-eval"
    
    # 初始化SwanLab
    if use_swanlab:
        import swanlab
        from swanlab.plugin.notification import LarkCallback

        lark_callback = LarkCallback(
            webhook_url=swanlab_lark_webhook_url, 
            secret=swanlab_lark_secret,
        )
        # try:
        if isinstance(eval_llm_config, list):
            model_names = [model_config.get("model","") for model_config in eval_llm_config]    
            eval_llm_config_push = copy.deepcopy(eval_llm_config)
            [model_config.update({"api_key": ""}) for model_config in eval_llm_config_push]
        else:
            model_names = [eval_llm_config.get("model","")]
            eval_llm_config_push = copy.deepcopy(eval_llm_config)
            eval_llm_config_push.update({"api_key": ""})

        swanlab.init(
            experiment_name=swanlab_exp_name,
            tags=swanlab_tags+model_names,
            project=swanlab_project,
            workspace=swanlab_workspace,
            mode=swanlab_mode,
            token=swanlab_token,
            callbacks=[lark_callback]
        )
        try:
            dataset_dict = json.load(open(jsonpath,'r'))
        except:
            logger.error(f"读取配置文件{jsonpath}失败")
            raise
        # 记录配置信息
        config_to_log = {
            'model': eval_llm_config_push,
            'max_per_eval_num': max_per_eval_num,
            'config_file': dataset_dict,
            'cutmode': cutmode, 
            'keepmode': keepmode, 
            "judge_llm": judge_llm_config.get("model","") if isinstance(judge_llm_config, dict) else "",
        }
        swanlab.config.update(config_to_log)
        logger.info(f"SwanLab初始化成功，实验名称: {swanlab_exp_name}")
        # except Exception as e:
        #     logger.error(f"SwanLab初始化失败: {str(e)}")
        #     swanlab = None

    dataseteval_dict = return_datainput(jsonpath,
                    max_per_eval_num=max_per_eval_num,
                    cutmode=cutmode, 
                    keepmode=keepmode)
    
     
    if isinstance(eval_llm_config, list):
        metric_list = []
        responses_list = []
        for model_config in eval_llm_config:
            logger.info(f'model_config: {model_config}')
            metric, responses = eval_from_data_dict(dataseteval_dict,model_config,judge_llm_config,**kwargs)  
            metric_list.append(metric)
            responses_list.append(responses)
        all_metric = pd.concat(metric_list, ignore_index=True)
        all_responses = pd.concat(responses_list, ignore_index=True)
    else:
        logger.info(f'eval_llm_config: {eval_llm_config}')
        all_metric, all_responses = eval_from_data_dict(dataseteval_dict, eval_llm_config,judge_llm_config,**kwargs)

    id_metric = 'sample_uuid'
    id_response = 'sample_id'
    id_metric = normalize_column_name(id_metric, all_metric.columns)
    id_response = normalize_column_name(id_response, all_responses.columns)
    all_data = pd.merge(all_metric, all_responses, left_on=id_metric, right_on=id_response, how='left')
    if save_path:
        all_metric.to_csv(os.path.join(save_path,f"all_metric_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"), index=False)
        all_responses.to_csv(os.path.join(save_path,f"all_responses_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"), index=False)

    # 计算并记录all_data的统计信息
    stats_info = calculate_all_data_statistics(all_data, all_metric)
    swanlab.log({'data_statistics': swanlab.Text(json.dumps(stats_info, ensure_ascii=False))})
    
    # swanlab.log({'total_metrics': f"total_metrics: {len(all_metric)}"})
    # swanlab.log({'unique_models': f"{len(all_metric['Model'].unique()) if 'Model' in all_metric.columns else 0}"})
    # swanlab.log({'unique_datasets': f"{len(all_metric['Dataset'].unique()) if 'Dataset' in all_metric.columns else 0}"})
    


    # try:
    # 计算平均分
    def swanlab_plot(all_data,main_column_name='Model',sub_column_name='',part="all"):
        # main_column_name = 'Model' # 主列
        # sub_column_name = 'Dataset' # 副列
        # part = "dataset" # swanlab 分组显示
        main_column_name = normalize_column_name(main_column_name, all_metric.columns)
        sub_column_name = normalize_column_name(sub_column_name, all_metric.columns)

        metric_columns = [] #'BLEU-1','ROUGE-1-F','ROUGE-1-R','SelfCondition','ToolCallUseAccuracy', 'ToolFormatAccuracy','ToolCategoryNameAccuracy', 'ToolParamValueAccuracy'
        metric_columns = normalize_column_name(metric_columns, all_metric.columns)

        groupby_columns = [main_column_name,sub_column_name]
        # 数据集指标绘制
        metric_column_name = "Metric"
        score_column_name = "Score"
        metric_keep_names = metric_columns
        averages = calculate_metric_averages(all_metric,group_by=groupby_columns,metric_columns=metric_keep_names,metric_column_name = metric_column_name,score_column_name=score_column_name)
        if save_path:
            averages.to_csv(os.path.join(save_path,f"averages_{main_column_name}_{sub_column_name}_{part}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"), index=False)
        swanlab_log_df(averages,f"table_{part}/metric")

        #运行指标绘制
        numeric_columns = ['use_time','prompt_tokens','completion_tokens','total_tokens']
        numeric_columns = normalize_column_name(numeric_columns, all_responses.columns)
        response_averages = calculate_response_averages(all_data,group_by=groupby_columns,numeric_columns=numeric_columns)
        swanlab_log_df(response_averages,f"table_{part}/spend")
        
        # 使用swanlablog模块中的函数记录雷达图
        radar_data_dict = generate_radar_chart_data(averages, main_column_name, sub_column_name, metric_columns, auto_delete_metrics=True)
        swanlab_log_radar(radar_data_dict,name_title=f"radar_{part}")
    swanlab_plot(all_data)
    swanlab_plot(all_data,main_column_name='Model',sub_column_name='channel',part="channel")
    swanlab_plot(all_data,main_column_name='Model',sub_column_name='Dataset',part="dataset")
    # 从all_metric中为每个模型和数据集抽取样本，并通过sample_id关联all_responses
    try:
        sample_select_characters = ['Model','Dataset']
        # keep_metric_columns = sample_select_characters + [id_metric]
        keep_metric_columns = []
        # keep_response_columns = ['input_messages', 'tools', 'ref_message', 'res_message', 'sample_id']
        keep_response_columns = ['input_messages', 'tools', 'ref_message', 'res_message']

        sample_select_characters = normalize_column_name(sample_select_characters, all_metric.columns)
        keep_response_columns = normalize_column_name(keep_response_columns, all_responses.columns)
        keep_metric_columns = normalize_column_name(keep_metric_columns, all_metric.columns)

        if id_metric in all_metric.columns and id_response in all_responses.columns:
            # 使用pandas操作抽取所需样本并合并数据
            # 1. 按模型和数据集分组，每组取第一个样本
            sampled_metrics = all_metric.groupby(sample_select_characters).first().reset_index()
            
            # 2. 如果指定了keep_metric_columns且非空，则进行列筛选
            if keep_metric_columns is not None and len(keep_metric_columns) > 0:
                # 确保分组列和id_metric列都在筛选列中
                required_columns = sample_select_characters + [id_metric]
                filtered_metric_cols = [col for col in keep_metric_columns if col in sampled_metrics.columns]
                # 添加必需的列
                for col in required_columns:
                    if col not in filtered_metric_cols and col in sampled_metrics.columns:
                        filtered_metric_cols.append(col)
                sampled_metrics = sampled_metrics[filtered_metric_cols]
            
            # 3. 使用id_metric和id_response对齐合并all_responses中的对应行
            merged_data = pd.merge(sampled_metrics, all_responses, left_on=id_metric, right_on=id_response, how='left')
            
            # 4. 如果指定了keep_response_columns且非空，则进行列筛选
            if keep_response_columns is not None and len(keep_response_columns) > 0:
                # 保留必要的列：分组列、指标列和指定的响应列
                metric_cols = [col for col in sampled_metrics.columns if col not in [id_metric, id_response]]
                response_cols = [col for col in keep_response_columns if col in merged_data.columns]
                # 确保id_metric也在最终结果中
                final_columns = sample_select_characters + [id_metric] + response_cols
                # 去重并保持顺序
                seen = set()
                unique_final_columns = []
                for col in final_columns:
                    if col not in seen:
                        unique_final_columns.append(col)
                        seen.add(col)
                merged_data = merged_data[unique_final_columns]
            
            # 5. 处理JSON字段并限制长度
            # json_fields = ['input_messages', 'tools', 'ref_message', 'res_message']
            for field in merged_data.columns:
                merged_data[field] = merged_data[field].apply(safe_json_dumps)
                # 限制长文本字段的长度以避免显示过长
                merged_data[field] = merged_data[field].apply(
                    lambda x: x[:1000] + "..." if isinstance(x, str) and len(x) > 1000 else x
                )
            
            # 如果有样本结果，创建并记录到swanlab
            if not merged_data.empty:
                # 创建样本结果表格（参考182-190行代码）
                sample_table = swanlab.echarts.Table()
                
                # 获取表头和数据（使用pandas的列名和值）
                headers = list(merged_data.columns)
                rows = merged_data.values.tolist()
                
                # 添加数据到表格
                sample_table.add(headers, rows)
                
                # 记录样本结果表格到swanlab
                swanlab.log({"example/test_samples": sample_table})
                logger.info(f"模型和数据集样本结果{len(merged_data)}条表格已成功记录到SwanLab")
            else:
                logger.warning("未找到任何模型和数据集的样本结果")
        else:
            logger.warning("all_metric中缺少必要的列")
    except Exception as e:
        import traceback
        logger.error(f"处理样本结果时出错: {str(e)}")
        logger.error(f"详细错误信息: {traceback.format_exc()}")
    
    # 完成SwanLab实验
    swanlab.finish()
    logger.info("评估结果已成功记录到SwanLab")
    
    return all_metric, all_responses






def metric_add_sampleuuid(current_sample_ids,metric_df):
    metric_df['sample_uuid'] = metric_df['Sample/Num'].apply(lambda x: current_sample_ids[int(str(x).split('/')[0])-1])
    return metric_df

def return_datainput(jsonpath,
                    max_per_eval_num=None,
                    cutmode="assistant", 
                    keepmode="first",
                    skip_register=False):

    with open(jsonpath, 'r', encoding='utf-8') as f:
        dataseteval_dict = json.load(f)
    dataseteval_names = list(dataseteval_dict.keys())
    if not skip_register:
        try:
            register_datasets(dataseteval_dict)
        except Exception as e:
            logger.warning(f"注册数据集失败: {str(e)}")
            pass
    # 为确保ID生成的稳定性，使用基于时间戳和随机数的组合
    for evalname in dataseteval_names:#dataseteval_names[:2]dataseteval_names["bench_self_condition"]
        dataset = load_dataset([evalname])[0]
        dataset = cut_datasets_message(dataset,cutmode=cutmode, keepmode=keepmode)
        dataset = dataset_messages_rebuild(dataset)
        # dataset_input = random.sample(dataset, max_per_eval_num)
        if max_per_eval_num:
            dataset_input = dataset[:int(max_per_eval_num)]
        else:
            dataset_input = dataset
        dataseteval_dict[evalname]["dataset_input"] = dataset_input
    return dataseteval_dict



def eval_from_data_dict(dataseteval_dict, eval_llm_config, judge_llm_config, execution_mode='concurrent', max_workers=10, max_concurrency=10):
    '''
    从数据配置字典中评估模型性能
    
    Args:
        dataseteval_dict: 数据配置字典，键为数据集名称，值为包含'dataset_input'/channel等字典
        eval_llm_config: 评估模型配置
        judge_llm_config: 评判模型配置
        execution_mode: 执行模式，可选值：'sequential'(顺序), 'concurrent'(并发), 'async'(异步)
        max_workers: 并发模式下的最大工作线程数
        max_concurrency: 异步模式下的最大并发数
    Returns:
        tuple: (all_metric, all_responses) - 评估结果和响应数据
    '''
    all_metric = None
    all_responses = pd.DataFrame()
    # 为确保ID生成的稳定性，使用基于时间戳和随机数的组合
    model_name = eval_llm_config["model"]
    api_key = eval_llm_config["api_key"]
    base_url = eval_llm_config["base_url"]
    temperature = eval_llm_config["temperature"]
    
    # 验证执行模式
    valid_modes = ['sequential', 'concurrent', 'async']
    if execution_mode not in valid_modes:
        logger.warning(f"无效的执行模式: {execution_mode}，将使用默认模式 'sequential")
        execution_mode = 'sequential'
    
    logger.info(f"使用执行模式: {execution_mode}")
    
    for evalname, dataset_config in dataseteval_dict.items():#dataseteval_names[:2]dataseteval_names["bench_self_condition"]
        dataset_input = dataset_config["dataset_input"]
        logger.info(f"处理数据集: {evalname}, 数据量: {len(dataset_input)}")
        
        # 根据执行模式选择不同的函数调用OpenAI接口
        if execution_mode == 'sequential':
            # 顺序执行
            responses = response_messages(
                            dataset_input,
                            model=model_name,   
                            api_key=api_key,
                            base_url=base_url,
                            temperature=temperature,
                        )
        elif execution_mode == 'concurrent':
            # 并发执行（线程池）
            responses = response_messages_concurrent(
                            dataset_input,
                            model=model_name,   
                            api_key=api_key,
                            base_url=base_url,
                            temperature=temperature,
                            max_workers=max_workers
                        )
        else:  # execution_mode == 'async'
            # 异步执行
            try:
                responses = asyncio.run(response_messages_async(
                                        dataset_input,
                                        model=model_name,   
                                        api_key=api_key,
                                        base_url=base_url,
                                        temperature=temperature,
                                        max_concurrency=max_concurrency
                                    ))
            except Exception as e:
                logger.error(f"异步执行失败，回退到并发模式: {e}")
                # 回退到并发模式
                responses = response_messages_concurrent(
                                dataset_input,
                                model=model_name,   
                                api_key=api_key,
                                base_url=base_url,
                                temperature=temperature,
                                max_workers=max_workers
                            )
        
        # 为每个响应生成唯一ID并保存到all_responses
        current_sample_ids = []
        for idx, response in enumerate(responses):
            # 生成更稳定且唯一的ID：datasetname_idx_uuid12位
            # 使用更多位的UUID以确保唯一性，同时保持ID的可读性
            sample_id = f"{evalname}_{idx}_{uuid.uuid4().hex[:12]}"
            current_sample_ids.append(sample_id)
            
            # 添加sample_id到response字典
            response['sample_id'] = sample_id
            
            # 添加到all_responses DataFrame
            response_df = pd.DataFrame([response])
            all_responses = pd.concat([all_responses, response_df], ignore_index=True)
        
        # 计算指标并添加sample_id关联
        common_metric = common_metric_from_responses(responses, model_name=model_name, dataset_name=evalname, subset_name="", channel=dataseteval_dict[evalname].get('channel','chat'))
        # 确保sample_id列添加成功
        common_metric = metric_add_sampleuuid(current_sample_ids,common_metric)
        
        if all_metric is None:
            all_metric = common_metric
        else:
            all_metric = pd.concat([all_metric, common_metric])
        
        if dataseteval_dict[evalname].get('channel','') == 'self_condition':
            self_condition_metric = self_condition_metric_llm(responses,judge_llm_config, model_name=model_name, dataset_name=evalname, subset_name="", channel=dataseteval_dict[evalname].get('channel','self_condition'),max_concurrency=max_concurrency)
            self_condition_metric = metric_add_sampleuuid(current_sample_ids,self_condition_metric)
            all_metric = pd.concat([all_metric, self_condition_metric])
        
        if dataseteval_dict[evalname].get('channel','') == 'tool':
            # tool_call_metrics = tool_call_metrics_llm(responses,judge_llm_config, model_name=model_name, dataset_name=evalname, subset_name="", channel=dataseteval_dict[evalname].get('channel','tool'))
            tool_call_metrics = tool_call_metrics_parse(responses, model_name=model_name, dataset_name=evalname, subset_name="", channel=dataseteval_dict[evalname].get('channel','tool'))
            tool_call_metrics = metric_add_sampleuuid(current_sample_ids,tool_call_metrics)
            all_metric = pd.concat([all_metric, tool_call_metrics])
    
    return all_metric, all_responses

def eval_from_data_config(jsonpath,eval_llm_config,
                            max_per_eval_num=100,
                            cutmode="assistant", 
                            keepmode="all"):
    '''
    从数据配置文件中评估模型性能
    
    Args:
        jsonpath: 数据配置文件路径
        eval_llm_config: 评估模型配置
        max_per_eval_num: 每个评估的最大样本数
        cutmode: 消息截断模式，默认"assistant"/"tool_call"
        keepmode: 消息保留模式，默认"all"/"first"/"last"
    
    Returns:
        tuple: (all_metric, all_responses) - 评估结果和响应数据
    '''
    dataseteval_dict = return_datainput(jsonpath,
                    max_per_eval_num=max_per_eval_num,
                    cutmode=cutmode, 
                    keepmode=keepmode)
    all_metric, all_responses = eval_from_data_dict(dataseteval_dict,eval_llm_config)   
    return all_metric, all_responses
    

if __name__ == '__main__':
    import json
    from config import llm_config,llm_config_list,judge_llm_config
    # 选择评估方式：
    # 使用原评估函数（无SwanLab）
    # all_metric, all_responses = eval_from_data_config('config_test_data.json', eval_llm_config=llm_config, max_per_eval_num=2)
    # averages = calculate_metric_averages(all_metric, group_by=['Model','Dataset','Channel'])
    # default_averages.to_csv('default_averages.csv', index=False)

    # 使用带SwanLab功能的评估函数
    all_metric, all_responses= eval_from_data_config_swanlab(
        jsonpath='config_test_data.json',
        eval_llm_config=llm_config_list,
        judge_llm_config=judge_llm_config,
        max_per_eval_num=2,
        cutmode="assistant", 
        keepmode="all", 
        # 以下参数可根据需要调整
        use_swanlab=True,       
        swanlab_tags=[],
        swanlab_token='YOUR_SWANLAB_TOKEN',
        swanlab_project= "model-eval",#'demo_project',
        swanlab_workspace='demo',
        swanlab_lark_webhook_url='YOUR_LARK_WEBHOOK_URL',
        # 飞书Webhook密钥
        swanlab_lark_secret='YOUR_LARK_SECRET',
        swanlab_exp_name=f"all-model-eval"  # 可选，默认为模型名称+eval
    )
    
    # 计算并保存平均分
    default_averages = calculate_metric_averages(all_metric)
    default_averages.to_csv('default_averages.csv', index=False)
    
    # 保存所有指标和响应数据
    all_metric.to_csv('all_metric.csv', index=False)
    all_responses.to_csv('all_responses.csv', index=False)
    
    # 打印结果
    print("\n=== 评估结果 ===")
    print(all_metric)
    print("\n=== 平均分结果 ===")
    print(default_averages)
    print(f"\n=== 响应数据已保存 ===")
    print(f"总共有 {len(all_responses)} 条响应记录，已保存到 all_responses.csv")
    print(f"可以通过 sample_id 关联 all_metric 和 all_responses 数据")


def calculate_all_data_statistics(all_data, all_metric):
    """
    计算all_data的统计信息，包括：
    1. 模型列表和每个模型的测试条目数
    2. 数据集数量和每个数据集的样本数（从Sample/Num列获取并使用.split('/')[0]处理）
    
    Args:
        all_data (pd.DataFrame): 合并后的所有数据
        all_metric (pd.DataFrame): 包含指标的数据框
        
    Returns:
        dict: 结构化的统计信息字典
    """
    import pandas as pd
    import json
    
    # 初始化统计信息字典
    stats_info = {
        "model_stats": {},
        "dataset_stats": {}
    }
    
    # 1. 获取模型列表和每个模型的测试条目数
    if 'Model' in all_data.columns:
        model_counts = all_data['Model'].value_counts()
        stats_info["model_stats"] = {
            "total_models": len(model_counts),
            "model_entries": {}
        }
        for model, count in model_counts.items():
            stats_info["model_stats"]["model_entries"][model] = count
    else:
        stats_info["model_stats"] = {
            "error": "未找到Model列"
        }
    
    # 2. 获取数据集数量和每个数据集的样本数
    if 'Dataset' in all_data.columns and 'Sample/Num' in all_metric.columns:
        # 获取每个数据集的样本数（从Sample/Num列获取并使用.split('/')[0]处理）
        dataset_samples = {}
        for _, row in all_metric.iterrows():
            dataset = row.get('Dataset', 'Unknown')
            sample_num = row.get('Sample/Num', '0/0')
            try:
                # 使用.split('/')[0]获取样本数
                sample_count = int(str(sample_num).split('/')[0])
                if dataset not in dataset_samples or sample_count > dataset_samples[dataset]:
                    dataset_samples[dataset] = sample_count
            except (ValueError, IndexError):
                # 如果解析失败，使用默认值0
                if dataset not in dataset_samples:
                    dataset_samples[dataset] = 0
        
        stats_info["dataset_stats"] = {
            "total_datasets": len(dataset_samples),
            "dataset_samples": {}
        }
        for dataset, sample_count in dataset_samples.items():
            stats_info["dataset_stats"]["dataset_samples"][dataset] = sample_count
    else:
        stats_info["dataset_stats"] = {
            "error": "未找到Dataset或Sample/Num列"
        }
    
    return stats_info

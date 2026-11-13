import json
import pandas as pd

def toolcall2contant(message):
    """
    将工具调用列表转换为字符串。
    """
    content = message.get('content','')
            # messages.append(messages_now)
            # 处理assistant的工具调用
    if message['role'] == 'assistant':
        if message.get('tool_calls', None) and len(message['tool_calls']) > 0:
            for tool_call in message['tool_calls']:
                try:
                    content += "\n<tool_call>\n" + json.dumps({
                        'name': tool_call['function']['name'],
                        'arguments': json.loads(tool_call['function']['arguments'])
                    },ensure_ascii=False)+"\n</tool_call>"
                except:
                    logger.error(f"工具调用格式错误: {tool_call}")
                    pass
        message['content'] = content
        message['tool_calls'] = None
    return message

def contant2toolcall(message):
    """
    将工具调用字符串转换为工具调用列表（toolcall2contant的逆向过程）。
    适配OpenAI的tool_calls格式，包含必要的id和type字段。
    """
    import re
    import json
    import uuid
    content = message.get('content','')
    
    # 处理assistant的工具调用
    if message['role'] == 'assistant':
        tool_calls = []
        
        # 正则表达式匹配工具调用格式：\n<tool_call>\n{json}\n</tool_call>
        pattern = r'\n<tool_call>\n(.*?)\n</tool_call>'
        matches = re.findall(pattern, content, re.DOTALL)
        
        # 解析每个工具调用的JSON数据
        for match in matches:
            try:
                tool_data = json.loads(match)
                # 构建工具调用格式，添加id和type字段以适配OpenAI的格式
                tool_call = {
                    'id': str(uuid.uuid4()),  # 生成唯一ID
                    'type': 'function',      # 设置type为function
                    'function': {
                        'name': tool_data['name'],
                        'arguments': json.dumps(tool_data['arguments'], ensure_ascii=False)
                    }
                }
                tool_calls.append(tool_call)
            except Exception as e:
                print(f"解析工具调用失败: {e}")
                continue
        
        # 从content中移除所有工具调用部分
        new_content = re.sub(pattern, '', content)
        message['content'] = new_content.strip()
        
        # 如果有工具调用，则设置tool_calls字段
        if tool_calls and len(tool_calls) > 0:
            message['tool_calls'] = tool_calls
    
    return message

def dataset_messages_rebuild(dataset):
    """
    对数据集的messages字段进行重建，将工具调用字符串转换为工具调用列表。
    """
    for item in dataset:
        item['messages'] = [contant2toolcall(message) for message in item['messages']]
    return dataset

def normalize_column_name(input_name, available_columns):
    """
    标准化列名，使其大小写不敏感且可以无视连接符
    
    Args:
        input_name: 输入的列名
        available_columns: 可用的列名列表
        
    Returns:
        str or list: 匹配的列名或原始输入
    """
    if isinstance(input_name, list):
        return [normalize_column_name(name, available_columns) for name in input_name]
    
    # 将输入名称转为小写并移除连接符
    def clean_name(name):
        return name.lower().replace('_', '').replace('-', '').replace(' ', '')
    
    cleaned_input = clean_name(input_name)
    
    # 查找匹配的列名
    for col in available_columns:
        if clean_name(col) == cleaned_input:
            return col
    
    # 如果没有找到匹配项，返回原始输入
    return input_name

def calculate_metric_averages(all_metric, group_by=None,metric_column_name = "Metric",score_column_name = "Score",
    metric_keep_names = None,**filters):
    """
    计算pandas数据中不同类别数据的metric平均分
    
    该函数可以灵活地对评估指标数据进行多维度分组统计，支持过滤特定条件的数据，并返回格式化的平均分结果。
    
    参数:
    all_metric (pd.DataFrame): 包含指标数据的DataFrame，需包含以下列：
                              - Model: 模型名称
                              - Dataset: 数据集名称
                              - Metric: 指标类型
                              - Subset: 子集名称
                              - Sample/Num: 样本数量
                              - Score: 得分值
                              - Channel: 通道类型（如'chat'、'tool'等）
    group_by (list, optional): 分组的维度列表，默认为['Model', 'Channel']
                               支持的分组维度包括：'Model', 'Dataset', 'Subset', 'Channel'等
    **filters: 用于筛选数据的键值对，例如：
              - model="Qwen3-0.6B-finetune": 筛选特定模型
              - dataset="self_condition": 筛选特定数据集
              - channel="tool": 筛选特定通道
              - subset="subset_name": 筛选特定子集
    
    返回:
    pd.DataFrame: 按指定维度分组的metric平均分，每行代表一个分组，
                 每列代表一个metric类型的平均分
    
    示例用法:
    --------
    # 默认用法：按模型和通道分组计算所有metric的平均分
    averages = calculate_metric_averages(all_metric)
    
    # 筛选特定模型和数据集
    averages = calculate_metric_averages(
        all_metric,
        model="Qwen3-0.6B-finetune",
        dataset="self_condition"
    )
    
    # 自定义分组维度
    averages = calculate_metric_averages(
        all_metric,
        group_by=['Model', 'Dataset', 'Channel']
    )
    
    # 结合筛选和自定义分组
    averages = calculate_metric_averages(
        all_metric,
        group_by=['Model', 'Subset'],
        channel="tool",
        dataset="bench_self_condition"
    )
    """

    
    # 设置默认分组维度
    if group_by is None:
        group_by = ['Model', 'Channel']
    
    # 复制数据以避免修改原始数据
    df = all_metric.copy()
    
    # 应用过滤条件
    for key, value in filters.items():
        # 转换键为小写用于大小写不敏感的匹配
        key_lower = key.lower()
        
        # 查找匹配的列名（大小写不敏感）
        matching_col = None
        for col in df.columns:
            if col.lower() == key_lower:
                matching_col = col
                break
        
        # 如果找到匹配的列，则应用过滤
        if matching_col:
            df = df[df[matching_col] == value]
    
    # 确保所有分组列都存在
    valid_group_cols = []
    for group_col in group_by:
        # 查找匹配的列名（大小写不敏感）
        for col in df.columns:
            if col.lower() == group_col.lower():
                valid_group_cols.append(col)
                break
    
    if not valid_group_cols:
        raise ValueError("None of the specified group_by columns exist in the DataFrame")
    
    # 按指定维度分组并计算平均分
    # 首先确保Metric和Score列存在（大小写不敏感）
    has_metric = False
    has_score = False
    metric_col = None
    score_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == metric_column_name.lower():
            has_metric = True
            metric_col = col
        elif col_lower == score_column_name.lower():
            has_score = True
            score_col = col
    
    if not has_metric or not has_score:
        raise ValueError(f"DataFrame must contain '{metric_column_name}' and '{score_column_name}' columns")
    
    # 如果指定了metric_keep_names，筛选出这些metric
    if metric_keep_names is not None and len(metric_keep_names) > 0:
        df = df[df[metric_col].isin(metric_keep_names)]
    
    # 使用pivot_table来重塑数据，按valid_group_cols和Metric分组，计算Score的平均值
    result = df.pivot_table(
        index=valid_group_cols,
        columns=metric_col,
        values=score_col,
        aggfunc='mean'
    ).reset_index()
    
    # 对数值列进行格式化，保留四位有效数字
    numeric_cols = result.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        result[col] = result[col].apply(lambda x: '{:.4g}'.format(x) if pd.notna(x) else x)
    
    # 将字符串格式的数字转换回浮点数类型，以便后续处理
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors='coerce')
    
    return result

def safe_json_dumps(x):
    # Handle NaN/None values
    try:
        if pd.isna(x):
            return ''
    except:
        pass
    
    # Try to serialize to JSON, fallback to string representation
    try:
        return json.dumps(x, ensure_ascii=False)
    except:
        return str(x)

def calculate_response_averages(merged_df, numeric_columns=None, group_by=None, **filters):
    """
    计算all_metric和all_response合并后指定数值列的平均值
    
    该函数可以灵活地对评估指标和响应数据进行多维度分组统计，支持过滤特定条件的数据，
    并返回指定数值列的平均值结果。
    
    参数:
    
    merged_df (pd.DataFrame): 合并后的DataFrame，包含指标数据和响应数据all_metric (pd.DataFrame): 包含指标数据的DataFrameall_response (pd.DataFrame): 包含响应数据的DataFrame
    numeric_columns (list, optional): 要统计平均值的数值列名列表，默认为['prompt_tokens', 'completion_tokens', 'total_tokens', 'use_time']
                                     支持的列包括：'prompt_tokens', 'completion_tokens', 'total_tokens', 'use_time'等
    group_by (list, optional): 分组的维度列表，默认为['Model', 'Channel']
                               支持的分组维度包括：'Model', 'Dataset', 'Subset', 'Channel'等
    **filters: 用于筛选数据的键值对，例如：
              - model="Qwen3-0.6B-finetune": 筛选特定模型
              - dataset="self_condition": 筛选特定数据集
              - channel="tool": 筛选特定通道
              - subset="subset_name": 筛选特定子集
    
    返回:
    pd.DataFrame: 按指定维度分组的数值列平均值，每行代表一个分组，
                 每列代表一个数值列的平均值
    
    示例用法:
    --------
    # 默认用法：按模型和通道分组计算默认数值列的平均值
    averages = calculate_response_averages(all_metric, all_response)
    
    # 筛选特定模型和数据集
    averages = calculate_response_averages(
        all_metric,
        all_response,
        model="Qwen3-0.6B-finetune",
        dataset="self_condition"
    )
    
    # 自定义分组维度和数值列
    averages = calculate_response_averages(
        all_metric,
        all_response,
        numeric_columns=['prompt_tokens', 'use_time'],
        group_by=['Model', 'Dataset', 'Channel']
    )
    
    # 结合筛选和自定义分组
    averages = calculate_response_averages(
        all_metric,
        all_response,
        numeric_columns=['total_tokens', 'use_time'],
        group_by=['Model', 'Subset'],
        channel="tool",
        dataset="bench_self_condition"
    )
    """
    
    # 设置默认数值列
    if numeric_columns is None:
        numeric_columns = ['prompt_tokens', 'completion_tokens', 'total_tokens', 'use_time']
    
    # 设置默认分组维度
    if group_by is None:
        group_by = ['Model', 'Channel']
    
    # 复制数据以避免修改原始数据
    df = merged_df.copy()
    
    # 应用过滤条件
    for key, value in filters.items():
        # 转换键为小写用于大小写不敏感的匹配
        key_lower = key.lower()
        
        # 查找匹配的列名（大小写不敏感）
        matching_col = None
        for col in df.columns:
            if col.lower() == key_lower:
                matching_col = col
                break
        
        # 如果找到匹配的列，则应用过滤
        if matching_col:
            df = df[df[matching_col] == value]
    
    # 确保所有分组列都存在
    valid_group_cols = []
    for group_col in group_by:
        # 查找匹配的列名（大小写不敏感）
        for col in df.columns:
            if col.lower() == group_col.lower():
                valid_group_cols.append(col)
                break
    
    if not valid_group_cols:
        raise ValueError("None of the specified group_by columns exist in the DataFrame")
    
    # 确保所有数值列都存在
    valid_numeric_cols = []
    for numeric_col in numeric_columns:
        # 查找匹配的列名（大小写不敏感）
        for col in df.columns:
            if col.lower() == numeric_col.lower():
                valid_numeric_cols.append(col)
                break
    
    if not valid_numeric_cols:
        raise ValueError("None of the specified numeric_columns exist in the DataFrame")
    
    # 按指定维度分组并计算数值列的平均值
    result = df.groupby(valid_group_cols)[valid_numeric_cols].mean().reset_index()
    
    # 对数值列进行格式化，保留四位有效数字
    for col in valid_numeric_cols:
        result[col] = result[col].apply(lambda x: '{:.4g}'.format(x) if pd.notna(x) else x)
    
    # 将字符串格式的数字转换回浮点数类型，以便后续处理
    for col in valid_numeric_cols:
        result[col] = pd.to_numeric(result[col], errors='coerce')
    
    return result


def get_radar_data(df, model_name, dataset_name, channel_name):
    """
    获取雷达图数据
    
    Args:
        df: 包含metric数据的DataFrame
        model_name: 模型名称
        dataset_name: 数据集名称
        channel_name: 通道名称
    
    Returns:
        dict: 包含雷达图数据的字典，包含以下键：
            - metrics: 指标列表
            - values: 对应指标的数值列表
    """

def generate_radar_chart_data(averages, main_column_name, sub_column_name=None, metric_columns=None, auto_delete_metrics=True):
    """
    生成用于SwanLab雷达图绘制的数据
    
    参数:
    averages (pd.DataFrame): 包含平均指标数据的DataFrame
    main_column_name (str): 主分组列名（如'Model'）
    sub_column_name (str, optional): 子分组列名（如'Dataset'或'Channel'）。如果为None，则计算main_column_name所有指标的平均值
    metric_columns (list, optional): 指标列名列表，如果为None则自动检测
    auto_delete_metrics (bool): 是否删除所有main_value为0或无数值的雷达图顶点metric
    
    返回:
    dict: 包含所有雷达图数据的字典，key为sub_value，value为雷达图数据
    字典结构：
    """
    import pandas as pd
    import numpy as np
    
    # 检查必要的列是否存在
    # 当sub_column_name为None时，创建一个虚拟列
    if not sub_column_name:
        # 创建一个虚拟列，所有行的值都为"overall"
        dummy_column_name = "__dummy_sub_column__"
        averages = averages.copy()  # 避免修改原始数据
        averages[dummy_column_name] = "overall"
        sub_column_name = dummy_column_name
    else:
        # 检查main_column_name和sub_column_name是否都存在
        groupby_columns = [main_column_name, sub_column_name]
        if not all(col in averages.columns for col in groupby_columns):
            raise ValueError(f"DataFrame中缺少必要的分组列: {groupby_columns}")
    
    # 获取所有唯一的main_value和sub_value
    main_values = averages[main_column_name].unique()
    sub_values = averages[sub_column_name].unique()
    
    # 初始化结果字典
    radar_data_dict = {}
    
    # 为不同模型定义不同颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_map = {model: colors[i % len(colors)] for i, model in enumerate(main_values)}
    
    for sub_value in sub_values:
        # 筛选当前sub_value的数据
        sub_data = averages[averages[sub_column_name] == sub_value]
        
        # 如果当前sub_value没有数据，跳过
        if len(sub_data) == 0:
            continue
            
        # 获取当前sub_value的所有指标列（排除非指标列）
        if not metric_columns or len(metric_columns) == 0:
            current_metric_columns = [col for col in sub_data.columns if col not in [main_column_name, sub_column_name]]
        else:
            # 标准化metric_columns中的列名
            current_metric_columns = []
            for col_name in metric_columns:
                if col_name not in sub_data.columns:
                    normalized_name = normalize_column_name(col_name, sub_data.columns)
                    if normalized_name != col_name:
                        logger.info(f"列名 '{col_name}' 标准化为 '{normalized_name}'")
                    current_metric_columns.append(normalized_name)
                else:
                    current_metric_columns.append(col_name)
        
        # 如果没有指标列，跳过
        if len(current_metric_columns) == 0:
            continue
            
        # 如果启用auto_delete_metrics，删除所有main_value为0或无数值的指标
        if auto_delete_metrics:
            columns_to_remove = []
            for metric in current_metric_columns:
                # 检查该指标是否所有main_value都是0或NaN
                all_zero_or_nan = True
                for main_value in main_values:
                    main_sub_data = sub_data[sub_data[main_column_name] == main_value]
                    if len(main_sub_data) > 0:
                        value = main_sub_data[metric].values[0]
                        # 检查是否为有效数值且不为0
                        if isinstance(value, (int, float)) and not np.isnan(value) and value != 0:
                            all_zero_or_nan = False
                            break
                        else:
                            # 尝试转换为数值
                            try:
                                converted_value = float(value)
                                if not np.isnan(converted_value) and converted_value != 0:
                                    all_zero_or_nan = False
                                    break
                            except (ValueError, TypeError):
                                pass
                
                if all_zero_or_nan:
                    columns_to_remove.append(metric)
            
            # 从指标列中移除这些无效指标
            current_metric_columns = [col for col in current_metric_columns if col not in columns_to_remove]
            
            # 如果删除后没有指标列了，跳过
            if len(current_metric_columns) == 0:
                continue
        
        # 计算每个指标的最大值作为雷达图的最大值
        max_values = {}
        for metric in current_metric_columns:
            max_values[metric] = sub_data[metric].max()
        
        # 创建雷达图数据结构
        radar_data = {
            'schema': [{"name": metric, "max": max_values[metric]} for metric in current_metric_columns],
            'series': [],
            'sub_value': sub_value
        }
        
        # 按main_value添加数据
        for main_value in main_values:    
            # 从sub_data中筛选当前main_value的数据
            main_sub_data = sub_data[sub_data[main_column_name] == main_value]
            if len(main_sub_data) > 0:
                # try:
                # 获取该main_value在当前sub_value的各指标值，并确保是数值类型
                main_metrics = []
                for metric in current_metric_columns:
                    value = main_sub_data[metric].values[0]
                    # 确保值是数值类型
                    if isinstance(value, (int, float)):
                        main_metrics.append(value)
                    else:
                        # 尝试转换为数值
                        try:
                            main_metrics.append(float(value))
                        except (ValueError, TypeError):
                            # 如果无法转换，使用0作为默认值
                            main_metrics.append(0)
                            logger.warning(f"无法转换指标 {metric} 的值 {value} 为数值，使用默认值0")
                
                # 添加数据到雷达图
                radar_data['series'].append({
                    'name': main_value,
                    'data': main_metrics,
                    'color': color_map[main_value]
                })
                #     logger.info(f"成功添加模型 {main_value} 的数据到 {sub_value} 的雷达图,metric_columns={current_metric_columns}")
                # except Exception as e:
                #     logger.error(f"添加模型 {main_value} 的数据到雷达图时出错: {str(e)}")
        
        # 只有当有数据时才添加到结果字典
        if radar_data['series']:
            # 如果使用了虚拟的sub_column_name，将键名改为更有意义的名称
            if sub_column_name == "__dummy_sub_column__":
                radar_data_dict[main_column_name] = radar_data
            else:
                radar_data_dict[sub_value] = radar_data
    
    return radar_data_dict
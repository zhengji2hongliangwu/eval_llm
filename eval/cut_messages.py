from .utils import toolcall2contant,contant2toolcall

def cut_dataset_message(dataset_item, cutmode="tool_call", keepmode="first"):
    """
    切割messages根据指定的模式
    
    Args:
        dataset_item: 包含messages的数据项
        cutmode: 切割模式
            - "tool_call": 只在遇到包含tool_calls的消息时切割
            - "assistant": 在遇到assistant角色的消息时切割
        keepmode: 保留模式
            - "first": 保留到第一个符合条件为止的消息
            - "last": 保留到最后一个符合条件为止的消息
            - "all": 每个切割节点都保留从开始到当前切割节点的messages（扩充dataset）
    """
    # 确保dataset_item是一个字典且包含messages键
    if not isinstance(dataset_item, dict) or 'messages' not in dataset_item:
        return [dataset_item] if keepmode == "all" else dataset_item
    
    messages = dataset_item['messages']
    if not isinstance(messages, list):
        return [dataset_item] if keepmode == "all" else dataset_item
    
    # 根据cutmode定义切割条件
    def should_cut(message,cut_mode = cutmode):
        if cut_mode == "tool_call":
            flag = False
            if message.get('role', '') == "assistant":
                flag = (len(message.get('tool_calls', [])) > 0) or ("<tool_call>" in message.get('content', '') and "</tool_call>" in message.get('content', ''))
            return flag
        elif cut_mode == "assistant":
            return message.get('role', '') == "assistant"
        else:
            raise ValueError(f"不支持的cutmode: {cut_mode}")
    
    # 根据keepmode处理messages
    if keepmode == "first":
        # 只保留到第一个符合条件为止的消息
        cut_messages = []
        for item in messages:
            cut_messages.append(item)
            if should_cut(item):
                break
        ## 保底使用切割
        if len(cut_messages) == 0:
            cut_messages = messages
        dataset_item['messages'] = cut_messages
        return dataset_item
    
    elif keepmode == "last":
        # 保留到最后一个符合条件为止的消息
        cut_messages = []
        last_cut_index = -1
        
        # 找到最后一个符合条件的位置
        for i, item in enumerate(messages):
            if should_cut(item):
                last_cut_index = i
        
        # 保留到最后一个符合条件的位置
        for i, item in enumerate(messages):
            if i <= last_cut_index:
                cut_messages.append(item)
            else:
                break
        ## 保底使用切割
        if len(cut_messages) == 0:
            cut_messages = messages
        dataset_item['messages'] = cut_messages
        return dataset_item
    
    elif keepmode == "all":
        # 每个符合条件的位置都生成一个切割版本（扩充dataset）
        result_items = []
        cut_indices = []
        
        # 找到所有符合条件的位置
        for i, item in enumerate(messages):
            if should_cut(item):
                cut_indices.append(i)
        
        # 如果没有找到符合条件的消息，返回原始数据
        if not cut_indices:
            return [dataset_item]
        
        # 为每个切割点生成一个数据项
        for cut_index in cut_indices:
            new_item = dataset_item.copy()
            new_item['messages'] = messages[:cut_index + 1]
            result_items.append(new_item)
        return result_items
    
    else:
        raise ValueError(f"不支持的keepmode: {keepmode}")

def cut_datasets_message(dataset_obj, cutmode="assistant", keepmode="all"):
    """
    批量处理数据集的messages切割
    
    Args:
        dataset_obj: 数据集对象（huggingface dataset、列表或其他可迭代对象）
        cutmode: 切割模式（"tool_call" 或 "assistant"）
        keepmode: 保留模式（"first"、"last" 或 "all"）
    """
    dataset_list = []
    try:
        # 尝试将dataset转换为列表进行处理
        if hasattr(dataset_obj, 'to_list'):
            # 如果是huggingface dataset，转换为列表
            data_list = dataset_obj.to_list()
            for dataset_item in data_list:
                data = cut_dataset_message(dataset_item, cutmode, keepmode)
                # 如果keepmode="all"，cut_dataset_message会返回列表
                if isinstance(data, list):
                    dataset_list.extend(data)
                else:
                    dataset_list.append(data)
        elif isinstance(dataset_obj, list):
            # 普通列表处理
            for dataset_item in dataset_obj:
                data = cut_dataset_message(dataset_item, cutmode, keepmode)
                # 如果keepmode="all"，cut_dataset_message会返回列表
                if isinstance(data, list):
                    dataset_list.extend(data)
                else:
                    dataset_list.append(data)
        elif hasattr(dataset_obj, '__iter__') and not isinstance(dataset_obj, dict):
            # 其他可迭代对象（非字典）
            for dataset_item in dataset_obj:
                data = cut_dataset_message(dataset_item, cutmode, keepmode)
                # 如果keepmode="all"，cut_dataset_message会返回列表
                if isinstance(data, list):
                    dataset_list.extend(data)
                else:
                    dataset_list.append(data)
        else:
            # 单个数据项处理
            data = cut_dataset_message(dataset_obj, cutmode, keepmode)
            if isinstance(data, list):
                dataset_list.extend(data)
            else:
                dataset_list.append(data)
                
    except Exception as e:
        print(f"处理数据集时出错: {e}")
        # 出错时返回原始数据
        if isinstance(dataset_obj, list):
            return dataset_obj
        else:
            return [dataset_obj]
    
    return dataset_list


def example_usage():
    """
    示例：展示如何使用不同的cutmode和keepmode
    """
    # 示例数据
    sample_data = {
        'messages': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'},
            {'role': 'assistant', 'content': 'Let me help you with tool calls', 'tool_calls': [{'id': 'call1', 'function': 'some_func'}]},
            {'role': 'assistant', 'content': 'Tool result processing'},
            {'role': 'assistant', 'content': '<tool_call> Tool result processing</tool_call>'},
            {'role': 'assistant', 'content': 'Final response'}
        ]
    }
    
    print("=== cutmode='tool_call', keepmode='first' ===")
    result1 = cut_dataset_message(sample_data.copy(), cutmode="tool_call", keepmode="first")
    print(f"Messages count: {len(result1['messages'])}")
    
    print("\n=== cutmode='assistant', keepmode='first' ===")
    result2 = cut_dataset_message(sample_data.copy(), cutmode="assistant", keepmode="first")
    print(f"Messages count: {len(result2['messages'])}")
    
    print("\n=== cutmode='tool_call', keepmode='last' ===")
    result3 = cut_dataset_message(sample_data.copy(), cutmode="tool_call", keepmode="last")
    print(f"Messages count: {len(result3['messages'])}")
    
    print("\n=== cutmode='tool_call', keepmode='all' ===")
    result4 = cut_dataset_message(sample_data.copy(), cutmode="tool_call", keepmode="all")
    print(f"Generated datasets count: {len(result4)}")
    for i, item in enumerate(result4):
        print(f"Dataset {i+1}: Messages count: {len(item['messages'])}")
    print("\n=== cutmode='assistant', keepmode='all' ===")
    result4 = cut_dataset_message(sample_data.copy(), cutmode="assistant", keepmode="all")
    print(f"Generated datasets count: {len(result4)}")
    for i, item in enumerate(result4):
        print(f"Dataset {i+1}: Messages count: {len(item['messages'])}")

if __name__ == "__main__":
    example_usage()

from inspect import TPFLAGS_IS_ABSTRACT
from typing import Dict, Any, List, Optional
import os
import json

def create_datasetfloder_dict(dataset_folder: str, dataset_names:list|Any=None, sample_num: int=None, split_dataset_ratio: float=0.0,preprocess:str="auto", channel: str='chat') -> Dict[str, Dict[str, Any]]:
    '''
    datasetload_dict = {
    'gesture_1013': {
        'dataset_path': '/workspace/llm/dataset/self_build/origin_data/20251013/custom_dataset_1013/dataset_output/gesture_1013.jsonl',
        # 'enable_channel_loss': True,
        # 'preprocess_func': ToolCallPreprocessor_Custom(),
        
    }
    '''
    datasetload_dict = {}
    if dataset_names is None:
        dataset_names = os.listdir(dataset_folder)
        dataset_names = [dataset_name.split('.jsonl')[0] for dataset_name in dataset_names]
    for dataset_name in dataset_names:
        if sample_num:
            dataset_path = f'{dataset_folder}/{dataset_name}.jsonl#{sample_num}'
        else:
            dataset_path = f'{dataset_folder}/{dataset_name}.jsonl'
        datasetload_dict.update(create_dict(dataset_name, dataset_path, preprocess, split_dataset_ratio, channel))
    return datasetload_dict

def create_dict(dataset_name: str, dataset_path: str, preprocess: str, split_dataset_ratio: float = 0.0, channel: str='chat') -> Dict[str, Dict[str, Any]]:
    return {
        dataset_name: {
            'dataset_path': dataset_path,
            'preprocess': preprocess,
            'split_dataset_ratio': split_dataset_ratio,
            'channel': channel,
        }
    }

# 生成数据集配置字典
datasetload_dict = create_datasetfloder_dict(
    dataset_folder='/workspace/llm/dataset/datasetfinetuning/dataset/dataset_toolcall',
    dataset_names=None,#['gesture_1013', 'initialize_1013'],
    preprocess="custom",
    channel = 'tool',
)
datasetload_dict.update(create_dict(
    dataset_path='/workspace/llm/dataset/datasetfinetuning/dataset/self_condition/self_condition_train.jsonl',
    dataset_name="self_condition",#['gesture_1013', 'initialize_1013'],
    preprocess="auto",
    channel = 'self_condition',
))
datasetload_dict.update(create_dict(
    dataset_path='/workspace/llm/dataset/datasetfinetuning/dataset/dataset_opensource/wild_chat_en_zh_dedup_v3/wild_chat_en_zh_dedup_v3_filter_train_4520.jsonl',
    dataset_name="wild_chat_en_zh_dedup_v3",#['gesture_1013', 'initialize_1013'],
    preprocess="auto",
    channel = 'chat',
))
datasetload_dict.update(create_dict(
    dataset_path='/workspace/llm/dataset/datasetfinetuning/dataset/dataset_opensource/train_0.5M_CN/Belle_open_source_0.5M.head5000.jsonl',
    dataset_name="Belle_open_source_0.5M",#['gesture_1013', 'initialize_1013'],
    preprocess="auto",
    channel = 'chat',
))

# 将字典保存到同目录下的配置文件中
config_file_path = os.path.join(os.path.dirname(__file__), 'config_traing_data.json')
with open(config_file_path, 'w', encoding='utf-8') as f:
    json.dump(datasetload_dict, f, ensure_ascii=False, indent=4)

print(f"配置已保存到: {config_file_path}")
print(datasetload_dict)


# 生成数据集配置字典
dataseteval_dict = create_datasetfloder_dict(
    dataset_folder='/workspace/llm/dataset/datasetfinetuning/dataset/benchmark/bench_toolcall',
    dataset_names=None,#['gesture_1013', 'initialize_1013'],
    preprocess="custom",
    channel = 'tool',
)
dataseteval_dict.update(create_dict(
    dataset_path='/workspace/llm/dataset/datasetfinetuning/dataset/benchmark/self_condition/self_condition_test.jsonl',
    dataset_name="bench_self_condition",#['gesture_1013', 'initialize_1013'],
    preprocess="auto",
    channel = 'self_condition',
))
dataseteval_dict.update(create_dict(
    dataset_path='/workspace/llm/dataset/datasetfinetuning/dataset/benchmark/bench_commensense_chat/bench_commensense.jsonl',
    dataset_name="bench_commensense",#['gesture_1013', 'initialize_1013'],
    preprocess="auto",
    channel = 'chat',
))

# 将字典保存到同目录下的配置文件中
config_file_path = os.path.join(os.path.dirname(__file__), 'config_test_data.json')
with open(config_file_path, 'w', encoding='utf-8') as f:
    json.dump(dataseteval_dict, f, ensure_ascii=False, indent=4)

print(f"配置已保存到: {config_file_path}")
print(dataseteval_dict)


# Copyright (c) Alibaba, Inc. and its affiliates.


from atexit import register
from typing import Any, Dict, Optional
import json 
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import DatasetMeta, ResponsePreprocessor, load_dataset, register_dataset
from dataset import AutoPreprocessor,CustomPreprocessor



'''
# 当前自定义数据格式为openai接口返回的数据格式 messages，并添加了输入的tools字段，输出的messages中有tool_call字段如
{messages:[{role:'user',content:'...'},{role:'tool_call',content:'...',tool_call:[]}],tools:[...]}
'''

datasets_dict = {
    'bench_gesture':{
        "dataset_path":'/workspace/llm/dataset/datasetfinetuning/dataset/benchmark/bench_toolcall/bench_gesture.jsonl',
        "preprocess":"custom",  # 取消注释并使用新的预处理器
        "tags":['tool'],
        "split_dataset_ratio":0.2,
    },
    'bench_commensense':{
        "dataset_path":'/workspace/llm/dataset/datasetfinetuning/dataset/benchmark/bench_commensense_chat/bench_commensense.jsonl',
        "preprocess":"auto",  # 取消注释并使用新的预处理器
        "tags":['chat'],
        "split_dataset_ratio":0.2,
    },
    'selfcondition':{
        "dataset_path":'/workspace/llm/dataset/datasetfinetuning/dataset/self_condition/self_condition_train.jsonl',
        "preprocess":"auto",  # 取消注释并使用新的预处理器
        "tags":['chat'],
        "split_dataset_ratio":0.2,
    },
    'chat':{
        "dataset_path":'/workspace/llm/dataset/datasetfinetuning/dataset/dataset_opensource/wild_chat_en_zh_dedup_v3/wild_chat_en_zh_dedup_v3_filter_train_4520.jsonl',
        "preprocess":"auto",  # 取消注释并使用新的预处理器
        "tags":['chat'],
        "split_dataset_ratio":0.2,
    }
}


def register_datasets(datasets_list:Dict[str,Dict[str,Any]]):
    
    for dataset_name, dataset in datasets_list.items():
        if dataset.get('preprocess') == 'auto':
            preprocess_func = AutoPreprocessor(dataset_sample=dataset.get('dataset_sample',None))
        elif dataset.get('preprocess') == 'custom':
            preprocess_func = CustomPreprocessor(dataset_sample=dataset.get('dataset_sample',None))
        else:
            preprocess_func = None
        register_dataset(
            DatasetMeta(
                dataset_path=dataset['dataset_path'],
                dataset_name=dataset_name,
                preprocess_func=preprocess_func,
                tags=dataset['tags'],
            ))
        
register_datasets(datasets_dict)

def load_datasets(datasets_dict:Dict[str,Dict[str,Any]]):
    datasets = {}
    for dataset_name, dataset in datasets_dict.items():
        data = load_dataset(
            [dataset_name],
            split_dataset_ratio=dataset['split_dataset_ratio'],
        )[0]
        datasets[dataset_name] = {'tags':dataset['tags'],"data":data}
    return datasets
        
datasets = load_datasets(datasets_dict)
for dataset_name, dataset in datasets.items():
    print(f'dataset_name: {dataset_name}')
    print(f'dataset: {dataset}')
    print(f'dataset[0]: {dataset["data"][0]}')



# register_dataset(
#     DatasetMeta(
#         dataset_path='/workspace/llm/train/LLaMA-Factory/data/glaive_toolcall_en_demo.json',
#         dataset_name='glaive_toolcall_en_demo',
#         preprocess_func=AutoPreprocessor(),  # 取消注释并使用新的预处理器
#         tags=['tool']
#     ))
# dataset = load_dataset(['glaive_toolcall_en_demo#10'],split_dataset_ratio=0.1)
# dataset = load_dataset(['AI-ModelScope/leetcode-solutions-python#10'],split_dataset_ratio=0.1)






# if __name__ == '__main__':
#     pass
    
#     # 测试数据集加载，load_dataset默认兼容标准的alpaca，sharegpt，以及openai messages格式
    
#     dataset = load_dataset(['gesture_1013', 'initialize_1013'],split_dataset_ratio=0.1)
#     print(f'dataset_deal: {dataset}')
#     dataset = dataset[0]
#     print(f'dataset: {dataset}')
#     print(f'dataset[0]: {dataset[0]}')
    

#     '''
#     load_dataset 参数
#     datasets: Union[List[str], str], #注册好的书籍名称
#     *,
#     split_dataset_ratio: float = 0.,  # 数据集分割比例
#     seed: Union[int, np.random.RandomState, None] = 42,  # 随机种子
#     num_proc: int = 1,  # 处理进程数
#     load_from_cache_file: bool = True,  # 是否从缓存文件加载
#     shuffle: bool = False,  # 是否打乱数据
#     streaming: bool = False,  # 是否使用流式处理
#     interleave_prob: Optional[List[float]] = None,  # 交错概率
#     stopping_strategy: Literal['first_exhausted', 'all_exhausted'] = 'first_exhausted',  # 停止策略
#     shuffle_buffer_size: int = 1000,  # 打乱缓冲区大小
#     use_hf: Optional[bool] = None,  # 是否使用HuggingFace
#     hub_token: Optional[str] = None,  # Hub令牌
#     strict: bool = False,  # 严格模式
#     download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',  # 下载模式
#     columns: Optional[Dict[str, str]] = None,  # 列映射
#     remove_unused_columns: bool = True,  # 是否移除未使用的列
#     # 自我认知
#     model_name: Optional[Union[Tuple[str, str], List[str]]] = None,  # 模型名称（中文，英文）
#     model_author: Optional[Union[Tuple[str, str], List[str]]] = None,  # 模型作者（中文，英文）
#     '''

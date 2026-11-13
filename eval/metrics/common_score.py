import numpy as np
from typing import List, Dict
from tabulate import tabulate
import pandas as pd

# 导入metrics.py和rouge_metric.py中的方法
from .metrics import bleu_ngram_one_sample
from .rouge_metric import compute_rouge_score_one_sample_zh

# tokenize_mixed_text函数已不再需要，因为分词逻辑已包含在metrics.py和rouge_metric.py的方法中


def compute_bleu_scores(hypotheses: List[str], references: List[List[str]]) -> Dict[str, List[float]]:
    """计算每一对数据的1-4阶BLEU分数，返回包含各阶分数列表的字典
    支持中英文混合文本，使用metrics.py中的bleu_ngram_one_sample方法
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses与references长度必须一致")
    
    # 初始化每个n-gram级别的分数列表
    bleu_1_scores = []
    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []
    
    # 对每一对数据计算BLEU分数
    for hyp, refs in zip(hypotheses, references):
        try:
            # 取第一个参考文本进行计算（metrics.py中的方法支持单个参考）
            ref = refs[0] if isinstance(refs, list) and len(refs) > 0 else ""
            
            # 使用metrics.py中的bleu_ngram_one_sample计算BLEU分数
            bleu_scores = bleu_ngram_one_sample(hyp, ref)
            
            # 提取1-4阶n-gram的分数，保留4位小数
            bleu_1_scores.append(round(bleu_scores['bleu-1'], 4))
            bleu_2_scores.append(round(bleu_scores['bleu-2'], 4))
            bleu_3_scores.append(round(bleu_scores['bleu-3'], 4))
            bleu_4_scores.append(round(bleu_scores['bleu-4'], 4))
        except Exception as e:
            # 如果计算失败，填充0分
            print(f"计算单句BLEU失败: {e}")
            bleu_1_scores.append(0.0)
            bleu_2_scores.append(0.0)
            bleu_3_scores.append(0.0)
            bleu_4_scores.append(0.0)
    
    # 返回包含所有对分数列表的字典
    return {
        "BLEU-1": bleu_1_scores,
        "BLEU-2": bleu_2_scores,
        "BLEU-3": bleu_3_scores,
        "BLEU-4": bleu_4_scores
    }


def compute_rouge_scores(hypotheses: List[str], references: List[str]) -> Dict[str, List[float]]:
    """计算ROUGE-1、ROUGE-2、ROUGE-L的R/P/F分数列表
    支持中英文混合文本，使用rouge_metric.py中的compute_rouge_score_one_sample方法
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses与references长度必须一致")
    
    # 初始化ROUGE分数列表
    rouge1_r, rouge1_p, rouge1_f = [], [], []
    rouge2_r, rouge2_p, rouge2_f = [], [], []
    rougel_r, rougel_p, rougel_f = [], [], []
    
    for hyp, ref in zip(hypotheses, references):
        try:
            # 使用rouge_metric.py中的compute_rouge_score_one_sample计算ROUGE分数
            # 需要将输入转换为列表格式以符合函数要求
            rouge_scores = compute_rouge_score_one_sample_zh([hyp], [ref])
            
            # 注意：rouge_metric.py返回的键是大写的，需要使用正确的键名
            rouge1_r.append(round(rouge_scores.get('Rouge-1-R', 0.0), 4))
            rouge1_p.append(round(rouge_scores.get('Rouge-1-P', 0.0), 4))
            rouge1_f.append(round(rouge_scores.get('Rouge-1-F', 0.0), 4))
            rouge2_r.append(round(rouge_scores.get('Rouge-2-R', 0.0), 4))
            rouge2_p.append(round(rouge_scores.get('Rouge-2-P', 0.0), 4))
            rouge2_f.append(round(rouge_scores.get('Rouge-2-F', 0.0), 4))
            rougel_r.append(round(rouge_scores.get('Rouge-L-R', 0.0), 4))
            rougel_p.append(round(rouge_scores.get('Rouge-L-P', 0.0), 4))
            rougel_f.append(round(rouge_scores.get('Rouge-L-F', 0.0), 4))
        except Exception as e:
            # 如果计算失败，填充0分
            print(f"计算ROUGE失败: {e}")
            rouge1_r.append(0.0)
            rouge1_p.append(0.0)
            rouge1_f.append(0.0)
            rouge2_r.append(0.0)
            rouge2_p.append(0.0)
            rouge2_f.append(0.0)
            rougel_r.append(0.0)
            rougel_p.append(0.0)
            rougel_f.append(0.0)
    
    return {
        "Rouge-1-R": rouge1_r,
        "Rouge-1-P": rouge1_p,
        "Rouge-1-F": rouge1_f,
        "Rouge-2-R": rouge2_r,
        "Rouge-2-P": rouge2_p,
        "Rouge-2-F": rouge2_f,
        "Rouge-L-R": rougel_r,
        "Rouge-L-P": rougel_p,
        "Rouge-L-F": rougel_f
    }



def evaluate_texts(
    hypotheses: List[str],
    references: List[str],
    model_name: str = "default",
    dataset_name: str = "default",
    subset_name: str = "default",
    channel: str = "default",
    per_sample: bool = False,
    show_table: bool = False
) -> None:
    """整合所有指标并输出表格，show_per_sample=True时显示每个样本的分数"""
    num_samples = len(hypotheses)
    if num_samples == 0:
        raise ValueError("输入样本不能为空")
    
    # 转换参考文本为多参考格式（sacrebleu要求List[List[str]]）
    references_multi = [[ref] for ref in references]
    
    # 计算指标
    bleu_scores = compute_bleu_scores(hypotheses, references_multi)
    rouge_scores = compute_rouge_scores(hypotheses, references)
    
    # 生成表格
    table_data = []
    
    if per_sample:
        # 显示每个样本的分数
        for i in range(num_samples):
            # 添加BLEU分数
            for metric_name, scores_list in bleu_scores.items():
                table_data.append([
                    model_name, dataset_name, metric_name, subset_name, f"{i+1}/{num_samples}", scores_list[i], channel
                ])
            # 添加ROUGE分数
            for metric_name, scores_list in rouge_scores.items():
                table_data.append([
                    model_name, dataset_name, metric_name, subset_name, f"{i+1}/{num_samples}", scores_list[i], channel
                ])
    else:
        # 显示平均分数
        all_metrics = {}
        # 计算BLEU平均分数
        for metric_name, scores_list in bleu_scores.items():
            all_metrics[f"mean_{metric_name}"] = round(np.mean(scores_list), 4)
        # 计算ROUGE平均分数
        for metric_name, scores_list in rouge_scores.items():
            all_metrics[metric_name] = round(np.mean(scores_list), 4)
        
        # 添加到表格数据
        for metric_name, score in all_metrics.items():
            table_data.append([
                model_name, dataset_name, metric_name, subset_name, f"{num_samples}", score, channel
            ])
    
    headers = ["Model", "Dataset", "Metric", "Subset", "Sample/Num", "Score", "Channel"]
    if show_table:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # 转换为DataFrame
    df = pd.DataFrame(table_data, columns=headers)
    
    return df


# test
if __name__ == "__main__":
    # 添加中英文混合的测试例子
    hypotheses = [
        "你好 world",
        "这是一个中英文混合例子",
        "我喜欢 programming and coding",
        # "北京 is the capital of China",
        # "今天天气很好 today is sunny",
        # "machine learning 很有趣",
        # "i like ml i love machine learning",
        # "hello world hi there",
        # "the sky is blue",
        # "1+1=2"
    ]
    
    references = [
        "你好 world",
        "这是一个中英文混合例子",
        "我喜欢 programming",
        # "Beijing is the capital of China",
        # "今天天气很好",
        # "machine learning is interesting",
        # "i love machine learning",
        # "hi there",
        # "blue sky",
        # "1+1=2"
    ]
    
    print("=== 中英文混合测试 ===")
    evaluate_texts(
        model_name="qwen-plus",
        dataset_name="general_qa",
        subset_name="mixed_language",
        hypotheses=hypotheses,
        references=references,
        per_sample=True
    )
    
    # 同时显示平均值
    print("\n=== 中英文混合测试 - 平均值 ===")
    evaluate_texts(
        model_name="qwen-plus",
        dataset_name="general_qa",
        subset_name="mixed_language",
        hypotheses=hypotheses,
        references=references,
        per_sample=False
    )
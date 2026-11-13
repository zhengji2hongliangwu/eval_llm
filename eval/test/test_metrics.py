#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/workspace/llm/eval/eval_rebuild/eval')

from metrics.common_score import evaluate_texts

if __name__ == "__main__":
    print("=== 测试修复后的ROUGE计算 ===")
    
    # 测试用例
    hypotheses = [
        # "你好 world",
        # "这是一个中英文混合例子",
        # "我喜欢 programming and coding",
        # "dasf  daff fff",
        '\n\n\n\n<tool_call>\n{"name": "move_robot", "arguments": {"velocity": 0.8}}\n</tool_call>', 
        '\n\n\n\n<tool_call>\n{"name": "move_robot", "arguments": {"velocity": 0.8}}\n</tool_call>'
    ]
    
    references = [
        # "你好 world",
        # "这是一个中英文混合例子", 
        # "我喜欢 programming",
        # "dasf  daff fff",
        '\n<tool_call>\n{"name": "move_robot", "arguments": {"velocity": 0.8}}\n</tool_call>', 
        '好的，正在全速前进啦，请注意避让哦～'
    ]
    
    print("测试文本:")
    for i, (hyp, ref) in enumerate(zip(hypotheses, references)):
        print(f"  样本 {i+1}:")
        print(f"    预测: {hyp}")
        print(f"    参考: {ref}")
    print()
    
    print("=== 修复后的评估结果 ===")
    result = evaluate_texts(
        model_name="qwen-plus",
        dataset_name="general_qa", 
        subset_name="mixed_language",
        hypotheses=hypotheses,
        references=references,
        per_sample=True,
        show_table=True
    )
    print(result)
    print("\n=== 检查ROUGE-1-R分数 ===")
    rouge_1_r_scores = result[result['Metric'] == 'Rouge-1-R']['Score'].tolist()
    print(f"ROUGE-1-R分数: {rouge_1_r_scores}")


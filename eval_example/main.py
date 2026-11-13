import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import llm_config,llm_config_list,judge_llm_config
from eval.eval_run import eval_from_data_config_swanlab,eval_from_data_config



# 选择评估方式：
# 使用原评估函数（无SwanLab）
# all_metric, all_responses = eval_from_data_config('config_test_data.json', eval_llm_config=llm_config, max_per_eval_num=2)
# averages = calculate_metric_averages(all_metric, group_by=['Model','Dataset','Channel'])
# default_averages.to_csv('default_averages.csv', index=False)

# 使用带SwanLab功能的评估函数
all_metric, all_responses = eval_from_data_config_swanlab(
    jsonpath='config_test_1.json',
    eval_llm_config=llm_config_list,
    judge_llm_config=judge_llm_config,
    max_per_eval_num=10,
    cutmode="assistant", 
    keepmode="all", 
    save_path= "./output",
    # 以下参数可根据需要调整
    use_swanlab=True,       
    swanlab_tags=[],
    swanlab_token='YOUR_SWANLAB_TOKEN',
    swanlab_project= 'test',# "model-eval",#,
    swanlab_workspace='demo',
    swanlab_lark_webhook_url='YOUR_LARK_WEBHOOK_URL',
    # 飞书Webhook密钥
    swanlab_lark_secret='YOUR_LARK_SECRET',
    swanlab_exp_name=f"all-model-eval",  # 可选，默认为模型名称+eval
    execution_mode='sequential', # 可选async/concurrent/sequential，默认为concurrent
    max_workers=10, 
    max_concurrency=10
)

# 计算并保存平均分
# default_averages = calculate_metric_averages(all_metric)
# default_averages.to_csv('./output/default_averages.csv', index=False)
os.makedirs('./output', exist_ok=True)
# 保存所有指标和响应数据
all_metric.to_csv('./output/all_metric.csv', index=False)
all_responses.to_csv('./output/all_responses.csv', index=False)

# 打印结果
print("\n=== 评估结果 ===")
print(all_metric)
# print("\n=== 平均分结果 ===")
# print(default_averages)
print(f"\n=== 响应数据已保存 ===")
print(f"总共有 {len(all_responses)} 条响应记录，已保存到 all_responses.csv")
print(f"可以通过 sample_id 关联 all_metric 和 all_responses 数据")

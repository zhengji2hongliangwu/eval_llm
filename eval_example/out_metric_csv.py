
import pandas as pd
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.utils import (toolcall2contant,
                        contant2toolcall,
                        calculate_metric_averages,
                        calculate_response_averages,
                        generate_radar_chart_data,
                        normalize_column_name,
                        safe_json_dumps,
                        dataset_messages_rebuild)
all_metric = pd.read_csv('./output/all_metric.csv')
all_responses = pd.read_csv('./output/all_responses.csv')

id_metric = 'sample_uuid'
id_response = 'sample_id'

all_data = pd.merge(all_metric, all_responses, left_on=id_metric, right_on=id_response, how='left')

groupby_columns = ['Model','Dataset'] # ,'Channel'
metric_keep_names = []
metric_column_name = 'Metric'
score_column_name = 'Score'
averages = calculate_metric_averages(all_metric,group_by=groupby_columns,metric_columns=metric_keep_names,metric_column_name = metric_column_name,score_column_name=score_column_name)



numeric_columns = ['use_time','prompt_tokens','completion_tokens','total_tokens']
numeric_columns = normalize_column_name(numeric_columns, all_responses.columns)
response_averages = calculate_response_averages(all_data,group_by=groupby_columns,numeric_columns=numeric_columns)
response_averages.to_csv('./output/response_averages_Dataset.csv', index=False)
averages.to_csv('./output/averages_Dataset.csv', index=False)



groupby_columns = ['Model','Channel']
averages = calculate_metric_averages(all_metric,group_by=groupby_columns,metric_columns=metric_keep_names,metric_column_name = metric_column_name,score_column_name=score_column_name)

numeric_columns = ['use_time','prompt_tokens','completion_tokens','total_tokens']
numeric_columns = normalize_column_name(numeric_columns, all_responses.columns)
response_averages = calculate_response_averages(all_data,group_by=groupby_columns,numeric_columns=numeric_columns)
response_averages.to_csv('./output/response_averages_Channel.csv', index=False)
averages.to_csv('./output/averages_Channel.csv', index=False)


groupby_columns = ['Model']
averages = calculate_metric_averages(all_metric,group_by=groupby_columns,metric_columns=metric_keep_names,metric_column_name = metric_column_name,score_column_name=score_column_name)
numeric_columns = ['use_time','prompt_tokens','completion_tokens','total_tokens']
numeric_columns = normalize_column_name(numeric_columns, all_responses.columns)
response_averages = calculate_response_averages(all_data,group_by=groupby_columns,numeric_columns=numeric_columns)
response_averages.to_csv('./output/response_averages_all.csv', index=False)
averages.to_csv('./output/averages_all.csv', index=False)




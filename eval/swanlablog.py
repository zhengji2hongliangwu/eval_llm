import swanlab
import pandas as pd
from loguru import logger

logger.add("./logs/swanlablog.log", rotation="10 MB", encoding="utf-8")

def swanlab_log_df(averages, df_name):
    """
    将评估结果数据记录到SwanLab表格中
    
    参数:
    averages (pd.DataFrame): 包含评估结果平均值的DataFrame
    df_name (str): 数据框名称，用于命名日志项
    
    返回:
    None
    """
    try:
        # 创建echarts table对象
        table = swanlab.echarts.Table()
        # 获取表头和数据
        headers = list(averages.columns)
        rows = averages.values.tolist()
        # 添加数据到表格
        table.add(headers, rows)
        # 记录表格到swanlab
        swanlab.log({df_name: table})
        # logger.info("评估结果表格已成功记录到SwanLab")
    except Exception as e:
        logger.error(f"记录评估结果表格到SwanLab时出错: {str(e)}")
        raise


def swanlab_log_radar(radar_data_dict,name_title="eval_radar"):
    """
    将雷达图数据记录到SwanLab中
    
    参数:
    radar_data_dict (dict): 包含雷达图数据的字典
        结构如下:
        {
            'sub_value': {
                'schema': [
                    {'name': 'metric1', 'max': max_value1},
                    {'name': 'metric2', 'max': max_value2},
                    ...
                ],
                'series': [
                    {
                        'name': 'model1',           # 系列名称
                        'data': [value1, value2, ...], # 各指标的数值
                        'color': '#1f77b4'          # 颜色（可选）
                    },
                    ...
                ]
            },
            ...
        }
        
    返回:
    None
    """
    try:
        # 遍历所有雷达图数据
        for sub_value, radar_data in radar_data_dict.items():
            # 创建雷达图
            radar = swanlab.echarts.Radar()
            
            # 设置雷达图维度与数据范围
            radar.add_schema(schema=radar_data['schema'])
            
            # 按系列添加数据
            for series_data in radar_data['series']:
                radar.add(
                    series_data['name'],        # 系列名称
                    [series_data['data']],      # 数据（注意要嵌套在列表中）
                    color=series_data.get('color', None),  # 设置颜色（可选）
                )
            
            # 记录雷达图到swanlab
            swanlab.log({f"{name_title}/{sub_value}": radar})
            logger.info(f"{sub_value} 的雷达图已成功记录到SwanLab")
    except Exception as e:
        logger.error(f"记录雷达图到SwanLab时出错: {str(e)}")
        # raise
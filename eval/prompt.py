"""提示词定义模块，用于LLM评估工具调用指标和自我认知一致性"""

# 自我认知一致性评估提示词
SELF_CONDITION_METRICS_PROMPT = """
你是一个专业的评估专家，负责评估模型输出和参考输出对自我认知的一致性。请你严格分析以下内容：

任务：判断模型回答（res_message）中关于自我认知的描述是否与参考回答（ref_message）中的自我认知一致。
自我认知包括但不限于：模型名称、开发者/公司、版本信息、训练数据来源等关于模型自身身份的关键信息。

评分标准：
- 如果模型回答中的自我认知描述与参考回答中的自我认知描述完全一致，得1分
- 如果模型回答中的自我认知描述与参考回答中的自我认知描述不一致，得0分
- 如果模型回答中没有明确的自我认知描述，得0分

请基于以下信息进行评分：

参考回答（ref_message）: {ref_message}
模型回答（res_message）: {res_message}

请严格按照以下JSON格式输出评估结果，不要输出任何其他内容：
{{
    "score": 0或1
}}

注意：
1. 请确保输出格式严格为JSON，不包含任何额外文本
2. 请重点关注模型名称、开发者等核心身份信息的一致性
3. 忽略非关键信息的差异，如标点符号、大小写等细微差别
"""

# 工具调用指标评估提示词
TOOL_CALL_METRICS_PROMPT = """
你是一个专业的评估专家，负责评估模型在工具调用任务中的表现。请你基于参考回答（ref_message）和模型回答（res_message），对以下四个指标进行评分：

1. ToolCallUseAccuracy：是否调用工具判断的准确性（0分或1分）
   - 评分标准：如果参考回答中没有工具调用，而模型回答也没有工具调用，得1分；如果参考回答中有工具调用，而模型回答也有工具调用，得1分；其他情况得0分

2. ToolFormatAccuracy：调用工具格式准确性（0分或1分）
   - 评分标准：模型回答中的工具调用必须包含所有必要字段，且function中包含name和arguments字段；当工具调用数量相同时，参数键也需要匹配。完全符合得1分，否则得0分

3. ToolCategoryNameAccuracy：工具调用类别和名称准确性（0分或1分）
   - 评分标准：模型回答中的工具调用名称集合与参考回答中的工具调用名称集合完全相同得1分，否则得0分

4. ToolParamValueAccuracy：工具调用参数值准确性（0分或1分）
   - 评分标准：模型回答中的工具调用参数值与参考回答中的对应工具调用参数值完全相同得1分，否则得0分

请基于以下信息进行评分：

参考回答（ref_message）: {ref_message}
模型回答（res_message）: {res_message}

请严格按照以下JSON格式输出评估结果，不要输出任何其他内容：
{{
    "ToolCallUseAccuracy": 0或1,
    "ToolFormatAccuracy": 0或1或null,
    "ToolCategoryNameAccuracy": 0或1或null,
    "ToolParamValueAccuracy": 0或1或null
}}

注意：
1. 当ToolCallUseAccuracy为0时，其他三个指标应为null
2. 当参考回答中没有工具调用且模型回答也没有工具调用时，所有指标均为null
3. 请确保输出格式严格为JSON，不包含任何额外文本
"""

# 构建评估消息的函数
def build_evaluation_messages(ref_message, res_message, metric_type="tool_call"):
    """
    构建用于评估的消息列表
    
    Args:
        ref_message: 参考消息
        res_message: 模型消息
        metric_type: 评估类型，可选值："tool_call"（工具调用评估）或"self_condition"（自我认知评估）
        
    Returns:
        list: 消息列表
    """
    if metric_type == "self_condition":
        return [
            {
                "role": "system",
                "content": "你是一个专业的评估专家，负责评估模型输出和参考输出对自我认知的一致性。"
            },
            {
                "role": "user",
                "content": SELF_CONDITION_METRICS_PROMPT.format(
                    ref_message=str(ref_message),
                    res_message=str(res_message)
                )
            }
        ]
    else:  # 默认工具调用评估
        return [
            {
                "role": "system",
                "content": "你是一个专业的评估专家，负责评估模型在工具调用任务中的表现。"
            },
            {
                "role": "user",
                "content": TOOL_CALL_METRICS_PROMPT.format(
                    ref_message=str(ref_message),
                    res_message=str(res_message)
                )
            }
        ]
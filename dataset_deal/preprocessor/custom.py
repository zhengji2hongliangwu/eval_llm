from .core import  ResponsePreprocessor,AutoPreprocessor
from typing import Dict, Any, Optional
import json
class CustomPreprocessor(ResponsePreprocessor):
    """处理工具调用数据的预处理器"""
    
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 主要的数据处理逻辑
        '''
        将tool_call转换为messages格式拼接在assistant的content中
        '''
        output = {"messages": [], "tools": []}
        output["tools"] = row.get("tools", [])
        output["channel"] = []
        messages = []
        
        for i, message in enumerate(row['messages']):
            # 添加基础消息
            messages_now = message
            # messages.append(messages_now)
            # 处理assistant的工具调用
            if message['role'] == 'assistant':
                if message.get('tool_calls', None) and len(message['tool_calls']) > 0:
                    for tool_call in message['tool_calls']:
                        messages_now['content'] += "\n<tool_call>\n" + json.dumps({
                            'name': tool_call['function']['name'],
                            'arguments': json.loads(tool_call['function']['arguments'])
                        },ensure_ascii=False)+"\n</tool_call>"
                    output["channel"].append("tool")  # 可以兼容channel loss技术
                else:
                    output["channel"].append("chat")  # 可以兼容channel loss技术
            if messages_now['role'] in ['tool','tool_response','function_response','observation']:
                messages_now['role'] = 'tool_response'  # 可以兼容channel loss技术
            messages.append(messages_now)
        output["messages"] = messages
        return output


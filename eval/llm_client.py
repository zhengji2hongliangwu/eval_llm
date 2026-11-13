import asyncio
from typing import List, Dict, Any, Optional
from openai import OpenAI
from openai import AsyncOpenAI
from loguru import logger

class OpenAIClient:
    """OpenAI API客户端"""
    
    def __init__(self, base_url: str, api_key: str, model: str = 'qwen'):
        """
        初始化OpenAI客户端
        
        Args:
            base_url: API接口URL
            api_key: API密钥
            model: 模型名称 
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        # 初始化异步OpenAI客户端
        self.async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       tools: Optional[List[Dict[str, Any]]] = None,
                       model: str = None,
                       temperature: float = 0.7,
                       max_tokens: int = 2048,
                       top_p: float = 0.9,
                       max_retries: int = 1,
                    #    frequency_penalty: float = 0.0,
                    #    presence_penalty: float = 0.0
                       ) -> Dict:
        """
        调用聊天完成接口
        
        Args:
            messages: 对话消息列表
            tools: 可用工具列表
            temperature: 生成文本的随机性
            max_tokens: 最大生成长度
            top_p: 核采样参数
            max_retries: 最大重试次数
            frequency_penalty: 频率惩罚
            presence_penalty: 存在惩罚
            
        Returns:
            模型响应的choices[0]内容字典
        """
        if not model:
            model = self.model
        # 准备请求参数
        request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "extra_body": {"enable_thinking": False},
            # "frequency_penalty": frequency_penalty,
            # "presence_penalty": presence_penalty
        }
        
        # 如果提供了工具，添加到请求参数中
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = "auto"
        
        # 调用API，添加重试机制
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(**request_params)
                return response
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"请求失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}，正在重试...")
                else:
                    logger.error(f"请求失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}，已达到最大重试次数")
                    raise
    
    async def achat_completion(self, messages: List[Dict[str, str]], 
                               model: str = None,
                               tools: Optional[List[Dict[str, Any]]] = None,
                               temperature: float = 0.7,
                               max_tokens: int = 2048,
                               top_p: float = 0.9,
                               max_retries: int = 1,
                        #    frequency_penalty: float = 0.0,
                        #    presence_penalty: float = 0.0
                           ) -> Dict:
        """
        异步调用聊天完成接口
        
        Args:
            messages: 对话消息列表
            tools: 可用工具列表
            temperature: 生成文本的随机性
            max_tokens: 最大生成长度
            top_p: 核采样参数
            max_retries: 最大重试次数
            frequency_penalty: 频率惩罚
            presence_penalty: 存在惩罚
            
        Returns:
            模型响应的choices[0]内容字典
        """
        if not model:
            model = self.model
        # 准备请求参数
        request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            'extra_body':{'enable_thinking': False},
            # "frequency_penalty": frequency_penalty,
            # "presence_penalty": presence_penalty
        }
        
        # 如果提供了工具，添加到请求参数中
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = "auto"
        
        # 异步调用API，添加重试机制
        for attempt in range(max_retries + 1):
            try:
                response = await self.async_client.chat.completions.create(**request_params)
                return response
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"请求失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}，正在重试...")
                else:
                    logger.error(f"请求失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}，已达到最大重试次数")
                    raise


# 简化的示例函数
def simple_example():
    """
    简单的使用示例
    """
    from config import llm_config
    client = OpenAIClient(
        base_url=llm_config["base_url"],
        api_key=llm_config["api_key"],
        model=llm_config["model"]
    )
    
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "请解释什么是人工智能？20字"}
    ]
    
    # 调用API并获取响应
    response_data = client.chat_completion(messages=messages)
    print(response_data)


if __name__ == "__main__":
    simple_example()
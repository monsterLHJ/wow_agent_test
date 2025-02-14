import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import re

# 加载环境变量
load_dotenv()
# 从环境变量中读取api_key
api_key = os.getenv('QWEN_API_KEY')
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
chat_model = "qwen-max"
emb_model = "embedding-3"

client = OpenAI(
    api_key = api_key,
    base_url = base_url
)

from openai import OpenAI
from pydantic import Field  # 导入Field，用于Pydantic模型中定义字段的元数据
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    LLMMetadata,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms.callbacks import llm_completion_callback
from typing import List, Any, Generator
# 定义OurLLM类，继承自CustomLLM基类
class OurLLM(CustomLLM):
    api_key: str = Field(default=api_key)
    base_url: str = Field(default=base_url)
    model_name: str = Field(default=chat_model)
    client: OpenAI = Field(default=None, exclude=True)  # 显式声明 client 字段

    def __init__(self, api_key: str, base_url: str, model_name: str = chat_model, **data: Any):
        super().__init__(**data)
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)  # 使用传入的api_key和base_url初始化 client 实例

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        if hasattr(response, 'choices') and len(response.choices) > 0:
            response_text = response.choices[0].message.content
            return CompletionResponse(text=response_text)
        else:
            raise Exception(f"Unexpected response format: {response}")

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> Generator[CompletionResponse, None, None]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        try:
            for chunk in response:
                chunk_message = chunk.choices[0].delta
                if not chunk_message.content:
                    continue
                content = chunk_message.content
                yield CompletionResponse(text=content, delta=content)

        except Exception as e:
            raise Exception(f"Unexpected response format: {e}")

llm = OurLLM(api_key=api_key, base_url=base_url, model_name=chat_model)

response = llm.stream_complete("你是谁？")
for chunk in response:
    print(chunk, end="", flush=True)

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

def get_weather(city: str) -> int:
    """
    Gets the weather temperature of a specified city.

    Args:
    city (str): The name or abbreviation of the city.

    Returns:
    int: The temperature of the city. Returns 20 for 'NY' (New York),
         30 for 'BJ' (Beijing), and -1 for unknown cities.
    """

    # Convert the input city to uppercase to handle case-insensitive comparisons
    city = city.upper()

    # Check if the city is New York ('NY')
    if city == "NY":
        return 20  # Return 20°C for New York

    # Check if the city is Beijing ('BJ')
    elif city == "BJ":
        return 30  # Return 30°C for Beijing

    # If the city is neither 'NY' nor 'BJ', return -1 to indicate unknown city
    else:
        return -1


def main():

    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    add_tool = FunctionTool.from_defaults(fn=add)
    weather_tool = FunctionTool.from_defaults(fn=get_weather)

    # 创建ReActAgent实例
    agent = ReActAgent.from_tools([multiply_tool, add_tool, weather_tool], llm=llm, verbose=True)

    response = agent.chat("纽约天气怎么样?")

    print(response)


if __name__ == "__main__":
    main()
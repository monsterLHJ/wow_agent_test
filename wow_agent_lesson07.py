import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
# 从环境变量中读取api_key
api_key = os.getenv('QWEN_API_KEY')
ZHIPU_API_KEY = os.getenv('ZISHU_API_KEY')
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
chat_model = "qwen-max"
emb_model = "embedding-3"

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

#response = llm.stream_complete("你是谁？")
#for chunk in response:
#    print(chunk, end="", flush=True)

from datetime import datetime
# 定义千问联网搜索工具
def qwen_web_search_tool(query: str) -> str:
    """
    使用通义千问max模型进行联网搜索，返回搜索结果的字符串。
    
    参数:
    - query: 搜索关键词

    返回:
    - 搜索结果的字符串形式
    """
    # 初始化客户端
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 获取当前日期
    current_date = datetime.now().strftime("%Y-%m-%d")

    print("current_date:", current_date)
    
    # 系统提示模板，包含时间信息
    system_prompt = f"""你是一个具备网络访问能力的智能助手，在适当情况下，优先使用网络信息来回答，
    以确保用户得到最新、准确的帮助。当前日期是 {current_date}。"""
        
    # 构建消息
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
        
    # 调用API
    response = client.chat.completions.create(
        model=chat_model,
        messages=messages,
        tools=[{"type": "web_search"}]
    )
    
    # 返回结果
    return response.choices[0].message.content

# 修改测试代码
rst = qwen_web_search_tool("2025年为什么会闰六月？")
print(rst)
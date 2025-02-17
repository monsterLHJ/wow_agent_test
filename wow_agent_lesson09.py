import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
# 从环境变量中读取api_key
api_key = os.getenv('QWEN_API_KEY')
ZHIPU_API_KEY = os.getenv('ZISHU_API_KEY')
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
chat_model = "qwen-max"

from typing import List
from zigent.agents import ABCAgent, BaseAgent
from zigent.llm.agent_llms import LLM
from zigent.commons import TaskPackage
from zigent.actions.BaseAction import BaseAction
# from zigent.logging.multi_agent_log import AgentLogger
from duckduckgo_search import DDGS

llm = LLM(api_key=api_key, base_url=base_url, model_name=chat_model)
# response = llm.run("你是谁？")
# print(response)

class DuckSearchAction(BaseAction):
    def __init__(self) -> None:
        action_name = "DuckDuckGo_Search"
        action_desc = "Using this action to search online content."
        params_doc = {"query": "the search string. be simple."}
        self.ddgs = DDGS()
        super().__init__(
            action_name=action_name, 
            action_desc=action_desc, 
            params_doc=params_doc,
        )

    def __call__(self, query):
        results = self.ddgs.chat(query)
        return results
    
search_action = DuckSearchAction()
# results = search_action("什么是 agent")
# print(results)

class DuckSearchAgent(BaseAgent):
    def __init__(
        self,
        llm: LLM,
        actions: List[BaseAction] = [DuckSearchAction()],
        manager: ABCAgent = None,
        **kwargs
    ):
        name = "duck_search_agent"
        role = "You can answer questions by using duck duck go search content."
        super().__init__(
            name=name,
            role=role,
            llm=llm,
            actions=actions,
            manager=manager
        )

def do_search_agent():
    # 创建代理实例
    search_agent = DuckSearchAgent(llm=llm)

    # 创建任务
    task = "what is the found date of microsoft"
    task_pack = TaskPackage(instruction=task)

    # 执行任务并获取响应
    response = search_agent(task_pack)
    print("response:", response)

if __name__ == "__main__":
    do_search_agent()
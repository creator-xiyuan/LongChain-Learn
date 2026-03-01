"""
当出现以下情况时，多智能体系统非常有用：
    单个智能体拥有太多工具，导致其难以决定使用哪个工具
    上下文或记忆对于单个智能体来说变得过于庞大，无法有效跟踪
    任务需要专业化（例如，一个规划者、一个研究员、一个数学专家）

多智能体设计的核心是上下文工程 (context engineering)——决定每个智能体看到哪些信息。LangChain 允许您对以下内容进行精细控制：
    传递给每个智能体的对话或状态的哪些部分
    为子智能体量身定制的专业提示
    包含/排除中间推理
    为每个智能体定制输入/输出格式
"""

# TODO 智能体模式一：工具调用（由一个master agent负责统筹和交互，将其他agent作为tool调用），适合场景：任务编排、结构化工作流程
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

subagent1 = create_agent(model="deepseek:deepseek-chat", tools=[])

@tool(
    "subagent_math",
    description="你是数学领域的专家"
)
def call_subagent_math(query: str):
    print("============调用数学subAgent=================")
    result = subagent1.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content

agent = create_agent(model="deepseek:deepseek-chat", tools=[call_subagent_math])
from langchain_core.messages import HumanMessage

res = agent.invoke(
    {"messages": [HumanMessage("请问 1+1 的结果")]}
)

# TODO 智能体模式二：交接（当前智能体决定将控制权转移给另一个智能体），适应场景：多领域对话、专家接管
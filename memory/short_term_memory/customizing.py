from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool


@tool
def get_user_info(name: str) -> str:
    """一个简单的示例工具：根据用户名返回问候语。"""
    return f"Nice to meet you, {name}!"

# TODO 自定义状态，扩展 AgentState 以添加额外的字段
class CustomAgentState(AgentState):
    user_id: str
    preferences: dict

agent = create_agent(
    "openai:gpt-5",
    [get_user_info],
    state_schema=CustomAgentState,  # TODO 传入自定义对象
    checkpointer=InMemorySaver(),
)

# Custom state can be passed in invoke
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark"}
    },
    {"configurable": {"thread_id": "1"}})
# 工具可以通过 ToolRuntime 参数访问运行时信息，该参数提供：
#   State（状态） - 流经执行的可变数据（消息、计数器、自定义字段）
#   Context（上下文） - 不可变的配置，如用户 ID、会话详细信息或特定于应用程序的配置
#   Store（存储） - 跨对话的持久长期记忆
#   Stream Writer（流写入器） - 在工具执行时流式传输自定义更新
#   Config（配置） - 执行的 RunnableConfig
#   Tool Call ID（工具调用 ID） - 当前工具调用的 ID

from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI


# Access the current conversation state
# TODO runtime参数对模型不可见
@tool
def summarize_conversation(runtime: ToolRuntime) -> str:
    """Summarize the conversation so far."""
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"

from langgraph.types import Command
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

# 通过Tool访问并更新上下文
@tool
def clear_conversation() -> Command:
    """Clear the conversation history."""

    return Command(
        # 对应图状态的state["messages"]
        update={
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }
    )

# Update the user_name in the agent state
@tool
def update_user_name(
    new_name: str,
) -> Command:
    """Update the user's name."""
    # 对应图状态的state["user_name"]
    return Command(update={"user_name": new_name})
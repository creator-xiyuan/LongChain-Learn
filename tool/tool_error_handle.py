from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent


@tool
def search(query: str) -> str:
    """搜索信息。"""
    return f"结果：{query}"

@tool
def get_weather(location: str) -> str:
    """获取位置的天气信息。"""

    if location == "上海":
        raise ValueError("暂时无法获取上海的天气，请稍后再试。")
    return f"{location} 的天气：晴朗，72°F"

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage


@wrap_tool_call
def handle_tool_errors(request, handler):
    """使用自定义消息处理工具执行错误。"""
    try:
        return handler(request)
    except Exception as e:
        # 向模型返回自定义错误消息
        return ToolMessage(
            content=f"工具错误：请检查您的输入并重试。({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="deepseek:deepseek-chat",
    tools=[search, get_weather],
    middleware=[handle_tool_errors])


if __name__ == "__main__":
    res = agent.invoke({"messages": {"role": "user", "content": "查询上海的天气"}})
    print(res)
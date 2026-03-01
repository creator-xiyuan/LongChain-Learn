from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent

@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer

    # Stream custom updates as the tool executes
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")

    return f"It's always sunny in {city}!"

agent = create_agent(
    model="deepseek:deepseek-chat",
    tools=[get_weather]
)
for mode, chunk in agent.stream(
    {
        "messages": [{"role": "user", "content": "查询北京的天气"}]
    },
    stream_mode=["updates", "custom"],
):
    if mode == "custom":
        print("自定义流事件:", chunk)
    else:
        print("状态更新:", chunk)
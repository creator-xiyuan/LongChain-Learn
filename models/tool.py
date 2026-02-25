# 将（可能多个）工具绑定到模型
from langchain.chat_models import init_chat_model
from langchain.tools import tool

model = init_chat_model("deepseek:deepseek-chat")

@tool
def get_weather(location: str) -> str:
    """获取位置的天气信息。"""

    return f"{location} 的天气：晴朗，72°F"

model_with_tools = model.bind_tools([get_weather])

# TODO 强制使用工具
# tool_choice="any" 将强制模型使用所有工具
# model_with_tools = model.bind_tools([get_weather], tool_choice="any")
# tool_choice="get_weather" 将强制模型使用工具get_weather
# model_with_tools = model.bind_tools([get_weather], tool_choice="get_weather")

# 步骤 1：模型生成工具调用
messages = [{"role": "user", "content": "波士顿的天气怎么样？"}]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)
print(ai_msg)

# 步骤 2：执行工具并收集结果
# TODO 模型可能会生成多个工具调用（波士顿和东京的天气怎么样？），此时可以使用 async 并行执行
for tool_call in ai_msg.tool_calls:
    print(tool_call)
    # 每个由工具返回的 ToolMessage 包含一个与原始工具调用匹配的 tool_call_id，帮助模型将结果与请求相关联
    # {'name': 'get_weather', 'args': {'location': '波士顿'}, 'id': 'call_00_h7fSsJNKMEnPv4yPL3FjM5Z6', 'type': 'tool_call'}
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

# 步骤 3：将结果传递回模型以获取最终响应
final_response = model_with_tools.invoke(messages)
print(final_response.text)
# "波士顿当前天气为 72°F，晴朗。"

# TODO 工具的流式调用
for chunk in model_with_tools.stream(
    "波士顿和东京的天气怎么样？"
):
    # 工具调用块逐步到达
    for tool_chunk in chunk.tool_call_chunks:
        if name := tool_chunk.get("name"):
            print(f"工具：{name}")
        if id_ := tool_chunk.get("id"):
            print(f"ID：{id_}")
        if args := tool_chunk.get("args"):
            print(f"参数：{args}")

# 输出：
# 工具：get_weather
# ID：call_SvMlU1TVIZugrFLckFE2ceRE
# 参数：{"lo
# 参数：catio
# 参数：n": "B
# 参数：osto
# 参数：n"}
# 工具：get_weather
# ID：call_QMZdy6qInx13oWKE7KhuhOLR
# 参数：{"lo
# 参数：catio
# 参数：n": "T
# 参数：okyo
# 参数："}
gathered = None
for chunk in model_with_tools.stream("波士顿的天气怎么样？"):
    # gathered + chunk 实际上调用的是 __add__ 方法
    gathered = chunk if gathered is None else gathered + chunk
    print(gathered)
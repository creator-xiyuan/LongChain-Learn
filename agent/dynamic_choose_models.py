from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model


basic_model = init_chat_model("deepseek:deepseek-chat")
advanced_model = init_chat_model("deepseek:deepseek-chat")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """根据对话复杂性选择模型。"""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # 对较长的对话使用高级模型
        model = advanced_model
    else:
        model = basic_model

    request.model = model
    return handler(request)

agent = create_agent(
    model=basic_model,  # 默认模型
    middleware=[dynamic_model_selection]
)

if __name__ == "__main__":
    # 模拟一段短对话（消息数 <= 10，会走 basic_model）
    short_history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，我是智能助手。"},
        {"role": "user", "content": "今天天气怎么样？"},
    ]

    # 模拟一段长对话（消息数 > 10，会走 advanced_model）
    long_history = []
    for i in range(12):
        long_history.append({"role": "user", "content": f"这是第 {i+1} 条用户消息"})
        long_history.append({"role": "assistant", "content": f"这是第 {i+1} 条助手回复"})

    # 只保留最近若干条作为“当前对话消息”，以触发你的判断逻辑
    long_history_for_request = long_history[-12:]

    print("=== 短对话请求 ===")
    res1 = agent.invoke({"messages": short_history})
    print(res1)

    print("\n=== 长对话请求 ===")
    res2 = agent.invoke({"messages": long_history_for_request})
    print(res2)
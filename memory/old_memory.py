"""使用 ConversationChain + ConversationBufferMemory 做多轮对话练手。"""
# ConversationChain 在 v1 中需从 langchain-classic 导入（pip install langchain-classic）
try:
    from langchain.chains import ConversationChain
except ImportError:
    from langchain_classic.chains import ConversationChain

try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    from langchain_classic.memory import ConversationBufferMemory

from langchain.chat_models import init_chat_model

# 初始化模型
llm = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.5,
    max_tokens=1000,
)

# 记忆：保存完整对话历史
memory = ConversationBufferMemory()

# 对话链：自动把 memory 里的 history 和当前 input 拼成 prompt 调用 llm，并把本轮写入 memory
chain = ConversationChain(llm=llm, memory=memory)


def chat(user_input: str) -> str:
    """带记忆的一轮对话。"""
    result = chain.invoke({"input": user_input})
    return result.get("response", result.get("output", str(result)))


if __name__ == "__main__":
    turns = [
        "hi, I'm focused",
        "我刚才说我的名字叫什么？",
        "谢谢，再见",
    ]
    for user_say in turns:
        print(f"User: {user_say}")
        reply = chat(user_say)
        print(f"AI:  {reply}\n")

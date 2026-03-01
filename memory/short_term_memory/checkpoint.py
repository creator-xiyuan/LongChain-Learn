"""
短期记忆（checkpoint）示例

- 短期记忆是 **线程级** 的，最常见的就是会话历史记录；
- Agent 将短期记忆作为状态 (state) 的一部分进行管理；
- 状态通过 checkpointer 持久化到数据库中，以便线程可以随时恢复；
- 当代理被调用或一个步骤（如工具调用）完成时，短期记忆会更新，并在每个步骤开始时根据 thread_id 读取之前的状态。
"""

from langchain.agents import create_agent
from langchain.tools import tool

@tool
def get_user_info(name: str) -> str:
    """一个简单的示例工具：根据用户名返回问候语。"""
    return f"Nice to meet you, {name}!"


DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"

if __name__ == "__main__":
    # 生产环境使用数据库
    # # 使用 PostgresSaver 作为 checkpointer，将对话状态持久化到 PostgreSQL
    # with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    #     # 自动在 Postgres 中创建存储 checkpoint 所需的表
    #     checkpointer.setup()

    # 演示使用内存
    from langgraph.checkpoint.memory import InMemorySaver
    agent = create_agent(
        model="deepseek:deepseek-chat",
        tools=[get_user_info],
        checkpointer=InMemorySaver(),
    )

    # 第一次调用：在线程 1 中告诉智能体我的名字
    first = agent.invoke(
        {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
        # TODO 使用checkpointer时，thread_id是必传的，否则直接报错：ValueError: Checkpointer requires one or more of the following 'configurable' keys: thread_id, checkpoint_ns, checkpoint_id
        {"configurable": {"thread_id": "1"}},
    )
    print("Thread 1 - first turn:", first)

    # 第二次调用：同一个 thread_id=1，模型可以利用已持久化的短期记忆
    second = agent.invoke(
        {"messages": [{"role": "user", "content": "What is my name?"}]},
        {"configurable": {"thread_id": "1"}},
    )
    print("Thread 1 - second turn:", second)

    # 第三次调用：不同的线程 2，没有共享线程 1 的对话历史
    third = agent.invoke(
        {"messages": [{"role": "user", "content": "What is my name?"}]},
        {"configurable": {"thread_id": "2"}},
    )
    print("Thread 2 - first turn:", third)
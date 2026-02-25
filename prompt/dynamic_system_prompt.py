from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest


class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """根据用户角色生成系统提示。"""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "你是一个有帮助的助手。"

    if user_role == "expert":
        return f"{base_prompt} 提供详细的技术响应。"
    elif user_role == "beginner":
        return f"{base_prompt} 简单解释概念，避免使用行话。"

    return base_prompt

agent = create_agent(
    model="deepseek:deepseek-chat",
    middleware=[user_role_prompt],
    context_schema=Context
)


if __name__ == "__main__":
    # 系统提示将根据上下文动态设置
    res1 = agent.invoke({"messages": {"role": "user", "content": "什么是Agent"}},
                       context={"user_role": "expert"})
    print(res1)

    res2 = agent.invoke({"messages": {"role": "user", "content": "什么是Agent"}},
                        context={"user_role": "beginner"})
    print(res2)
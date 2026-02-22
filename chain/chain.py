"""
使用 langchain-classic 的 MultiPromptChain + LLMRouterChain 完成路由任务。
由 LLM 根据问题选择「物理 / 数学 / 默认」之一，再进入对应子链。
"""
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_classic.chains import LLMChain, MultiPromptChain

# 共用同一个模型
llm = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.3,
    max_tokens=500,
)

# prompt_infos：name 会作为路由目标（LLM 输出 destination 必须是这些名字或 "DEFAULT"）
# description 会出现在路由 prompt 里，帮助 LLM 判断
# prompt_template 用 {input} 占位，子链调用时传入 {"input": "用户问题"}
prompt_infos = [
    {
        "name": "physics",
        "description": "适合物理相关的问题，如力学、运动、能量等。",
        "prompt_template": "你是一名物理老师，用简洁、准确的语言回答物理问题。\n\n{input}",
    },
    {
        "name": "math",
        "description": "适合数学问题，如方程、代数、几何等，需要写出关键步骤。",
        "prompt_template": "你是一名数学老师，解答时写出关键步骤。\n\n{input}",
    },
]

# 一键构建：内部会创建 LLMRouterChain + 各 destination 的 LLMChain + 默认 ConversationChain
chain = MultiPromptChain.from_prompts(
    llm=llm,
    prompt_infos=prompt_infos,
    default_chain=None,  # 不传则用 ConversationChain(llm=llm, output_key="text")
)


def run(question: str) -> str:
    """执行路由链。输入为「用户问题」，输出为对应子链的回复文本。"""
    result = chain.invoke({"input": question})
    return result.get("text", str(result))


if __name__ == "__main__":
    questions = [
        "什么是牛顿第一定律？",
        "二次方程 x^2 - 5x + 6 = 0 怎么解？",
        "今天天气怎么样？",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {run(q)}\n")

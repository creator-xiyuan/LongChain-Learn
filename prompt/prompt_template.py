from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent

SYSTEM_PROMPT = """You are a boy"""

template = ChatPromptTemplate(
    [
        ("system", "{system}"),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ]
)

prompt_value = template.invoke(
    {
        "system": SYSTEM_PROMPT,
        "user_input": "What is your name?",
    }
)

from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)

agent = create_agent(
    model=model,
)

response = agent.invoke({"messages": prompt_value.messages})

print(response)

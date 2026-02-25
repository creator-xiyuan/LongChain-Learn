from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent1 = create_agent(
    model="deepseek:deepseek-chat",
    response_format=ToolStrategy(ContactInfo)
    # ToolStrategy uses artificial tool calling to generate structured output. This works with any model that supports tool calling.
    # TODO ToolStrategy should be used when provider-native structured output (via ProviderStrategy) is not available or reliable.
)

from langchain.agents.structured_output import ProviderStrategy

agent2 = create_agent(
    model="openai:gpt-4o",
    response_format=ProviderStrategy(ContactInfo)
)

result1 = agent1.invoke({
    "messages": [{"role": "user", "content": "从以下内容提取联系信息：John Doe, john@example.com, (555) 123-4567"}]
})

print(result1)
# 约定的固定字段
result1["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
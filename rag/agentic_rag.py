"""
智能体（由 LLM 驱动）不是在回答之前检索文档，而是逐步推理，并决定在交互过程中何时以及如何检索信息
"""
import requests
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def fetch_url(url: str) -> str:
    """Fetch text content from a URL"""
    response = requests.get(url, timeout=10.0)
    response.raise_for_status()
    return response.text

system_prompt = """\
Use fetch_url when you need to fetch information from a web-page; quote relevant snippets.
"""

agent = create_agent(
    model="deepseek:deepseek-chat",
    tools=[fetch_url], # A tool for retrieval
    system_prompt=system_prompt,
)
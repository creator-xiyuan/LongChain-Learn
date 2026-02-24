# 吴恩达 LangChain for LLM Application Development - RAG 查询与回答
# 流程：加载 CSV -> 向量化存储 -> 检索相似文档 -> 结合 LLM 生成回答

import os
# LangChain v1 标准导入
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 数据文件路径（可与脚本同目录或指定绝对路径）
DATA_FILE = os.path.join(os.path.dirname(__file__), "OutdoorClothingCatalog_1000.csv")
if not os.path.isfile(DATA_FILE):
    DATA_FILE = "OutdoorClothingCatalog_1000.csv"

# 加载文档
loader = CSVLoader(file_path=DATA_FILE)
documents = loader.load()

OLLAMA_EMBEDDING_MODEL = "modelscope.cn/OllmOne/bge-m3-GGUF:latest"
OLLAMA_EMBEDDING_BASE_URL = "http://127.0.0.1:11434"

embeddings = OllamaEmbeddings(
    model=OLLAMA_EMBEDDING_MODEL,
    base_url=OLLAMA_EMBEDDING_BASE_URL,
    sync_client_kwargs={"trust_env": False}, # 关掉代理
)
db = DocArrayInMemorySearch.from_documents(documents, embeddings)

def get_retriever(k: int = 4):
    return db.as_retriever(search_type="similarity", search_kwargs={"k": k})

# RAG 提示：用检索到的上下文回答用户问题
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个商品推荐助手。请仅根据下面「相关商品信息」回答问题，不要编造内容。若信息不足，请说明。"),
    ("human", "相关商品信息：\n\n{context}\n\n用户问题：{question}\n\n请给出简洁、有用的回答。"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(llm=None):
    if llm is None:
        llm = init_chat_model(model="deepseek:deepseek-chat", temperature=0.3, max_tokens=500)
    retriever = get_retriever(k=3)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain

def query_and_answer(question: str, llm=None) -> str:
    chain = build_rag_chain(llm=llm)
    return chain.invoke(question)

def similarity_search_only(query: str, k: int):
    return db.similarity_search(query, k=k)


if __name__ == "__main__":
    query = "Please suggest a shirt with sunblocking"
    docs = similarity_search_only(query, k=3)
    print("=== 相似度检索结果（前 3 条）===")
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] {doc.page_content[:200]}...")
    print()

    # 完整 RAG 回答
    print("=== RAG 回答 ===")
    answer = query_and_answer(query)
    print(f"Q: {query}\nA: {answer}")
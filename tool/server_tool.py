from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4.1-mini")

tool = {"type": "web_search"}
model_with_tools = model.bind_tools([tool])

# TODO 服务端工具调用，工具逻辑由模型服务提供方托管并运行（比如某些内置搜索、代码执行沙箱等）
# 某些提供商支持服务器端工具调用
response = model_with_tools.invoke("今天有什么正面新闻？")
response.content_blocks
# [
#     {
#         "type": "server_tool_call",
#         "name": "web_search",
#         "args": {
#             "query": "positive news stories today",
#             "type": "search"
#         },
#         "id": "ws_abc123"
#     },
#     {
#         "type": "server_tool_result",
#         "tool_call_id": "ws_abc123",
#         "status": "success"
#     },
#     {
#         "type": "text",
#         "text": "以下是今天的一些正面新闻...",
#         "annotations": [
#             {
#                 "end_index": 410,
#                 "start_index": 337,
#                 "title": "文章标题",
#                 "type": "citation",
#                 "url": "..."
#             }
#         ]
#     }
# ]
from langchain.chat_models import init_chat_model

model = init_chat_model("deepseek:deepseek-chat")
for chunk in model.stream("Why do parrots have colorful feathers?"):
    print(chunk.text, end="|", flush=True)
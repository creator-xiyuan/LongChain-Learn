import asyncio
from time import time
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.2,
    check_every_n_seconds=0.1,
    max_bucket_size=2,
)

model = init_chat_model(
    model="deepseek:deepseek-chat",
    rate_limiter=rate_limiter,
)

async def call_once(i: int):
    start = time()
    res = await model.ainvoke(f"hello {i}")
    print(i, "start:", start, "end: ", time(),  "耗时：", time() - start)
    return res

async def main():
    tasks = [call_once(i) for i in range(5)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
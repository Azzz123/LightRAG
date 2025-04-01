import asyncio
import os
import inspect
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "../outputs/aljazeera_r_u_news"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="qwen2.5:32b",
    llm_model_max_async=2,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=512,
        func=lambda texts: ollama_embedding(
            texts, embed_model="quentinz/bge-large-zh-v1.5", host="http://localhost:11434"
        ),
    ),
    addon_params={"language": "Simplified Chinese",
                  "entity_types": ["主体", "实验设备", "时间轴-时间日期", "位置", "演习内容", "部署的军事力量",
                                   "支持物质", "事故结果", "展示设备", "伤害数量", "外交内容", "舆论内容"],
                  "insert_batch_size": 16}
)

with open("../inputs/aljazeera_r_u_news_processed_with_time.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# # Perform naive search
# print(
#     rag.query("俄乌冲突对世界各方势力带来的影响是什么？", param=QueryParam(mode="naive"))
# )
#
# # Perform local search
# print(
#     rag.query("俄乌冲突对世界各方势力带来的影响是什么？", param=QueryParam(mode="local"))
# )
#
# # Perform global search
# print(
#     rag.query("俄乌冲突对世界各方势力带来的影响是什么？", param=QueryParam(mode="global"))
# )
#
# # Perform hybrid search
# print(
#     rag.query("俄乌冲突对世界各方势力带来的影响是什么？", param=QueryParam(mode="hybrid"))
# )
rag.query("俄乌冲突对世界各方势力带来的影响是什么？", param=QueryParam(mode="naive"))
rag.query("俄乌冲突对世界各方势力带来的影响是什么？", param=QueryParam(mode="local"))
rag.query("俄乌冲突对世界各方势力带来的影响是什么？", param=QueryParam(mode="global"))
rag.query("俄乌冲突对世界各方势力带来的影响是什么？", param=QueryParam(mode="hybrid"))

# stream response
resp = rag.query(
    "俄乌冲突对世界各方势力带来的影响是什么？",
    param=QueryParam(mode="hybrid", stream=True),
)


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


if inspect.isasyncgen(resp):
    asyncio.run(print_stream(resp))
else:
    print(resp)

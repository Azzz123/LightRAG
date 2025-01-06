import asyncio
import os
import inspect
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "./outputs/sputniknews_r_u_news"

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
    addon_params={"example_number": 1,
                  "language": "Simplified Chinese",
                  "entity_types": ["实验_主体", "实验_设备", "实验_日期", "实验_位置", "演习_主体", "演习_内容",
                                   "演习_日期", "演习_区域", "部署_主体", "部署_军事力量", "部署_日期", "部署_位置",
                                   "支持_主体", "支持_对象", "支持_材料", "支持_日期", "事故_主体", "事故_结果",
                                   "事故_日期", "事故_位置", "展示_主体", "展示_设备", "展示_日期", "展示_位置",
                                   "冲突_主体", "冲突_对象", "冲突_日期", "冲突_位置", "伤害_主体", "伤害_数量",
                                   "伤害_日期", "伤害_位置", "外交_主体", "外交_对象", "外交_内容", "外交_日期",
                                   "舆论_主体", "舆论_对象", "舆论_内容", "舆论_日期", "时间轴_时间"],
                  "insert_batch_size": 16}
)

with open("./inputs/sputniknews_r_u_news_processed_with_time.txt", "r", encoding="utf-8") as f:
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

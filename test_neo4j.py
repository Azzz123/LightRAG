import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

WORKING_DIR = "./outputs/sputniknews_r_u_news"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="qwen2.5:32b",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=512,
        func=lambda texts: ollama_embedding(
            texts, embed_model="quentinz/bge-large-zh-v1.5", host="http://localhost:11434"
        ),
    ),
    graph_storage="Neo4JStorage",
    log_level="DEBUG"
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

#
# with open("./inputs/aljazeera_r_u_news_processed.txt") as f:
#     rag.insert(f.read())

# Perform naive search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
# )
#
# # Perform local search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))
# )
#
# # Perform global search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
# )
#
# # Perform hybrid search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
# )

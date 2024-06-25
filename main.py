import asyncio
import time

from dotenv import dotenv_values
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.legacy.llms import OpenAILike as OpenAI
from qdrant_client import models
from tqdm.asyncio import tqdm

from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.qa import read_jsonl, save_answers
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval
import os


def list_directories(path):
    # 获取指定路径下的所有文件和文件夹
    items = os.listdir(path)
    # 过滤出文件夹
    directories = [item for item in items if os.path.isdir(os.path.join(path, item))]
    return directories


async def main():
    config = dotenv_values(".env")

    path = './data'  # 示例路径（请替换为你的目录路径）
    folders = list_directories(path)  # 获取文件夹名称列表
    print(folders)

    # 初始化 LLM 嵌入模型 和 Reranker
    llm = OpenAI(
        api_key=config["GLM_KEY"],
        model="glm-4",
        api_base="https://open.bigmodel.cn/api/paas/v4/",
        is_chat_model=True,
    )
    embeding = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-zh-v1.5",
        cache_folder="./",
        embed_batch_size=128,
    )
    Settings.embed_model = embeding

    # 初始化 数据ingestion pipeline 和 vector store
    client, vector_stores = await build_vector_store(config, folders, reindex=False)

    # collection_info = await client.get_collection(
    #     config["COLLECTION_NAME"] or "aiops24"
    # )
    #
    # if collection_info.points_count == 0:
    #     data = read_data("data")
    #     pipeline = build_pipeline(llm, embeding, vector_store=vector_store)
    #     # 暂时停止实时索引
    #     await client.update_collection(
    #         collection_name=config["COLLECTION_NAME"] or "aiops24",
    #         optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
    #     )
    #     await pipeline.arun(documents=data, show_progress=True, num_workers=1)
    #     # 恢复实时索引
    #     await client.update_collection(
    #         collection_name=config["COLLECTION_NAME"] or "aiops24",
    #         optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
    #     )
    #     print(len(data))

    for folder in folders:
        collection_info = await client.get_collection(folder)
        if collection_info.points_count == 0:
            data = read_data("data/" + folder)
            pipeline = build_pipeline(llm, embeding, vector_store=vector_stores[folder])
            # 暂时停止实时索引
            await client.update_collection(
                collection_name=folder,
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
            )
            await pipeline.arun(documents=data, show_progress=True, num_workers=1)
            # 恢复实时索引
            await client.update_collection(

                collection_name=folder,
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
            )
            print(folder + "：" + str(len(data)))

    # retriever = QdrantRetriever(vector_store, embeding, similarity_top_k=config["VECTOR_TOP_K"])

    queries = read_jsonl("question.jsonl")

    # 生成答案
    print("Start generating answers...")

    results = []
    for query in tqdm(queries, total=len(queries)):
        result = await generation_with_knowledge_retrieval(query_str=query["query"], llm=llm,
                                                           document=query["document"], config=config,
                                                           vector_stores=vector_stores,
                                                           embeding=embeding,
                                                           )
        results.append(result)

    # 处理结果
    save_answers(queries, results, "submit_result.jsonl")


if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    execution_time = end_time - start_time
    print("程序运行时间为：", execution_time, "秒")

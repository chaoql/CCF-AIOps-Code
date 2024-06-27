import asyncio
import time
from dotenv import load_dotenv, find_dotenv
from dotenv import dotenv_values
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.legacy.llms import OpenAILike as OpenAI
from qdrant_client import models
from tqdm.asyncio import tqdm
from llama_index.embeddings.openai import OpenAIEmbedding
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
    _ = load_dotenv(find_dotenv())  # 导入环境
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
    embeding = OpenAIEmbedding(model="text-embedding-3-large")
    # embeding = HuggingFaceEmbedding(
    #     model_name="BAAI/bge-large-zh-v1.5",
    #     cache_folder="./BAAI/",
    #     embed_batch_size=128,
    # )
    Settings.embed_model = embeding

    # 初始化 数据ingestion pipeline 和 vector store
    client, vector_stores = await build_vector_store(config, folders, reindex=False)

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

    queries = read_jsonl("questionPlus.jsonl")

    # 生成答案
    print("Start generating answers...")

    results = []
    for query in tqdm(queries, total=len(queries)):
        # Multi Query: Different Perspectives
        result = await generation_with_knowledge_retrieval(query_str=query["query"], llm=llm,
                                                           document=query["document"], config=config,
                                                           abbreviation=query["abbreviation"],
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
    print("程序运行时间为：", execution_time / 60.0, "分钟")

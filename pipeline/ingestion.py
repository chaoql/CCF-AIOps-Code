from llama_index.core import SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.extractors import SummaryExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms.llm import LLM
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, MetadataMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from llama_index.llms.openai import OpenAI
from custom.template import SUMMARY_EXTRACT_TEMPLATE
from custom.transformation import CustomFilePathExtractor, CustomTitleExtractor


def read_data(path: str = "data") -> list[Document]:
    reader = SimpleDirectoryReader(
        input_dir=path,
        recursive=True,
        required_exts=[
            ".txt",
        ],
    )
    return reader.load_data()


def build_pipeline(
        llm: LLM,
        embed_model: BaseEmbedding,
        template: str = None,
        vector_store: BasePydanticVectorStore = None,
) -> IngestionPipeline:
    transformation = [
        # SentenceSplitter(chunk_size=2048, chunk_overlap=256),
        SentenceSplitter(chunk_size=1024, chunk_overlap=50),
        CustomTitleExtractor(metadata_mode=MetadataMode.EMBED),
        CustomFilePathExtractor(last_path_length=4, metadata_mode=MetadataMode.EMBED),
        SummaryExtractor(
            llm = OpenAI(model="gpt-3.5-turbo"),
            metadata_mode=MetadataMode.EMBED,
            prompt_template=template or SUMMARY_EXTRACT_TEMPLATE,
        ),
        embed_model,
    ]

    return IngestionPipeline(transformations=transformation, vector_store=vector_store)


async def build_vector_store(
        config: dict, folders, reindex: bool = False
):
    client = AsyncQdrantClient(  # Qdrant向量数据库
        # url=config["QDRANT_URL"],
        # location=":memory:",
        path="D:\\MyPyCharm\\LLMTuning\\aiops24-RAG-demo-glm\\demo\\VecData-text-embedding-3-large",
    )
    VSList = {}
    if reindex:  # 重新索引
        try:
            for forder in folders:
                await client.delete_collection(forder)
            # await client.delete_collection(config["COLLECTION_NAME"] or "aiops24")
        except UnexpectedResponse as e:
            print(f"Collection not found: {e}")
    try:
        for forder in folders:
            await client.create_collection(  # 生成collection
                collection_name=forder,
                vectors_config=models.VectorParams(
                    size=int(config["VECTOR_SIZE"]), distance=models.Distance.DOT
                ),
            )
            VSList[forder] = QdrantVectorStore(
                aclient=client,
                collection_name=forder,
                parallel=4,
                batch_size=32,
            )
        # await client.create_collection(  # 生成collection
        #     collection_name=config["COLLECTION_NAME"] or "aiops24",
        #     vectors_config=models.VectorParams(
        #         size=config["VECTOR_SIZE"] or 1024, distance=models.Distance.DOT
        #     ),
        # )
    except Exception as e:
        for forder in folders:
            VSList[forder] = QdrantVectorStore(
                aclient=client,
                collection_name=forder,
                parallel=4,
                batch_size=32,
            )
        print("Collection already exists")
    return client, VSList

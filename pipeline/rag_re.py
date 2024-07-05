from typing import List
import qdrant_client
from llama_index.legacy.llms import OpenAILike as OpenAI
from dotenv import dotenv_values
from llama_index.core.llms.llm import LLM
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import (
    QueryBundle,
    PromptTemplate,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from .imageSumm import image_summary


# from custom.template import QA_TEMPLATE
def getFullName(log_file_path):
    global lines
    # 使用with语句安全地打开文件
    with open(log_file_path, 'r', encoding='utf-8') as file:
        # 读取所有行到一个列表中
        lines = file.readlines()

    # 创建一个新列表，用于存储没有空行的行
    non_blank_lines = [line.strip() for line in lines if line.strip()]

    logWords = {}
    suolue = ""
    suols = set()
    wanzs = set()
    for line in non_blank_lines:
        if len(line) < 10:
            if len(wanzs):
                wanzslist = list(wanzs)
                logWords[suolue] = wanzslist
                wanzs.clear()
            if line in suols:
                continue
            suols.add(line)
            suolue = line
        else:
            wanzs.add(line)
    return logWords


class QdrantRetriever(BaseRetriever):
    def __init__(
            self,
            vector_store: QdrantVectorStore,
            embed_model: BaseEmbedding,
            similarity_top_k: int = 2,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k
        super().__init__()

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding, similarity_top_k=self._similarity_top_k
        )
        query_result = await self._vector_store.aquery(vector_store_query)

        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding, similarity_top_k=self._similarity_top_k
        )
        query_result = self._vector_store.query(vector_store_query)

        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores


async def generation_with_knowledge_retrieval(
        query_str: str,
        # retriever: BaseRetriever,
        llm: LLM,
        document: str,
        config,
        abbreviation: str,
        embeding,
        vector_stores,
        # qa_template: str = QA_TEMPLATE,
        # reranker: BaseNodePostprocessor,
        debug: bool = False,
        progress=None,
) -> CompletionResponse:
    # 12. 全称+简写两个问题，每个问题5选1，且去重
    log_file_path = 'D:\\MyPyCharm\\LLMTuning\\aiops24-RAG-demo-glm\\demo\\logName.txt'
    Fullnames = getFullName(log_file_path)
    # 单轮10选3rerank，cosine
    retriever = QdrantRetriever(vector_stores[document], embeding, similarity_top_k=int(config["VECTOR_TOP_K"]))
    template = """你的任务是对检索到的上下文进行判断，判断其是否有助于回答当前问题。
问题：{question}
上下文: {context}
如果检索到的上下文有助于回答问题，请回答 "是"，否则请回答 "否"，并阐述理由。"""
    query_strs = set()
    query_strs.add(query_str)
    abs = abbreviation.split(",")  # 当前问题包含的所有缩略词
    context_strs = ""
    mutiFullName = ""
    top_n = 2
    if abs:  # 有缩写
        query_str2 = query_str
        for ab in abs:  # 抽取英文缩写
            if ab in Fullnames.keys():
                if len(Fullnames[ab]) > 1:
                    mutiFullName = ab
                    continue
                else:
                    print(ab)
                    print(Fullnames[ab][0])
                    query_str2 = query_str2.replace(ab, Fullnames[ab][0])
        query_strs.add(query_str2)
        if mutiFullName:
            for i in Fullnames[mutiFullName]:
                query_str2 = query_str2.replace(mutiFullName, i)
                query_strs.add(query_str2)
        # query_strs.append(query_str2)
    query_strs = list(query_strs)
    print(query_strs)
    # print("----------------")
    tempList = set()
    file_paths = []
    for i in range(3):
        for query_i in query_strs:  # 遍历每种提问
            # query_plus = query_str
            query_bundle = QueryBundle(query_str=query_i)
            node_with_scores = await retriever.aretrieve(query_bundle)  # 返回包含得分信息的节点列表

            reranker = FlagEmbeddingReranker(
                top_n=5,
                model="BAAI/bge-reranker-large",
                use_fp16=True,
            )
            node_with_scores = reranker._postprocess_nodes(node_with_scores, query_bundle)

            context_str = ""  # 当前问题检索得到的上下文信息
            k = 0
            for index, node in enumerate(node_with_scores):  # 遍历当前提问的检索上下文
                if node.text in tempList:  # 搜索出重复的就向后取
                    continue
                if k == top_n:  # 只取top_n个
                    break

                prompt = ChatPromptTemplate.from_template(template).format(question=query_str,
                                                                           context=context_strs + node.text)
                retTemp = await llm.acomplete(prompt)
                print(retTemp.text)
                if retTemp.text[0] == "是":  # 检索到的上下文对回答问题有帮助
                    tempList.add(node.text)
                    str = f"\n{node.metadata['document_title']}: {node.text}"
                    context_str += str
                    k += 1
                    file_paths.append(node.metadata['file_path'])
                    print(node.metadata['file_path'])
            context_strs += context_str  # 所有问题检索得到的上下文信息

    template = """你是问题解答任务的助手。使用以下检索到的上下文回答问题，答案最多不超过三句话，简明扼要。

问题： {question} 

上下文： {context} 

答案："""
    fmt_qa_prompt = ChatPromptTemplate.from_template(template).format(context=context_strs, question=query_str)
    ret = await llm.acomplete(fmt_qa_prompt)
    if progress:
        progress.update(1)
    return [ret, context_strs]

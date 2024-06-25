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
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
from langchain import hub


# from custom.template import QA_TEMPLATE


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
        origin_query: str,
        query_strs,
        # retriever: BaseRetriever,
        llm: LLM,
        document: str,
        config,
        embeding,
        vector_stores,
        # qa_template: str = QA_TEMPLATE,
        reranker: BaseNodePostprocessor | None = None,
        debug: bool = False,
        progress=None,
) -> CompletionResponse:
    global query_bundle
    retriever = QdrantRetriever(vector_stores[document], embeding, similarity_top_k=int(config["VECTOR_TOP_K"]))
    context_strs = []
    all_node_with_scores = []
    for query_str in query_strs:
        query_bundle = QueryBundle(query_str=query_str)
        node_with_scores = await retriever.aretrieve(query_bundle)  # 返回包含得分信息的节点列表
        all_node_with_scores += node_with_scores
    if debug:
        print(f"retrieved:\n{all_node_with_scores}\n------")
    if reranker:
        config = dotenv_values(".env")
        reranker = RankGPTRerank(
            llm=OpenAI(
                api_key=config["GLM_KEY"],
                model="glm-4",
                api_base="https://open.bigmodel.cn/api/paas/v4/",
                is_chat_model=True,
            ),
            top_n=1,
            verbose=True,
        )
        node_with_scores = reranker.postprocess_nodes(all_node_with_scores, query_bundle)
        if debug:
            print(f"reranked:\n{node_with_scores}\n------")
    context_str = "\n\n".join(  # 抽取查询到的结果文本
        [f"{node.metadata['document_title']}: {node.text}" for node in all_node_with_scores]
    )
    context_strs.append(context_str)
    # set the LANGCHAIN_API_KEY environment variable (create key in settings)
    context_str = ["\n" + str for str in context_strs]

    fmt_qa_prompt = hub.pull("rlm/rag-prompt").format(
        context=context_str, question=origin_query
    )

    ret = await llm.acomplete(fmt_qa_prompt)
    if progress:
        progress.update(1)
    return ret

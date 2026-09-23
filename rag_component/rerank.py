import asyncio
import os
import logging
from typing import Any, Optional, Sequence

import httpx
from dotenv import load_dotenv
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import Field

load_dotenv(override=True)
logger = logging.getLogger(__name__)


class SiliconFlowRerank(BaseDocumentCompressor):
    """SiliconFlow 原生 /v1/rerank 重排器（其协议与 Cohere 不兼容，故自行封装）。"""

    # 以下字段通过 .env / 环境变量读取，也可在构造时显式传入
    base_url: str = Field(default_factory=lambda: os.getenv("SiliconFlow_BASE_URL", ""))
    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("SiliconFlow_API_KEY")
    )
    model: Optional[str] = Field(
        default_factory=lambda: os.getenv("SiliconFlow_Reranker_MODEL")
    )
    top_n: int = Field(
        default_factory=lambda: int(os.getenv("TOP_N_RERANK", 5))
    )  # 返回前 top_n 个（0 或负数=全部）
    score_threshold: float = Field(
        default_factory=lambda: float(os.getenv("SCORE_THRESHOLD_RERANK", "0.1"))
    )  # 低于该分值的文档被过滤（0 或负数=不过滤）
    timeout: float = 60.0  # 单次请求超时（秒）

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """同步入口占位（基类抽象方法必须实现，否则无法实例化）；本重排器只提供异步版。"""
        raise NotImplementedError(
            "SiliconFlowRerank 只提供异步重排，请改用 "
            "await compressor.acompress_documents(documents=..., query=...)"
        )

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """异步版：按与 query 的相关性重排，返回前 top_n 个中分值达标（≥ score_threshold）的文档。"""
        if not documents:
            return []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url.rstrip('/')}/rerank",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "query": query,
                    "documents": [doc.page_content for doc in documents],
                    "top_n": self.top_n,
                },
            )
        response.raise_for_status()
        results: Any = response.json()["results"]  # [{index, relevance_score}, ...]

        # 按返回顺序构造新 Document，打分写进 metadata，并过滤低相关文档
        reranked_docs = []
        for item in results:
            score = item["relevance_score"]
            if score < self.score_threshold:
                continue  # 相关度太低，直接丢弃
            origin_doc = documents[item["index"]]
            new_doc = Document(
                page_content=origin_doc.page_content,
                metadata=dict(origin_doc.metadata),  # 复制原 metadata，避免污染原对象
            )
            new_doc.metadata["relevance_score"] = score
            reranked_docs.append(new_doc)
        return reranked_docs


compressor = SiliconFlowRerank()


def _print_rerank_chunks(rerank_chunks):
    """打印 Rerank 结果（显式接收参数，不再依赖模块级全局变量）"""
    logger.debug("================== Rerank结果 ==================")
    for i, chunk in enumerate(rerank_chunks, start=1):
        relevance_score = chunk.metadata["relevance_score"]
        text = chunk.page_content

        logger.debug(f"Chunk[{i}] | metadata={chunk.metadata} \n{text}\n")

    logger.debug("================== Rerank结果 ==================")


async def aget_rerank_chunks(
    retrieve_chunks: list[dict], query: str
) -> Sequence[Document]:
    """把 Milvus 的检索结果转成 Document 后做重排"""
    retrieve_docs = [
        Document(
            page_content=chunk["entity"]["text"], metadata=chunk["entity"]["metadata"]
        )
        for chunk in retrieve_chunks
    ]
    rerank_chunks = await compressor.acompress_documents(
        documents=retrieve_docs, query=query
    )
    _print_rerank_chunks(rerank_chunks)
    return rerank_chunks


async def main():
    """异步冒烟测试：直接验证 acompress_documents"""
    docs = [
        Document(page_content="水果", metadata={"i": 0}),
        Document(page_content="桌子", metadata={"i": 1}),
        Document(page_content="时间", metadata={"i": 2}),
        Document(page_content="动物", metadata={"i": 3}),
        Document(page_content="我们", metadata={"i": 4}),
    ]
    query = "西瓜"
    print(f"Query: {query}")
    for d in await compressor.acompress_documents(documents=docs, query=query):
        print(f"{d.page_content} | relevance_score={d.metadata['relevance_score']:.5f}")


if __name__ == "__main__":
    asyncio.run(main())

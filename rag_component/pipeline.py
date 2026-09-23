"""pipeline.py —— RAG 入库流水线与检索入口。

职责：把"已落盘的文档"变成"可检索的向量"：解析 → 切分 → 向量化 → 写入 Milvus。
对外：ingest_file()（入库全流程）、upload_new_file()（脚本用）、retrieve_from_vector_db()（混合检索）。
"""

import asyncio
import inspect
import logging
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from constants import STEP_PERCENT  # 跨层共享的进度常量
from db.milvus.client import create_rag_db, upsertToMilvus, retrieve
from rag_component.embedding import embedding_model
from rag_component.loader import aload_document
from rag_component.splitter import split_docs

logger = logging.getLogger(__name__)

# 进度回调：同步、异步都支持，事件形如 {"step", "status", "message", "percent"}
ProgressCallback = Callable[[dict], Awaitable[None] | None]

# 向量化的分批大小（避免被嵌入服务按条数/长度拒掉）
EMBED_BATCH_SIZE = 32


def _print_retrieve_chunks(retrieve_chunks: list[dict]):
    logger.debug("================== 检索结果 ==================")
    for i, chunk in enumerate(retrieve_chunks, start=1):
        id = chunk["entity"]["id"]
        score = chunk["distance"]
        text = chunk["entity"]["text"]
        source = chunk["entity"].get("source", "unknown")

        logger.debug(
            f"Trunk[{i}] | id={id} | score={score:.5f} | source={source} \n{text}\n"
        )

    logger.debug("================== 检索结果 ==================")


async def _emit(
    on_progress: ProgressCallback | None,
    step: str,
    status: str,
    message: str,
    *,
    percent: int | None = None,
) -> None:
    """把一条进度事件交给回调（同步/异步回调皆可，协程会在这里被 await）。"""
    if on_progress is None:
        return
    event: dict = {"step": step, "status": status, "message": message}
    if percent is not None:
        event["percent"] = percent
    result = on_progress(event)
    if inspect.isawaitable(result):
        await result


async def ingest_file(
    file_path: str,
    *,
    doc_id: str,
    source: str | None = None,
    file_name: str | None = None,
    category: str = "",
    on_progress: ProgressCallback | None = None,
    embed_batch_size: int = EMBED_BATCH_SIZE,
) -> int:
    """把已落盘的文档：解析 → 切分 → 向量化 → 写入 Milvus，返回 chunk 数。

    doc_id 写进 chunk 主键 f"{doc_id}_{序号}"，保证同一文档多次上传不撞主键。
    只管"内容入库"（删旧 chunk、临时文件改名由调用方负责），失败直接抛异常。
    """
    source = source or Path(file_path).as_posix()
    file_name = file_name or Path(file_path).name

    # 1) 解析（PDF/Office 走 MinerU 在线解析，全流程最慢的一步）
    await _emit(on_progress, "parse", "running", f"正在解析文档：{file_name}")
    documents = await aload_document(file_path)
    await _emit(
        on_progress,
        "parse",
        "done",
        f"解析完成，共得到 {len(documents)} 个文档块",
        percent=STEP_PERCENT["parse"],
    )

    # 2) 切分（纯 CPU，同步调用）
    chunks = split_docs(documents)
    logger.info(f"{source} 共切分为 {len(chunks)} 个 chunk")
    await _emit(
        on_progress,
        "split",
        "done",
        f"切分完成，共 {len(chunks)} 个 chunk",
        percent=STEP_PERCENT["split"],
    )

    texts = [chunk.page_content for chunk in chunks]
    if not texts:
        # 空文件 / 解析失败：不要让 0 个 chunk 的文档进库
        raise ValueError("文档解析后没有可入库的内容（可能是空文件或解析失败）")

    # 3) 向量化（分批发送）
    vectors: list[list[float]] = []
    for start in range(0, len(texts), embed_batch_size):
        batch = texts[start : start + embed_batch_size]
        vectors.extend(await embedding_model.aembed_documents(batch))
        finished = len(vectors) >= len(texts)
        await _emit(
            on_progress,
            "embed",
            "done" if finished else "running",
            f"已向量化 {len(vectors)}/{len(texts)} 个 chunk",
            percent=STEP_PERCENT["embed"] if finished else None,
        )

    # 4) 组装写入数据（metadata 补文档级信息，检索命中时能回溯来源）
    # 注意：sparse_vector 不用传——BM25 Function 输出字段，服务端按 text 自动重算
    data = [
        {
            "id": f"{doc_id}_{i}",  # 主键 = 文档编号 + chunk 序号
            "dense_vector": vectors[i],
            "text": texts[i],
            "source": source,
            "metadata": {
                **chunks[i].metadata,
                # 覆盖 loader 记录的 source（上传链路里是临时文件名），统一成落盘相对路径
                "source": source,
                "doc_id": doc_id,
                "file_name": file_name,
                "category": category,
                "chunk_index": i,
            },
        }
        for i in range(len(texts))
    ]

    # 5) 写入 Milvus
    await upsertToMilvus(data)
    await _emit(
        on_progress,
        "write",
        "done",
        f"已写入向量库，共 {len(data)} 个 chunk",
        percent=STEP_PERCENT["write"],
    )

    return len(chunks)


async def upload_new_file(file_path: str) -> int:
    """单文件入库的最简用法（脚本 / 冒烟测试）；正式链路走 services.document_service。"""
    return await ingest_file(file_path, doc_id=uuid.uuid4().hex)


async def retrieve_from_vector_db(query: str, limit: int = 10) -> list[dict]:
    """混合检索：稠密向量（COSINE）+ BM25 稀疏（查询文本交服务端分词），RRF 融合返回 top-limit。"""
    query_vector = await embedding_model.aembed_query(query)
    retrieve_chunks = await retrieve(
        query_vector=query_vector, query_text=query, limit=limit
    )
    _print_retrieve_chunks(retrieve_chunks)
    return retrieve_chunks


async def main():
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s:%(lineno)d:%(message)s"
    )
    await create_rag_db()
    await upload_new_file("assets/demo_txt_1.txt")


if __name__ == "__main__":
    asyncio.run(main())

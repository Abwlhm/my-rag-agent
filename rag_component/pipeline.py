import asyncio
import inspect
import logging
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from constants import STEP_PERCENT
from db.milvus.client import create_rag_db, upsertToMilvus, retrieve
from rag_component.embedding import embedding_model
from rag_component.loader import aload_document
from rag_component.splitter import split_docs

logger = logging.getLogger(__name__)


ProgressCallback = Callable[[dict], Awaitable[None] | None]

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
    source = source or Path(file_path).as_posix()
    file_name = file_name or Path(file_path).name

    await _emit(on_progress, "parse", "running", f"正在解析文档：{file_name}")
    documents = await aload_document(file_path)
    await _emit(
        on_progress,
        "parse",
        "done",
        f"解析完成，共得到 {len(documents)} 个文档块",
        percent=STEP_PERCENT["parse"],
    )

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
        raise ValueError("文档解析后没有可入库的内容（可能是空文件或解析失败）")

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

    data = [
        {
            "id": f"{doc_id}_{i}",
            "vector": vectors[i],
            "text": texts[i],
            "source": source,
            "metadata": {
                **chunks[i].metadata,
                "source": source,
                "doc_id": doc_id,
                "file_name": file_name,
                "category": category,
                "chunk_index": i,
            },
        }
        for i in range(len(texts))
    ]

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
    return await ingest_file(file_path, doc_id=uuid.uuid4().hex)


async def retrieve_from_vector_db(query: str, limit: int = 10) -> list[dict]:
    query_vector = await embedding_model.aembed_query(query)
    retrieve_chunks = await retrieve(query_vector, limit=limit)
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

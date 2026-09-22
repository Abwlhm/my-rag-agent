import asyncio
import logging
import os

from dotenv import load_dotenv
from pymilvus import AsyncMilvusClient

load_dotenv(override=True)

MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
db_name = os.getenv("MILVUS_DB_NAME", "rag_demo")
collection_name = os.getenv("MILVUS_COLLECTION_NAME", "docs")
Embedding_DIM = int(os.getenv("MILVUS_EMBEDDING_DIM", "4096"))

logger = logging.getLogger(__name__)

_client = AsyncMilvusClient(uri=MILVUS_URI, db_name=db_name)


def get_client() -> AsyncMilvusClient:
    return _client


_ready = False
_ready_lock = asyncio.Lock()


async def create_rag_db():
    global _client, _ready
    _client = AsyncMilvusClient(uri=MILVUS_URI)
    milvus_client = get_client()

    existed_databases = await milvus_client.list_databases()

    logger.info("================== Existed_databases ==================")
    for db in existed_databases:
        logger.info(db)
    logger.info("================== Existed_databases ==================")

    if db_name not in existed_databases:
        await milvus_client.create_database(db_name=db_name)

    logger.info("================== Existed_databases ==================")
    for db in await milvus_client.list_databases():
        logger.info(db)
    logger.info("================== Existed_databases ==================")

    await milvus_client.use_database(db_name=db_name)

    _ready = False


async def ensure_docs_collection():
    global _ready
    if _ready:
        return

    async with _ready_lock:
        if _ready:
            return

        milvus_client = get_client()

        existed_databases = await milvus_client.list_databases()
        if db_name not in existed_databases:
            temp_client = AsyncMilvusClient(uri=MILVUS_URI)
            try:
                await temp_client.create_database(db_name=db_name)
            finally:
                await temp_client.close()

        await milvus_client.use_database(db_name=db_name)

        if not await milvus_client.has_collection(collection_name=collection_name):
            await milvus_client.create_collection(
                collection_name=collection_name,
                dimension=Embedding_DIM,
                metric_type="COSINE",
                auto_id=False,
                id_type="string",
                max_length=512,
            )

        _ready = True


async def recreate_docs_collec():
    global _ready
    milvus_client = get_client()
    if await milvus_client.has_collection(collection_name=collection_name):
        await milvus_client.drop_collection(collection_name=collection_name)

    await milvus_client.create_collection(
        collection_name=collection_name,
        dimension=Embedding_DIM,
        metric_type="COSINE",
        auto_id=False,
        id_type="string",
        max_length=512,
    )
    _ready = True
    await _list_collections()


async def _list_collections():
    milvus_client = get_client()
    existed_collections = await milvus_client.list_collections()
    logger.info("Existed collections:")
    for collec in existed_collections:
        logger.info(collec)
    logger.info("-" * 50)


def _filter_string_literal(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


async def upsertToMilvus(data: list[dict]):
    await ensure_docs_collection()
    milvus_client = get_client()
    upsert_res = await milvus_client.upsert(
        collection_name=collection_name,
        data=data,
    )
    logger.info(f"Upsert result : {upsert_res}")
    logger.info("-" * 50)
    await milvus_client.flush(collection_name=collection_name)

    stats = await milvus_client.get_collection_stats(collection_name=collection_name)
    logger.info(f"{collection_name} stats: {stats}")
    logger.info("-" * 50)


async def delete_chunks_by_source(source: str) -> int:
    await ensure_docs_collection()
    milvus_client = get_client()
    res = await milvus_client.delete(
        collection_name=collection_name,
        filter=f"source == {_filter_string_literal(source)}",
    )
    await milvus_client.flush(collection_name=collection_name)
    return int(res.get("delete_count", 0)) if isinstance(res, dict) else 0


async def count_chunks(source: str | None = None) -> int:
    await ensure_docs_collection()
    milvus_client = get_client()
    expr = "" if source is None else f"source == {_filter_string_literal(source)}"
    result = await milvus_client.query(
        collection_name=collection_name,
        filter=expr,
        output_fields=["count(*)"],
    )
    if not result:
        return 0
    return int(result[0].get("count(*)", 0))


async def aclose():
    await get_client().close()


async def retrieve(query_vector: list[float], limit: int = 10) -> list[dict]:
    milvus_client = get_client()
    results = await milvus_client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=limit,
        output_fields=["id", "text", "source", "metadata"],
    )
    return results[0]

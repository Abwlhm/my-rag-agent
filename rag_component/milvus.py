import asyncio
import logging

from pymilvus import AsyncMilvusClient
from rich import print as rprint
from rag_component.embedding import embedding_model

db_name = "rag_demo"
collection_name = "docs"
Embedding_DIM = 4096
client = AsyncMilvusClient(uri="http://127.0.0.1:19530", db_name=db_name)

logger = logging.getLogger(__name__)


async def create_rag_db():
    global client
    client = AsyncMilvusClient(uri="http://127.0.0.1:19530")

    existed_databases = await client.list_databases()

    logger.info("================== Existed_databases ==================")
    for db in existed_databases:
        logger.info(db)
    logger.info("================== Existed_databases ==================")

    if db_name not in existed_databases:
        await client.create_database(db_name=db_name)

    logger.info("================== Existed_databases ==================")
    for db in await client.list_databases():
        logger.info(db)
    logger.info("================== Existed_databases ==================")

    await client.use_database(db_name=db_name)


async def create_docs_collec():
    if await client.has_collection(collection_name=collection_name):
        await client.drop_collection(collection_name=collection_name)

    await client.create_collection(
        collection_name=collection_name,
        dimension=Embedding_DIM,
        metric_type="COSINE",
        auto_id=False,
    )
    await _list_collections()


async def _list_collections():
    existed_collections = await client.list_collections()
    logger.info("Existed collections:")
    for collec in existed_collections:
        logger.info(collec)
    logger.info("-" * 50)


async def upsertToMilvus(data: list[dict]):
    await create_docs_collec()
    upsert_res = await client.upsert(
        collection_name=collection_name,
        data=data,
    )
    logger.info(f"Upsert result : {upsert_res}")
    logger.info("-" * 50)
    await client.flush(collection_name=collection_name)

    stats = await client.get_collection_stats(collection_name=collection_name)
    logger.info(f"{collection_name} stats: {stats}")
    logger.info("-" * 50)


async def retrieve(query_vector: list[float], limit: int = 10) -> list[dict]:
    results = await client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=limit,
        output_fields=["id", "text", "source", "metadata"],
    )
    return results[0]


async def main():
    if db_name not in await client.list_databases():
        await client.create_database(db_name=db_name)
    await client.use_database(db_name=db_name)

    await _list_collections()
    if await client.has_collection(collection_name=collection_name):
        await client.drop_collection(collection_name=collection_name)
    await client.create_collection(
        collection_name=collection_name,
        dimension=Embedding_DIM,
        metric_type="COSINE",
        auto_id=False,
    )

    await _list_collections()
    metadata = await client.describe_collection(collection_name=collection_name)
    rprint(metadata)

    texts = ["水果", "家具", "宇宙", "漂亮"]
    vectors = await embedding_model.aembed_documents(texts)

    data = [
        {"id": i, "vector": vectors[i], "text": texts[i]} for i in range(len(texts))
    ]
    insert_res = await client.upsert(
        collection_name=collection_name,
        data=data,
    )
    print("insert result : ", insert_res)
    print("-" * 50)
    await client.flush(collection_name=collection_name)

    stats = await client.get_collection_stats(collection_name=collection_name)
    print("stats : ", stats)
    print("-" * 50)

    results = await client.query(
        collection_name=collection_name,
        filter="",
        output_fields=["id", "text"],
        limit=100,
    )

    for res in results:
        print(res)
    print("-" * 50)

    query = "好看"
    query_vector = await embedding_model.aembed_query(query)
    results = await client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=10,
        output_fields=["id", "text"],
    )

    print("query:", query)
    for res in results[0]:
        print(res)
    print("-" * 50)

    await client.close()

    print("\n\nDONE!")


if __name__ == "__main__":
    asyncio.run(main())

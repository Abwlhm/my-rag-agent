import asyncio
import logging

from db.milvus import client as milvus
from db.postgres import connection
from db.postgres import documents as doc_table


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s:%(lineno)d:%(message)s"
    )

    await milvus.create_rag_db()

    if await milvus.get_client().has_collection(collection_name=milvus.collection_name):
        print("重建前的集合结构：")
        print(await milvus.get_client().describe_collection(collection_name=milvus.collection_name))

    await milvus.recreate_docs_collec()
    print("重建后的集合结构：")
    print(await milvus.get_client().describe_collection(collection_name=milvus.collection_name))

    cleared = await doc_table.clear_documents()
    print(f"已清空 documents 台账表：{cleared} 行")

    print(f"docs 集合当前 chunk 数：{await milvus.count_chunks()}")

    await milvus.aclose()
    await connection.close()
    print("\nDONE!")


if __name__ == "__main__":
    asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)

r"""reset_docs_db.py —— 重建"知识库文档"存储（谨慎使用：会清空已入库内容）

删除并重建 Milvus docs 集合 + 清空 PostgreSQL documents 台账表；
assets 下的原始文件不会被删除。适用于集合结构变化、无法增量使用时。

运行（cwd 必须是工作区根目录）：
    & "D:\Python\Python314\python.exe" -m test.reset_docs_db
"""

import asyncio
import logging

from db.milvus import client as milvus
from db.postgres import connection
from db.postgres import documents as doc_table


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s:%(lineno)d:%(message)s"
    )

    # 建库 + 切库（不存在时创建）
    await milvus.create_rag_db()

    # 重建前后打印集合结构，便于对比 schema 改动
    if await milvus.get_client().has_collection(collection_name=milvus.collection_name):
        print("重建前的集合结构：")
        print(await milvus.get_client().describe_collection(collection_name=milvus.collection_name))

    await milvus.recreate_docs_collec()
    print("重建后的集合结构：")
    print(await milvus.get_client().describe_collection(collection_name=milvus.collection_name))

    # 清空台账表（与向量库保持同一状态）
    cleared = await doc_table.clear_documents()
    print(f"已清空 documents 台账表：{cleared} 行")

    print(f"docs 集合当前 chunk 数：{await milvus.count_chunks()}")

    # Milvus 客户端与 Postgres 连接池都要显式关闭，否则池的后台补连任务会拖住进程退出
    await milvus.aclose()
    await connection.close()
    print("\nDONE!")


if __name__ == "__main__":
    # Windows 上 psycopg 异步需要 SelectorEventLoop
    asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)

"""documents.py —— 文档台账（PostgreSQL）：哪些文档入了库、分了多少块、状态如何。

Milvus 只有 chunk 级数据、没有"文档"概念与上传状态，故另立台账表作为
Web 端"文档列表"的唯一数据来源，通过 source（相对路径）与 Milvus chunk 关联。
只做文档元数据增删改查，不碰 Milvus / LLM / 文件系统。
"""

import asyncio

from db.postgres.connection import ensure_pool

# 入库状态：解析中 / 已入库 / 失败
STATUS_PARSING = "parsing"
STATUS_READY = "ready"
STATUS_FAILED = "failed"

# 建表语句。关键约定：
#   id     文档编号，同时是 Milvus chunk 主键的前缀（f"{id}_{序号}"）
#   source Milvus chunk 的 source 字段值 = 磁盘相对路径；唯一约束即"同分类+同名=覆盖上传"
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    file_name    TEXT        NOT NULL,
    category     TEXT        NOT NULL DEFAULT '',
    source       TEXT        NOT NULL UNIQUE,
    stored_path  TEXT        NOT NULL,
    file_size    BIGINT      NOT NULL DEFAULT 0,
    chunk_count  INTEGER     NOT NULL DEFAULT 0,
    status       TEXT        NOT NULL,
    error        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS documents_updated_at_idx ON documents (updated_at DESC)
"""

# 懒加载标记：建表是幂等 DDL，没必要每次请求都发一次
_table_ready = False
_table_lock = asyncio.Lock()  # Python 3.10+ 的 Lock 不绑定事件循环，模块级安全


async def ensure_table() -> None:
    """幂等建表：首次调用真正执行 DDL，之后直接返回。"""
    global _table_ready
    if _table_ready:
        return

    async with _table_lock:
        if _table_ready:
            return
        pool = await ensure_pool()  # 顺带完成连接池初始化
        async with pool.connection() as conn:
            await conn.execute(CREATE_TABLE_SQL)
            await conn.execute(CREATE_INDEX_SQL)
        _table_ready = True


async def register_upload(
    *,
    doc_id: str,
    file_name: str,
    category: str,
    source: str,
    stored_path: str,
    file_size: int,
) -> str:
    """登记一次上传（同 source 走 UPDATE=覆盖上传，复用原 id），返回实际生效的文档编号。"""
    await ensure_table()

    sql = """
        INSERT INTO documents
            (id, file_name, category, source, stored_path, file_size,
             chunk_count, status, error)
        VALUES (%s, %s, %s, %s, %s, %s, 0, %s, NULL)
        ON CONFLICT (source) DO UPDATE SET
            file_name   = EXCLUDED.file_name,
            category    = EXCLUDED.category,
            stored_path = EXCLUDED.stored_path,
            file_size   = EXCLUDED.file_size,
            chunk_count = 0,
            status      = EXCLUDED.status,
            error       = NULL,
            updated_at  = now()
        RETURNING id
    """
    params = (
        doc_id,
        file_name,
        category,
        source,
        stored_path,
        file_size,
        STATUS_PARSING,
    )

    pool = await ensure_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, params)
        row = await cur.fetchone()

    # 池配置了 row_factory=dict_row，因此这里拿到的是 dict
    return str(row["id"]) if row else doc_id


async def mark_ready(doc_id: str, chunk_count: int) -> None:
    """标记入库成功：写入 chunk 数、清空错误信息。"""
    await ensure_table()
    sql = """
        UPDATE documents
        SET status = %s, chunk_count = %s, error = NULL, updated_at = now()
        WHERE id = %s
    """
    pool = await ensure_pool()
    async with pool.connection() as conn:
        await conn.execute(sql, (STATUS_READY, chunk_count, doc_id))


async def mark_failed(doc_id: str, error: str) -> None:
    """标记入库失败：记录原因，chunk 数归零（旧 chunk 已在覆盖流程里删掉）。"""
    await ensure_table()
    sql = """
        UPDATE documents
        SET status = %s, chunk_count = 0, error = %s, updated_at = now()
        WHERE id = %s
    """
    pool = await ensure_pool()
    async with pool.connection() as conn:
        await conn.execute(sql, (STATUS_FAILED, error[:2000], doc_id))


async def list_documents() -> list[dict]:
    """列出全部文档，最近更新的排前面（Web 端"文档列表"直接用它）。"""
    await ensure_table()
    sql = """
        SELECT id, file_name, category, source, stored_path, file_size,
               chunk_count, status, error, created_at, updated_at
        FROM documents
        ORDER BY updated_at DESC
    """
    pool = await ensure_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql)
        rows = await cur.fetchall()
    return list(rows)


async def get_document(doc_id: str) -> dict | None:
    """按文档编号取一条，不存在时返回 None。"""
    await ensure_table()
    sql = """
        SELECT id, file_name, category, source, stored_path, file_size,
               chunk_count, status, error, created_at, updated_at
        FROM documents
        WHERE id = %s
    """
    pool = await ensure_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, (doc_id,))
        row = await cur.fetchone()
    return dict(row) if row else None


async def find_by_source(source: str) -> dict | None:
    """按 source（相对路径）取一条，不存在时返回 None。"""
    await ensure_table()
    sql = "SELECT id, source, status FROM documents WHERE source = %s"
    pool = await ensure_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, (source,))
        row = await cur.fetchone()
    return dict(row) if row else None


async def delete_document(doc_id: str) -> bool:
    """按文档编号删除台账行：删掉了返回 True，本来就不存在返回 False（幂等）。"""
    await ensure_table()
    sql = "DELETE FROM documents WHERE id = %s RETURNING id"
    pool = await ensure_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, (doc_id,))
        row = await cur.fetchone()
    return row is not None


async def list_categories() -> list[str]:
    """已有分类（去重、升序），前端用它渲染 datalist 候选。"""
    await ensure_table()
    sql = """
        SELECT DISTINCT category
        FROM documents
        WHERE category <> ''
        ORDER BY category
    """
    pool = await ensure_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql)
        rows = await cur.fetchall()
    return [str(row["category"]) for row in rows]


async def clear_documents() -> int:
    """清空整张台账表，返回删除行数（只给重置脚本/测试用；日常删除走 delete_document）。"""
    await ensure_table()
    sql = "DELETE FROM documents"
    pool = await ensure_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql)
    return cur.rowcount or 0

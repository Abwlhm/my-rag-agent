import asyncio

from db.postgres.connection import ensure_pool

STATUS_PARSING = "parsing"
STATUS_READY = "ready"
STATUS_FAILED = "failed"

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

_table_ready = False
_table_lock = asyncio.Lock()


async def ensure_table() -> None:
    global _table_ready
    if _table_ready:
        return

    async with _table_lock:
        if _table_ready:
            return
        pool = await ensure_pool()
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

    return str(row["id"]) if row else doc_id


async def mark_ready(doc_id: str, chunk_count: int) -> None:
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
    await ensure_table()
    sql = "SELECT id, source, status FROM documents WHERE source = %s"
    pool = await ensure_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, (source,))
        row = await cur.fetchone()
    return dict(row) if row else None


async def delete_document(doc_id: str) -> bool:
    await ensure_table()
    sql = "DELETE FROM documents WHERE id = %s RETURNING id"
    pool = await ensure_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, (doc_id,))
        row = await cur.fetchone()
    return row is not None


async def list_categories() -> list[str]:
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
    await ensure_table()
    sql = "DELETE FROM documents"
    pool = await ensure_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql)
    return cur.rowcount or 0

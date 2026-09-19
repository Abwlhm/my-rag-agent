import asyncio
import os

from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

load_dotenv(override=True)

_pool: AsyncConnectionPool | None = None
_checkpointer: AsyncPostgresSaver | None = None
_init_lock = asyncio.Lock()


async def get_checkpointer() -> AsyncPostgresSaver:
    global _pool, _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    async with _init_lock:
        if _checkpointer is None:
            db_url = os.getenv("POSTGRESQL_DB_URL")
            if not db_url:
                raise RuntimeError("缺少 POSTGRESQL_DB_URL 环境变量（请检查 .env）")

            pool = AsyncConnectionPool(
                conninfo=db_url,
                open=False,
                max_size=10,
                kwargs={
                    "autocommit": True,
                    "row_factory": dict_row,
                    "prepare_threshold": 0,
                },
            )
            await pool.open()
            try:
                checkpointer = AsyncPostgresSaver(pool)
                await checkpointer.setup()
            except Exception:
                await pool.close()
                raise

            _pool = pool
            _checkpointer = checkpointer

    assert _checkpointer is not None
    return _checkpointer


async def _ensure_pool() -> AsyncConnectionPool:
    await get_checkpointer()
    assert _pool is not None
    return _pool


async def list_threads() -> list[dict]:
    pool = await _ensure_pool()

    sql = """
        SELECT thread_id,
               MAX((checkpoint->>'ts')::timestamptz) AS updated_at
        FROM checkpoints
        WHERE checkpoint_ns = ''
        GROUP BY thread_id
        ORDER BY updated_at DESC
    """
    async with pool.connection() as conn:
        cur = await conn.execute(sql)
        rows = await cur.fetchall()
    return list(rows)


async def delete_thread(thread_id: str) -> None:
    checkpointer = await get_checkpointer()
    await checkpointer.adelete_thread(thread_id)


async def close() -> None:
    global _pool, _checkpointer
    if _pool is not None:
        await _pool.close()
    _pool = None
    _checkpointer = None

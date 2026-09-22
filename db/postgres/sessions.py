from db.postgres.connection import ensure_pool, get_checkpointer


async def list_threads() -> list[dict]:
    pool = await ensure_pool()

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

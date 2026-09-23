"""sessions.py —— 会话（thread）维度的查询与删除（数据来自 checkpointer 的 checkpoints 表）"""

from db.postgres.connection import ensure_pool, get_checkpointer


async def list_threads() -> list[dict]:
    """列出所有会话（最近活跃倒序）：按 thread_id 聚合取最新 checkpoint 的 ts，只统计根命名空间。"""
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
        cur = await conn.execute(sql)  # 池配置了 dict_row
        rows = await cur.fetchall()
    return list(rows)


async def delete_thread(thread_id: str) -> None:
    """删除某会话的全部持久化数据（checkpoints / blobs / writes 三张表，幂等）。"""
    checkpointer = await get_checkpointer()
    await checkpointer.adelete_thread(thread_id)

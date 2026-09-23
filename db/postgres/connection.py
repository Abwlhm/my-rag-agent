"""connection.py —— PostgreSQL 基础设施：连接池 + LangGraph checkpointer。

懒加载单例：`async with` 建 saver 只能写在 async 函数里，且 import 时事件循环可能未就绪，
故推迟到第一次真正使用时初始化。依赖方向单向：db 层绝不 import services / agent_core。
"""

import asyncio
import os

from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

# 自己加载 .env：让本模块自包含，不依赖"调用方恰好先加载过环境变量"的副作用
load_dotenv(override=True)

# 进程内单例：连接池与 checkpointer 一起创建、一起释放，保持一致
_pool: AsyncConnectionPool | None = None
_checkpointer: AsyncPostgresSaver | None = None
# Python 3.10+ 的 Lock 不再绑定创建时的事件循环，模块级安全
_init_lock = asyncio.Lock()


async def get_checkpointer() -> AsyncPostgresSaver:
    """懒加载单例：首次调用建连接池 → setup() 幂等建表 → 返回 checkpointer。

    连接池绑定"第一次调用时"的事件循环：一个进程里不要多次 asyncio.run(...) 数据库操作；
    Windows 上 psycopg(异步) 还要求 SelectorEventLoop。
    """
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
                open=False,  # 显式 open，出错好兜底
                max_size=10,
                kwargs={
                    # autocommit + dict_row 是 checkpointer 的硬性要求（官方 README）
                    "autocommit": True,
                    "row_factory": dict_row,
                    "prepare_threshold": 0,  # 禁用预处理语句，规避连接复用兼容问题
                },
            )
            await pool.open()
            try:
                checkpointer = AsyncPostgresSaver(pool)
                await checkpointer.setup()  # 幂等建表（checkpoints 等 4 张）
            except Exception:
                await pool.close()  # 初始化失败别把池悬着
                raise

            # 全部成功后才落单例：半途失败保持"未初始化"，下次调用可重试
            _pool = pool
            _checkpointer = checkpointer

    assert _checkpointer is not None
    return _checkpointer


async def _ensure_pool() -> AsyncConnectionPool:
    """确保连接池已建立后返回它。"""
    await get_checkpointer()
    assert _pool is not None
    return _pool


async def ensure_pool() -> AsyncConnectionPool:
    """公开版本：让 db 层其它模块复用同一个连接池。"""
    return await _ensure_pool()


async def close() -> None:
    """释放连接池并重置单例（可重复调用）。"""
    global _pool, _checkpointer
    if _pool is not None:
        await _pool.close()
    _pool = None
    _checkpointer = None

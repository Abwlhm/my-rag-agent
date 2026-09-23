"""db 层拆分后的冒烟测试（可重复执行；依赖本机 PostgreSQL + DeepSeek）。

验证：依赖方向检查、建池建表、图编译、一轮真实流式问答 → 读历史 → 会话列表 → 删除 → 收尾。
运行方式（项目根目录）：
    & "D:\\Python\\Python314\\python.exe" -m test.persistence_check

注意：Windows 上 psycopg(异步) 必须用 SelectorEventLoop（脚本末尾已处理）。
"""

import asyncio
import sys

# 先导入 db 层，随后立刻做依赖方向检查（必须发生在导入上层模块之前）
from db.postgres import connection
from db.postgres import sessions

# db 层只应依赖 pymilvus / psycopg：一旦 import 了上层包，说明方向反了
_UPPER_LAYERS = ("agent_core", "services", "rag_component", "backend_api")
_leaked = [m for m in sys.modules if m.split(".")[0] in _UPPER_LAYERS]
assert not _leaked, f"db 层反向依赖了上层：{_leaked}"

from services import chat_service  # noqa: E402
import backend_api.main  # noqa: E402,F401


async def main() -> None:
    thread_id = "persistence-smoke-0001"

    await sessions.delete_thread(thread_id)  # 清掉旧测试会话，保证可重复执行

    await sessions.list_threads()
    print("[1] sessions.list_threads() 可用（连接池 / checkpointer 初始化成功）")

    assert await chat_service.get_history(thread_id) == []
    print("[2] get_graph() 编译成功（新 thread 的历史为空，符合预期）")

    # 问候语走"直接回答"分支（不碰 Milvus / 重排），完整经过 condense → route → direct → llm
    print("[3] 流式问答（你好）...", end="", flush=True)
    pieces: list[str] = []
    async for text in chat_service.get_answer_stream("你好", thread_id):
        pieces.append(text)
    answer = "".join(pieces)
    assert answer.strip(), "模型没有返回任何内容"
    print(f"收到 {len(answer)} 字符：{answer[:50]!r}")

    history = await chat_service.get_history(thread_id)
    roles = [m["role"] for m in history]
    assert roles == ["user", "assistant"], f"历史角色序列不符合预期：{roles}"
    assert history[0]["content"] == "你好", f"第一条不是用户问题：{history[0]!r}"
    print(f"[4] get_history 返回 {len(history)} 条，角色 = {roles}")

    ids = [row["thread_id"] for row in await sessions.list_threads()]
    assert thread_id in ids, "新会话未出现在会话列表中"
    print("[5] 新会话已出现在 list_threads 结果里")

    await chat_service.delete_thread(thread_id)
    assert await chat_service.get_history(thread_id) == [], "删除后仍能读到历史"
    ids = [row["thread_id"] for row in await sessions.list_threads()]
    assert thread_id not in ids, "删除后仍出现在会话列表里"
    print("[6] delete_thread 生效：历史为空且已从列表移除")

    print("全部检查通过")


async def _run() -> None:
    try:
        await main()
    finally:
        await chat_service.aclose()
        print("[7] chat_service.aclose() 完成：连接池已释放")


if __name__ == "__main__":
    # Windows 上 psycopg(异步) 与默认 ProactorEventLoop 不兼容，必须 SelectorEventLoop
    asyncio.run(_run(), loop_factory=asyncio.SelectorEventLoop)

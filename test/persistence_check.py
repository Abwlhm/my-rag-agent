import asyncio
import sys

from db.postgres import connection
from db.postgres import sessions

_UPPER_LAYERS = ("agent_core", "services", "rag_component", "backend_api")
_leaked = [m for m in sys.modules if m.split(".")[0] in _UPPER_LAYERS]
assert not _leaked, f"db 层反向依赖了上层：{_leaked}"

from services import chat_service

import backend_api.main


async def main() -> None:
    thread_id = "persistence-smoke-0001"

    await sessions.delete_thread(thread_id)

    await sessions.list_threads()
    print("[1] sessions.list_threads() 可用（连接池 / checkpointer 初始化成功）")

    assert await chat_service.get_history(thread_id) == []
    print("[2] get_graph() 编译成功（新 thread 的历史为空，符合预期）")

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
    asyncio.run(_run(), loop_factory=asyncio.SelectorEventLoop)

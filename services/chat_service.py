"""chat_service.py —— 对话会话的应用服务（Web 层唯一依赖的对话入口）。

用例门面：组装 agent_core.rag_agent 的图与 db 层会话查询，自身不含 Agent 能力。
多轮历史由 checkpointer 按 thread_id 存 PostgreSQL，前端只透传 thread_id。
"""

from langchain_core.messages import HumanMessage

from agent_core.rag_agent import aclose_graph, aget_history, ask_stream
from db.postgres import sessions


async def get_answer_stream(
    query: str, thread_id: str | None = None, mode: str = "auto"
):
    """流式生成回答文本；thread_id 缺省时自动生成 uuid4（无记忆单轮问答）。

    mode："auto"=自动路由（默认）/"retrieve"=强制查询数据库/"direct"=强制不查询数据库。
    """
    async for part in ask_stream(query, thread_id, mode):
        yield part


async def list_threads() -> list[dict]:
    """列出所有历史会话，按最近活跃倒序。"""
    return await sessions.list_threads()


async def get_history(thread_id: str) -> list[dict]:
    """读取会话聊天记录，转成 [{"role", "content"}]（剔除 system；不存在返回空列表）。"""
    messages = await aget_history(thread_id)
    result: list[dict] = []
    for message in messages:
        result.append(
            {
                "role": "user" if isinstance(message, HumanMessage) else "assistant",
                "content": (
                    message.content
                    if isinstance(message.content, str)
                    else str(message.content)
                ),
            }
        )
    return result


async def delete_thread(thread_id: str) -> None:
    """删除某个会话的持久化数据（checkpoints / blobs / writes 三张表）。"""
    await sessions.delete_thread(thread_id)


async def aclose() -> None:
    """释放 Postgres 连接池（FastAPI shutdown 时调用）。"""
    await aclose_graph()

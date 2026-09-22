from langchain_core.messages import HumanMessage

from agent_core.rag_agent import aclose_graph, aget_history, ask_stream
from db.postgres import sessions


async def get_answer_stream(query: str, thread_id: str | None = None):
    async for part in ask_stream(query, thread_id):
        yield part


async def list_threads() -> list[dict]:
    return await sessions.list_threads()


async def get_history(thread_id: str) -> list[dict]:
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
    await sessions.delete_thread(thread_id)


async def aclose() -> None:
    await aclose_graph()

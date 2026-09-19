import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent_core import chat_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await chat_agent.aclose()


app = FastAPI(
    title="对话 Agent 接口",
    description="前端(React) -> 本接口(SSE/JSON) -> Agent(LangGraph) -> DeepSeek；会话与记忆存 PostgreSQL",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    query: str = Field(description="用户最新的输入内容")
    thread_id: str | None = Field(
        default=None,
        description=(
            "会话编号；不传时后端自动生成 uuid4，"
            "等价于无记忆的单轮问答（方便 curl 调试）"
        ),
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        try:
            async for text in chat_agent.get_answer_stream(req.query, req.thread_id):
                event = {"type": "content", "text": text}
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            event = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            yield "data: " + json.dumps({"type": "done"}) + "\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/sessions")
async def list_sessions():
    sessions = await chat_agent.list_threads()
    return {"sessions": sessions}


@app.get("/api/sessions/{thread_id}/messages")
async def get_session_messages(thread_id: str):
    messages = await chat_agent.get_history(thread_id)
    return {"thread_id": thread_id, "messages": messages}


@app.delete("/api/sessions/{thread_id}")
async def delete_session(thread_id: str):
    await chat_agent.delete_thread(thread_id)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend_api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        loop="asyncio:SelectorEventLoop",
    )

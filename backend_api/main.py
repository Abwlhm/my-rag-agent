import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from services import chat_service
from services import document_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await chat_service.aclose()
    await document_service.aclose()


app = FastAPI(
    title="对话 Agent 接口",
    description=(
        "前端(React) -> 本接口(SSE/JSON) -> Agent(LangGraph) -> DeepSeek；"
        "会话与记忆存 PostgreSQL，知识库 chunk 存 Milvus，文档台账存 PostgreSQL"
    ),
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
            async for text in chat_service.get_answer_stream(req.query, req.thread_id):
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
    sessions = await chat_service.list_threads()
    return {"sessions": sessions}


@app.get("/api/sessions/{thread_id}/messages")
async def get_session_messages(thread_id: str):
    messages = await chat_service.get_history(thread_id)
    return {"thread_id": thread_id, "messages": messages}


@app.delete("/api/sessions/{thread_id}")
async def delete_session(thread_id: str):
    await chat_service.delete_thread(thread_id)
    return {"ok": True}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


async def _iter_upload(upload: UploadFile) -> AsyncIterator[bytes]:
    while True:
        chunk = await upload.read(document_service.WRITE_CHUNK_SIZE)
        if not chunk:
            return
        yield chunk


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(description="要入库的文档（PDF/DOCX/PPTX/TXT/MD…）"),
    file_name: str = Form(default="", description="文件名；留空则用上传时的原始文件名"),
    category: str = Form(default="", description="分类（可留空 → 直接放 assets 根目录）"),
):
    try:
        plan = document_service.plan_upload(
            file_name=file_name or (file.filename or ""),
            category=category,
        )
    except document_service.UploadError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    async def event_generator():
        try:
            await document_service.ensure_backend()

            yield _sse(
                {
                    "type": "progress",
                    "step": "save",
                    "status": "running",
                    "message": f"正在接收文件：{plan.file_name}",
                }
            )
            file_size = await document_service.save_upload(_iter_upload(file), plan)
            size_text = (
                f"{file_size / 1024:.1f} KB"
                if file_size < 1024 * 1024
                else f"{file_size / 1024 / 1024:.2f} MB"
            )
            yield _sse(
                {
                    "type": "progress",
                    "step": "save",
                    "status": "done",
                    "message": f"文件已接收（{size_text}）",
                    "percent": document_service.STEP_PERCENT["save"],
                }
            )

            async for event in document_service.ingest(plan, file_size=file_size):
                yield _sse(event)
        except document_service.UploadError as exc:
            yield _sse({"type": "error", "message": str(exc)})
        except Exception as exc:
            logger.exception("文档入库失败")
            yield _sse({"type": "error", "message": f"入库失败：{exc}"})
        finally:
            await file.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/documents")
async def list_documents():
    await document_service.ensure_backend()
    return {"documents": await document_service.list_documents()}


@app.get("/api/documents/categories")
async def list_categories():
    await document_service.ensure_backend()
    return {"categories": await document_service.list_categories()}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    await document_service.ensure_backend()
    return await document_service.delete_document(doc_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend_api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        loop="asyncio:SelectorEventLoop",
    )

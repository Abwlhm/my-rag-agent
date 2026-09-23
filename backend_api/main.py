"""
main.py —— FastAPI 后端：会话/问答 + 知识库文档接口。

POST /api/chat/stream
    请求体: {"query": "用户提问", "thread_id": "会话编号（可选）"}
    响应体: SSE 文本流，每个事件一行：

        data: {"type": "content", "text": "回答的一段文本"}\n\n   <- 反复出现
        data: {"type": "done"}\n\n                                <- 回答结束
        （出错时会出现 data: {"type": "error", "message": "..."}）

GET    /api/sessions                      会话列表（最近活跃倒序）
GET    /api/sessions/{thread_id}/messages 会话聊天记录
DELETE /api/sessions/{thread_id}          删除会话（幂等）

知识库文档（内容 → Milvus，台账 → PostgreSQL）：
POST   /api/documents/upload              multipart 上传 + SSE 进度流
       进度: data: {"type":"progress","step":"save|parse|split|embed|write",
                    "status":"running|done","message":"...","percent":30}
       成功: data: {"type":"done","document":{...}}
       失败: data: {"type":"error","message":"..."}
       （流开始前就能发现的错误用 400 + {"detail":"..."}）
GET    /api/documents                     文档列表（最近更新倒序）
GET    /api/documents/categories          已有分类（前端 datalist 候选）
DELETE /api/documents/{doc_id}            删除文档：清 Milvus chunk + 删台账行（幂等）

多轮历史由服务端 checkpointer 按 thread_id 存 PostgreSQL，前端只透传 thread_id。
"""
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from services import chat_service
from services import document_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子：进程退出时释放 Postgres 连接池与 Milvus 客户端。"""
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
    """一次问答请求：最新问题 + 会话编号（历史由服务端按 thread_id 维护）+ 检索策略。"""
    query: str = Field(description="用户最新的输入内容")
    thread_id: str | None = Field(
        default=None,
        description=(
            "会话编号；不传时后端自动生成 uuid4，"
            "等价于无记忆的单轮问答（方便 curl 调试）"
        ),
    )
    mode: Literal["auto", "retrieve", "direct"] = Field(
        default="auto",
        description=(
            "检索策略：auto=自动路由（默认），"
            "retrieve=强制查询数据库，direct=强制不查询数据库"
        ),
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """核心接口：返回 SSE 流，把 Agent 的回答逐段推给调用方。"""

    async def event_generator():
        try:
            async for text in chat_service.get_answer_stream(
                req.query, req.thread_id, req.mode
            ):
                # 包成 JSON 事件，ensure_ascii=False 保证中文不转义
                event = {"type": "content", "text": text}
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            # 异常在流内告知前端，避免前端一直傻等
            event = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            yield "data: " + json.dumps({"type": "done"}) + "\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",      # 不允许缓存，保证实时
            "X-Accel-Buffering": "no",        # 关闭代理缓冲（否则不会逐字到达）
        },
    )


@app.get("/api/sessions")
async def list_sessions():
    """会话列表（读 PostgreSQL checkpoints 表，最近活跃倒序）。"""
    sessions = await chat_service.list_threads()
    return {"sessions": sessions}


@app.get("/api/sessions/{thread_id}/messages")
async def get_session_messages(thread_id: str):
    """某个会话的聊天记录（不含 system 提示词）。"""
    messages = await chat_service.get_history(thread_id)
    return {"thread_id": thread_id, "messages": messages}


@app.delete("/api/sessions/{thread_id}")
async def delete_session(thread_id: str):
    """删除会话的全部持久化数据（幂等：不存在也返回成功）。"""
    await chat_service.delete_thread(thread_id)
    return {"ok": True}


def _sse(payload: dict) -> str:
    """把一条事件编码成 SSE 文本（ensure_ascii=False 中文不转义；default=str 序列化 datetime）。"""
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


async def _iter_upload(upload: UploadFile) -> AsyncIterator[bytes]:
    """把 UploadFile 适配成异步字节块迭代器（显式读-产出循环，不依赖版本行为）。"""
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
    """上传并入库一个文档：返回 SSE 流，实时汇报各阶段进度。

    为什么体积超限也走 error 事件而不是 400：SSE 响应头在开始流式输出时已发出，
    中途错误没法再改状态码。流开始前能发现的错误（类型/分类非法）用 400 + JSON。
    """
    # 前置校验：类型白名单、文件名/分类合法性、落盘路径都在这算好
    try:
        plan = document_service.plan_upload(
            file_name=file_name or (file.filename or ""),
            category=category,
        )
    except document_service.UploadError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    async def event_generator():
        try:
            await document_service.ensure_backend()  # 懒加载 Milvus 库/集合与台账表

            # 1) 收文件（边收边做 50MB 限流）
            yield _sse(
                {
                    "type": "progress",
                    "step": "save",
                    "status": "running",
                    "message": f"正在接收文件：{plan.file_name}",
                }
            )
            file_size = await document_service.save_upload(_iter_upload(file), plan)
            # 字节数换成好读的单位（小文件别显示成 0.00 MB）
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

            # 2) 交给业务层：登记台账 → 删旧 chunk → 解析/切分/向量化/写入 → 改名收尾
            async for event in document_service.ingest(plan, file_size=file_size):
                yield _sse(event)
        except document_service.UploadError as exc:
            # 可预期的错误（超限、空文件…）：消息本身就是给用户看的提示
            yield _sse({"type": "error", "message": str(exc)})
        except Exception as exc:
            logger.exception("文档入库失败")
            yield _sse({"type": "error", "message": f"入库失败：{exc}"})
        finally:
            await file.close()  # 释放 Starlette 为上传临时落盘的文件句柄

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
    """文档列表：台账里全部文档（最近更新的排前面）。"""
    await document_service.ensure_backend()
    return {"documents": await document_service.list_documents()}


@app.get("/api/documents/categories")
async def list_categories():
    """已有分类（去重升序），前端分类输入框的 datalist 候选。

    注意：此路由必须排在 /api/documents/{doc_id} 之前，否则 categories 会被当成文档编号。
    """
    await document_service.ensure_backend()
    return {"categories": await document_service.list_categories()}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档：清 Milvus chunk + 删台账行（幂等）。"""
    await document_service.ensure_backend()
    return await document_service.delete_document(doc_id)


if __name__ == "__main__":
    logging.basicConfig(
            level=logging.INFO, format="%(levelname)s:%(name)s:%(lineno)d:%(message)s"
        )
    import uvicorn

    uvicorn.run(
        "backend_api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        loop="asyncio:SelectorEventLoop",
    )

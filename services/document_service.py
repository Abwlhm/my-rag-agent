import asyncio
import logging
import os
import re
import uuid
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from constants import STEP_PERCENT
from db.milvus import client as milvus_client
from db.postgres import documents as doc_table
from rag_component.loader import LOCAL_LOADERS, MINERU_EXTENSIONS
from rag_component.pipeline import ingest_file

logger = logging.getLogger(__name__)

MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

WRITE_CHUNK_SIZE = 1024 * 1024

ASSETS_DIR = Path("assets")

_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

MAX_CATEGORY_LEN = 32
MAX_FILE_NAME_LEN = 120

ALLOWED_SUFFIXES = frozenset(MINERU_EXTENSIONS) | frozenset(LOCAL_LOADERS)


class UploadError(Exception):
    pass


class UploadTooLarge(UploadError):
    pass


@dataclass(frozen=True)
class UploadPlan:

    doc_id: str
    file_name: str
    category: str
    suffix: str
    source: str
    stored_path: str
    target_path: Path
    temp_path: Path


def _clean_file_name(raw: str) -> str:
    name = (raw or "").strip().rstrip(". ")
    if not name:
        raise UploadError("文件名不能为空")
    if "/" in name or "\\" in name:
        raise UploadError("文件名不能包含路径分隔符")
    name = _ILLEGAL_CHARS.sub("_", name)
    if len(name) > MAX_FILE_NAME_LEN:
        suffix = Path(name).suffix
        name = f"{name[: MAX_FILE_NAME_LEN - len(suffix)]}{suffix}"
    if name.upper() in _RESERVED_NAMES:
        raise UploadError(f"文件名不能使用系统保留名：{raw}")
    return name


def _clean_category(raw: str) -> str:
    if not (raw or "").strip():
        return ""
    if "/" in raw or "\\" in raw:
        raise UploadError("分类里不能包含路径分隔符（/ 或 \\）")
    name = _ILLEGAL_CHARS.sub("_", raw.strip().rstrip(". "))
    if len(name) > MAX_CATEGORY_LEN:
        raise UploadError(f"分类名太长（最多 {MAX_CATEGORY_LEN} 个字符）")
    if name.upper() in _RESERVED_NAMES:
        raise UploadError(f"分类名不能使用系统保留名：{raw}")
    return name


def plan_upload(*, file_name: str, category: str = "") -> UploadPlan:
    safe_name = _clean_file_name(file_name)
    safe_category = _clean_category(category)

    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UploadError(f"不支持的文件类型：{suffix or '无扩展名'}")

    target_path = (
        ASSETS_DIR / safe_category / safe_name
        if safe_category
        else ASSETS_DIR / safe_name
    )

    parent = target_path.parent
    if parent.exists() and not parent.is_dir():
        raise UploadError(
            f"分类名与 assets 下已有文件重名，请换一个分类名：{safe_category}"
        )

    token = uuid.uuid4().hex
    return UploadPlan(
        doc_id=token,
        file_name=safe_name,
        category=safe_category,
        suffix=suffix,
        source=target_path.as_posix(),
        stored_path=target_path.as_posix(),
        target_path=target_path,
        temp_path=target_path.with_name(f".{token}{suffix}"),
    )


async def save_upload(chunks: AsyncIterable[bytes], plan: UploadPlan) -> int:
    try:
        plan.target_path.parent.mkdir(parents=True, exist_ok=True)
    except (FileExistsError, NotADirectoryError) as exc:
        raise UploadError(f"无法创建分类目录：{plan.target_path.parent}") from exc

    total = 0
    try:
        with plan.temp_path.open("wb") as fp:
            async for chunk in chunks:
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise UploadTooLarge(f"文件超过 {MAX_UPLOAD_MB}MB 上限，已中止上传")
                fp.write(chunk)
    except BaseException:
        plan.temp_path.unlink(missing_ok=True)
        raise

    if total == 0:
        plan.temp_path.unlink(missing_ok=True)
        raise UploadError("上传的文件是空的")

    return total


async def _forward_progress(
    task: "asyncio.Task[int]", queue: "asyncio.Queue[dict]"
) -> AsyncIterator[dict]:
    while True:
        getter: asyncio.Task = asyncio.create_task(queue.get())
        await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)

        if getter.done():
            yield getter.result()
            continue

        getter.cancel()
        while not queue.empty():
            yield queue.get_nowait()
        return


async def ingest(plan: UploadPlan, *, file_size: int) -> AsyncIterator[dict]:
    doc_id = await doc_table.register_upload(
        doc_id=plan.doc_id,
        file_name=plan.file_name,
        category=plan.category,
        source=plan.source,
        stored_path=plan.stored_path,
        file_size=file_size,
    )

    renamed = False
    try:
        deleted = await milvus_client.delete_chunks_by_source(plan.source)
        if deleted:
            logger.info("覆盖上传 %s：先删除 %s 个旧 chunk", plan.source, deleted)

        queue: asyncio.Queue[dict] = asyncio.Queue()

        async def on_progress(event: dict) -> None:
            await queue.put({"type": "progress", **event})

        task: asyncio.Task[int] = asyncio.create_task(
            ingest_file(
                str(plan.temp_path),
                doc_id=doc_id,
                source=plan.source,
                file_name=plan.file_name,
                category=plan.category,
                on_progress=on_progress,
            )
        )
        async for event in _forward_progress(task, queue):
            yield event
        chunk_count = await task

        os.replace(plan.temp_path, plan.target_path)
        renamed = True
        await doc_table.mark_ready(doc_id, chunk_count)

        logger.info("文档入库完成：%s（%s 个 chunk）", plan.source, chunk_count)
        document = await doc_table.get_document(doc_id)
        yield {"type": "done", "document": document, "chunk_count": chunk_count}
    except BaseException as exc:
        logger.warning(
            "文档入库失败：%s（%s: %s）", plan.source, type(exc).__name__, exc
        )
        try:
            await doc_table.mark_failed(doc_id, f"{type(exc).__name__}: {exc}")
        except Exception as mark_exc:
            logger.error("标记文档失败状态时又出错：%s", mark_exc)
        raise
    finally:
        if not renamed:
            plan.temp_path.unlink(missing_ok=True)


async def list_documents() -> list[dict]:
    return await doc_table.list_documents()


async def delete_document(doc_id: str) -> dict:
    doc = await doc_table.get_document(doc_id)
    if doc is None:
        return {"ok": True, "deleted_chunks": 0, "file_name": None}

    deleted_chunks = await milvus_client.delete_chunks_by_source(doc["source"])
    await doc_table.delete_document(doc_id)
    logger.info("删除文档 %s：清掉 %s 个 chunk", doc["source"], deleted_chunks)
    return {
        "ok": True,
        "deleted_chunks": deleted_chunks,
        "file_name": doc["file_name"],
    }


async def list_categories() -> list[str]:
    return await doc_table.list_categories()


_ready: bool = False
_ready_lock = asyncio.Lock()


async def ensure_backend() -> None:
    global _ready
    if _ready:
        return

    async with _ready_lock:
        if _ready:
            return
        await milvus_client.create_rag_db()
        await milvus_client.ensure_docs_collection()
        await doc_table.ensure_table()
        _ready = True


async def aclose() -> None:
    await milvus_client.aclose()

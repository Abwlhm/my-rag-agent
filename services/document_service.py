"""document_service.py —— 文档入库 / 管理的业务层（Web 端文档接口背后的唯一实现）。

把"收文件"（网络 IO + 体积限流）与"内容入库"（解析/向量化/写库）拆成两步，
ingest() 直接吃磁盘上已存在的文件，测试脚本不必伪造 HTTP 上传。
依赖方向：backend_api → services → {db, rag_component}（单向）。
"""

import asyncio
import logging
import os
import re
import uuid
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from constants import STEP_PERCENT  # 跨层共享的进度常量（定义见 constants.py）
from db.milvus import client as milvus_client  # Milvus：库/集合准备 + chunk 增删查
from db.postgres import documents as doc_table  # 文档台账表（PostgreSQL）
from rag_component.loader import LOCAL_LOADERS, MINERU_EXTENSIONS
from rag_component.pipeline import ingest_file

logger = logging.getLogger(__name__)

# 上传体积上限：前端也有一份同样的常量（frontend/src/api.ts 的 MAX_UPLOAD_MB），改动请同步
MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

WRITE_CHUNK_SIZE = 1024 * 1024  # 写盘分块大小（1MB）

# 原始文档根目录（相对路径 → 后端必须以工作区根目录为 cwd 启动）
ASSETS_DIR = Path("assets")

# 文件名 / 分类名禁止的字符：Windows 非法字符 + 路径分隔符（会被当路径段使用）
_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Windows 保留设备名（作为文件/目录名行为异常，直接拒绝）
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

# 允许上传的扩展名复用 loader 的分发表，保证"能上传=能解析"不漂移
ALLOWED_SUFFIXES = frozenset(MINERU_EXTENSIONS) | frozenset(LOCAL_LOADERS)


class UploadError(Exception):
    """可预期的上传错误：消息直接透传给前端展示。"""


class UploadTooLarge(UploadError):
    """文件超过体积上限。"""


@dataclass(frozen=True)
class UploadPlan:
    """一次上传的落盘方案：由 plan_upload() 校验通过后产出。"""

    doc_id: str  # 文档编号（覆盖上传时换成台账里原有的 id）
    file_name: str  # 展示 + 落盘用的文件名（已安全化）
    category: str  # 分类（空串 = 放在 assets 根目录）
    suffix: str  # 扩展名（小写，含点）
    source: str  # Milvus 的 source 字段值（相对路径，posix 风格）
    stored_path: str  # 正式文件路径（相对路径，posix 风格）
    target_path: Path  # 正式文件路径（Path 对象）
    temp_path: Path  # 临时文件（同目录、点号开头；扩展名不变以便解析器分发）


def _clean_file_name(raw: str) -> str:
    """把用户输入的文件名清理成安全的文件名（保留扩展名）。"""
    name = (raw or "").strip().rstrip(". ")
    if not name:
        raise UploadError("文件名不能为空")
    if "/" in name or "\\" in name:
        raise UploadError("文件名不能包含路径分隔符")
    name = _ILLEGAL_CHARS.sub("_", name)
    if len(name) > MAX_FILE_NAME_LEN:
        # 超长时截断主干、保留扩展名（扩展名决定用哪个解析器，不能丢）
        suffix = Path(name).suffix
        name = f"{name[: MAX_FILE_NAME_LEN - len(suffix)]}{suffix}"
    if name.upper() in _RESERVED_NAMES:
        raise UploadError(f"文件名不能使用系统保留名：{raw}")
    return name


def _clean_category(raw: str) -> str:
    """把用户输入的分类清理成安全的目录名；不填（或只有空白）时返回空串。"""
    if not (raw or "").strip():
        return ""  # 不填分类 = 放在 assets 根目录
    if "/" in raw or "\\" in raw:
        raise UploadError("分类里不能包含路径分隔符（/ 或 \\）")
    name = _ILLEGAL_CHARS.sub("_", raw.strip().rstrip(". "))
    if len(name) > MAX_CATEGORY_LEN:
        raise UploadError(f"分类名太长（最多 {MAX_CATEGORY_LEN} 个字符）")
    if name.upper() in _RESERVED_NAMES:
        raise UploadError(f"分类名不能使用系统保留名：{raw}")
    return name


def plan_upload(*, file_name: str, category: str = "") -> UploadPlan:
    """校验文件名/分类并算出落盘路径 assets/<分类>/<文件名>（不写文件）；同路径=同 source=覆盖上传。"""
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

    # 分类名与 assets 下已有文件重名时 mkdir 会失败（Windows 不允许同名文件与目录共存），提前拦截
    parent = target_path.parent
    if parent.exists() and not parent.is_dir():
        raise UploadError(
            f"分类名与 assets 下已有文件重名，请换一个分类名：{safe_category}"
        )

    token = uuid.uuid4().hex  # 一次上传共用一个随机串（文档编号 + 临时文件名）
    return UploadPlan(
        doc_id=token,
        file_name=safe_name,
        category=safe_category,
        suffix=suffix,
        # source 用 posix 风格（正斜杠）：Milvus filter 表达式里不必转义反斜杠
        source=target_path.as_posix(),
        stored_path=target_path.as_posix(),
        target_path=target_path,
        # 临时文件保留原扩展名：解析器按扩展名分发，改成 .part 会报"类型不支持"
        temp_path=target_path.with_name(f".{token}{suffix}"),
    )


async def save_upload(chunks: AsyncIterable[bytes], plan: UploadPlan) -> int:
    """把上传字节流写进 plan.temp_path（临时文件），返回写入的字节数。

    边写边限流：超过上限立即停手、删掉半截文件（只信 Content-Length 不可靠）。
    """
    try:
        plan.target_path.parent.mkdir(parents=True, exist_ok=True)
    except (FileExistsError, NotADirectoryError) as exc:
        # 例如分类目录名与已有文件重名（plan_upload 已尽量提前拦，这里兜底）
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
                fp.write(chunk)  # 同步写：分块 1MB，阻塞时间很短
    except BaseException:
        # 超限 / 客户端断开 / 磁盘错误：都不留半截临时文件
        plan.temp_path.unlink(missing_ok=True)
        raise

    if total == 0:
        plan.temp_path.unlink(missing_ok=True)
        raise UploadError("上传的文件是空的")

    return total


async def _forward_progress(
    task: "asyncio.Task[int]", queue: "asyncio.Queue[dict]"
) -> AsyncIterator[dict]:
    """在 task 运行期间把 queue 里的进度事件逐个 yield 出来（"推"回调 → "拉"生成器的转换）。"""
    while True:
        getter: asyncio.Task = asyncio.create_task(queue.get())
        await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)

        if getter.done():
            yield getter.result()
            continue

        # 入库任务先结束：排空残留事件后收工
        getter.cancel()
        while not queue.empty():
            yield queue.get_nowait()
        return


async def ingest(plan: UploadPlan, *, file_size: int) -> AsyncIterator[dict]:
    """把 plan.temp_path 里的文档入库，边做边 yield 进度事件（事件结构同前端 UploadEvent）。

    失败不吞异常：标记台账失败、清临时文件后抛给调用方（API 层转成 error 事件）。
    """
    # 1) 登记台账：同 source 复用原行（覆盖上传），状态先置为 parsing
    doc_id = await doc_table.register_upload(
        doc_id=plan.doc_id,
        file_name=plan.file_name,
        category=plan.category,
        source=plan.source,
        stored_path=plan.stored_path,
        file_size=file_size,
    )

    renamed = False  # 临时文件是否已改名成正式文件
    try:
        # 2) 覆盖旧版本：先按 source 删掉旧 chunk，避免新旧内容混在同一份文档里
        deleted = await milvus_client.delete_chunks_by_source(plan.source)
        if deleted:
            logger.info("覆盖上传 %s：先删除 %s 个旧 chunk", plan.source, deleted)

        # 3) 内容入库（解析 → 切分 → 向量化 → 写入），进度事件经队列转成异步生成器
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
        chunk_count = await task  # 入库失败的话，异常在这里抛出

        # 4) 收尾：临时文件改名成正式文件（同目录 rename 覆盖旧版本），再更新台账
        os.replace(plan.temp_path, plan.target_path)
        renamed = True
        await doc_table.mark_ready(doc_id, chunk_count)

        logger.info("文档入库完成：%s（%s 个 chunk）", plan.source, chunk_count)
        document = await doc_table.get_document(doc_id)
        yield {"type": "done", "document": document, "chunk_count": chunk_count}
    except BaseException as exc:  # 含 asyncio.CancelledError（客户端中途断开）
        logger.warning(
            "文档入库失败：%s（%s: %s）", plan.source, type(exc).__name__, exc
        )
        try:
            await doc_table.mark_failed(doc_id, f"{type(exc).__name__}: {exc}")
        except Exception as mark_exc:
            # 标记失败本身再失败时不能掩盖原始异常，只记日志
            logger.error("标记文档失败状态时又出错：%s", mark_exc)
        raise
    finally:
        if not renamed:
            # 失败时不留半截临时文件；正式文件（上一版）保持不动
            plan.temp_path.unlink(missing_ok=True)


async def list_documents() -> list[dict]:
    """文档列表：台账里的全部文档，最近更新的排在前面。"""
    return await doc_table.list_documents()


async def delete_document(doc_id: str) -> dict:
    """删除文档：清 Milvus chunk + 删台账行（幂等）；磁盘原始文件刻意保留，便于重新入库。"""
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
    """已有分类（去重升序），前端拿它渲染 datalist 候选。"""
    return await doc_table.list_categories()


# ---- 后端资源懒加载（Milvus 库/集合 + Postgres 台账表）----
_ready: bool = False
_ready_lock = asyncio.Lock()  # Python 3.10+ 的 Lock 不绑定事件循环，模块级安全


async def ensure_backend() -> None:
    """幂等准备文档功能依赖的资源。刻意懒加载（而非启动时）：Milvus 没起来不影响聊天，失败还能自愈。"""
    global _ready
    if _ready:
        return

    async with _ready_lock:
        if _ready:
            return
        await milvus_client.create_rag_db()  # 建库（不存在时）+ 切库
        await milvus_client.ensure_docs_collection()  # 建集合（不存在时）
        await doc_table.ensure_table()  # 建台账表
        _ready = True


async def aclose() -> None:
    """释放 Milvus 客户端（由本层转发给 db 层，backend_api 不必认识 Milvus）。"""
    await milvus_client.aclose()

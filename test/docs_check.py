import asyncio
import logging
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

from db.milvus import client as milvus
from db.postgres import connection
from db.postgres import documents as doc_table
from services import document_service

TEST_CATEGORY = "docs_check"
TEST_FILE_NAME = "docs_check_demo.txt"

DEMO_TEXT = """RAG（检索增强生成）是把外部知识库接进大模型的标准做法。

第一步是文档解析与切分：PDF、Office 这类文档先经 MinerU 在线解析成 Markdown，
再用递归字符切分器切成一个个 chunk，chunk 太大召回不准、太小上下文不完整。

第二步是向量化与写入向量库：embedding 模型把每个 chunk 变成向量，
连同原文与元数据一起 upsert 进 Milvus 的 docs 集合，主键用文档编号加序号。

第三步是检索与重排：用户提问先向量化，在 Milvus 里做 COSINE 相似度检索，
再用重排模型按相关性打分，只保留分数达标的片段拼进提示词，让模型基于资料作答。
"""


def _check(condition: bool, message: str) -> None:
    if condition:
        print(f"  [OK] {message}")
    else:
        raise AssertionError(f"  [FAIL] {message}")


async def _fake_upload(data: bytes) -> AsyncIterator[bytes]:
    for start in range(0, len(data), 4096):
        yield data[start : start + 4096]


async def _upload_once(
    text: str, *, file_name: str = TEST_FILE_NAME, category: str = TEST_CATEGORY
) -> dict:
    plan = document_service.plan_upload(file_name=file_name, category=category)
    size = await document_service.save_upload(_fake_upload(text.encode("utf-8")), plan)

    final: dict | None = None
    async for event in document_service.ingest(plan, file_size=size):
        if event["type"] == "progress":
            print(f"    进度 {event['step']:<5} {event['status']:<7} {event['message']}")
        else:
            final = event

    assert final is not None and final["type"] == "done", f"没有收到 done 事件：{final}"
    return final


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s:%(lineno)d:%(message)s"
    )

    await document_service.ensure_backend()

    target_dir = Path("assets") / TEST_CATEGORY
    source = (target_dir / TEST_FILE_NAME).as_posix()

    print("\n== 1) plan_upload：校验与落盘路径 ==")
    plan = document_service.plan_upload(file_name=TEST_FILE_NAME, category=TEST_CATEGORY)
    _check(plan.source == "assets/docs_check/docs_check_demo.txt", f"分类目录路径：{plan.source}")
    _check(
        plan.temp_path.name.startswith(".") and plan.temp_path.suffix == ".txt",
        f"临时文件点号开头且保留扩展名：{plan.temp_path.name}",
    )
    root_plan = document_service.plan_upload(file_name="a.txt", category="")
    _check(root_plan.source == "assets/a.txt", "不填分类 → 落在 assets 根目录")

    bad_cases = [
        ("a.exe", TEST_CATEGORY, "不支持的扩展名"),
        ("", TEST_CATEGORY, "空文件名"),
        ("..\\b.txt", TEST_CATEGORY, "文件名含路径分隔符"),
        ("a.txt", "../etc", "分类含路径分隔符"),
        ("a.txt", "x" * 40, "分类名超长"),
        ("a.txt", "CON", "分类名是系统保留名"),
    ]
    for bad_name, bad_category, hint in bad_cases:
        try:
            document_service.plan_upload(file_name=bad_name, category=bad_category)
        except document_service.UploadError as exc:
            _check(True, f"{hint} → 被拒绝：{exc}")
        else:
            _check(False, f"{hint} → 竟然通过了校验")

    print("\n== 2) save_upload：体积限流与空文件 ==")
    original_limit = document_service.MAX_UPLOAD_BYTES
    document_service.MAX_UPLOAD_BYTES = 1024
    try:
        big_plan = document_service.plan_upload(file_name="big.txt", category=TEST_CATEGORY)
        try:
            await document_service.save_upload(_fake_upload(b"x" * 2048), big_plan)
        except document_service.UploadTooLarge as exc:
            _check(not big_plan.temp_path.exists(), f"超限被中止且临时文件已清理：{exc}")
        else:
            _check(False, "超过上限竟然写盘成功")
    finally:
        document_service.MAX_UPLOAD_BYTES = original_limit

    empty_plan = document_service.plan_upload(file_name="empty.txt", category=TEST_CATEGORY)
    try:
        await document_service.save_upload(_fake_upload(b""), empty_plan)
    except document_service.UploadError as exc:
        _check(not empty_plan.temp_path.exists(), f"空文件被拒绝：{exc}")
    else:
        _check(False, "空文件竟然写盘成功")

    print("\n== 3) 首次入库 ==")
    first = await _upload_once(DEMO_TEXT)
    first_doc = first["document"]
    _check(first_doc["status"] == doc_table.STATUS_READY, f"台账状态：{first_doc['status']}")
    _check(first_doc["category"] == TEST_CATEGORY, "台账分类正确")
    _check(first["chunk_count"] > 1, f"切分出的 chunk 数：{first['chunk_count']}")
    _check(
        await milvus.count_chunks(source) == first["chunk_count"],
        "Milvus 里的 chunk 数与台账一致",
    )
    _check((target_dir / TEST_FILE_NAME).is_file(), "正式文件已落到 assets/docs_check/")
    _check(not list(target_dir.glob(".*")), "目录里没有残留的临时文件")

    rows = await milvus.get_client().query(
        collection_name=milvus.collection_name,
        filter=f'source == "{source}"',
        output_fields=["id", "text", "source", "metadata"],
        limit=1,
    )
    _check(bool(rows), "能按 source 查到 chunk")
    if rows:
        _check(rows[0]["id"].startswith(first_doc["id"]), f"主键带文档编号前缀：{rows[0]['id']}")
        meta = rows[0]["metadata"]
        _check(
            meta.get("doc_id") == first_doc["id"] and meta.get("category") == TEST_CATEGORY,
            f"chunk metadata 带上了文档信息：{meta}",
        )

    print("\n== 4) 覆盖上传（同分类 + 同名） ==")
    second = await _upload_once(DEMO_TEXT + "\n\n（第二版）这里再多加一段话，用来看覆盖上传的效果。\n")
    _check(second["document"]["id"] == first_doc["id"], "覆盖上传复用同一个文档编号")
    _check(
        await milvus.count_chunks(source) == second["chunk_count"],
        f"旧 chunk 已删、数量不累加：{second['chunk_count']}",
    )
    docs = await document_service.list_documents()
    _check(
        sum(1 for d in docs if d["source"] == source) == 1,
        "台账里该文档只有一行",
    )
    _check(
        any(c == TEST_CATEGORY for c in await document_service.list_categories()),
        f"分类候选里能看到 {TEST_CATEGORY}",
    )

    print("\n== 5) 失败路径：临时文件不存在 ==")
    fail_plan = document_service.plan_upload(file_name="broken.txt", category=TEST_CATEGORY)
    try:
        async for _ in document_service.ingest(fail_plan, file_size=10):
            pass
    except Exception as exc:
        _check(True, f"入库按预期抛错：{type(exc).__name__}")
    else:
        _check(False, "临时文件不存在，入库竟然成功了")

    failed_row = await doc_table.find_by_source(fail_plan.source)
    _check(
        failed_row is not None and failed_row["status"] == doc_table.STATUS_FAILED,
        "台账已标记为 failed",
    )
    if failed_row:
        await doc_table.delete_document(failed_row["id"])

    print("\n== 6) 删除文档 ==")
    result = await document_service.delete_document(first_doc["id"])
    _check(
        result["deleted_chunks"] == second["chunk_count"],
        f"删除的 chunk 数与台账一致：{result}",
    )
    _check(await milvus.count_chunks(source) == 0, "Milvus 里该 source 已无 chunk")
    _check(await doc_table.get_document(first_doc["id"]) is None, "台账行已删除")
    _check(
        (await document_service.delete_document(first_doc["id"]))["deleted_chunks"] == 0,
        "重复删除是幂等的",
    )
    _check(
        (target_dir / TEST_FILE_NAME).is_file(),
        "磁盘上的原始文件刻意保留（删除只清向量库与台账）",
    )

    print("\n== 7) 清理测试目录 ==")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    _check(not target_dir.exists(), f"已删除 {target_dir}")

    await milvus.aclose()
    await connection.close()

    print("\n全部通过 ✅")


if __name__ == "__main__":
    asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)

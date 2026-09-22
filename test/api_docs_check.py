import asyncio
import json
import shutil
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"
TEST_CATEGORY = "api_check"
TEST_FILE_NAME = "api_check_demo.txt"
DEMO_TEXT = """这是接口端到端测试用的文本，用来验证上传 → 入库 → 列表 → 删除这条链路。

接口会把它解析、切分、向量化，再写入 Milvus 的 docs 集合；
台账信息（文件名、分类、分块数、状态）则写进 PostgreSQL 的 documents 表。

删除时先按 source 清掉该文档的全部 chunk，再删掉台账行。""" * 3


def _check(condition: bool, message: str) -> None:
    if condition:
        print(f"  [OK] {message}")
    else:
        raise AssertionError(f"  [FAIL] {message}")


async def main() -> None:
    target_dir = Path("assets") / TEST_CATEGORY

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            await client.get(f"{BASE_URL}/api/sessions")
        except httpx.ConnectError as exc:
            print(f"连不上后端（{BASE_URL}）：请先启动 backend_api.main")
            raise SystemExit(1) from exc

        print("\n== 1) 前置校验：不支持的扩展名应返回 400 ==")
        resp = await client.post(
            f"{BASE_URL}/api/documents/upload",
            files={"file": ("bad.exe", b"MZ...", "application/octet-stream")},
            data={"file_name": "bad.exe", "category": TEST_CATEGORY},
        )
        _check(resp.status_code == 400, f"HTTP {resp.status_code}")
        print(f"      detail = {resp.json().get('detail')}")

        print("\n== 2) 上传入库（multipart + SSE 进度流）==")
        doc: dict | None = None
        async with client.stream(
            "POST",
            f"{BASE_URL}/api/documents/upload",
            files={"file": (TEST_FILE_NAME, DEMO_TEXT.encode("utf-8"), "text/plain")},
            data={"file_name": TEST_FILE_NAME, "category": TEST_CATEGORY},
        ) as resp:
            _check(resp.status_code == 200, f"HTTP {resp.status_code}")
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:].strip())
                if event["type"] == "progress":
                    print(
                        f"     进度 {event['step']:<5} {event['status']:<7} "
                        f"{event.get('percent', '-'):>3}%  {event['message']}"
                    )
                elif event["type"] == "done":
                    doc = event["document"]
                    print(f"     done: chunk_count={event['chunk_count']}")
                elif event["type"] == "error":
                    raise AssertionError(f"入库失败：{event['message']}")

        _check(doc is not None, "收到了 done 事件")
        assert doc is not None
        _check(doc["status"] == "ready", f"台账状态：{doc['status']}")
        _check(doc["chunk_count"] > 0, f"分块数：{doc['chunk_count']}")

        print("\n== 3) 文档列表 ==")
        resp = await client.get(f"{BASE_URL}/api/documents")
        _check(resp.status_code == 200, f"HTTP {resp.status_code}")
        documents = resp.json()["documents"]
        mine = [item for item in documents if item["id"] == doc["id"]]
        _check(len(mine) == 1, f"列表里能找到刚上传的文档（共 {len(documents)} 篇）")
        print(f"      {mine[0]['file_name']} | {mine[0]['category']} | "
              f"{mine[0]['chunk_count']} chunk | {mine[0]['source']}")

        print("\n== 4) 分类候选 ==")
        resp = await client.get(f"{BASE_URL}/api/documents/categories")
        categories = resp.json()["categories"]
        _check(TEST_CATEGORY in categories, f"包含 {TEST_CATEGORY}：{categories}")

        print("\n== 5) 删除文档 ==")
        resp = await client.delete(f"{BASE_URL}/api/documents/{doc['id']}")
        _check(resp.status_code == 200, f"HTTP {resp.status_code}")
        deleted = resp.json()
        _check(deleted["deleted_chunks"] == doc["chunk_count"], f"清掉的 chunk 数：{deleted}")

        resp = await client.get(f"{BASE_URL}/api/documents")
        left = [item for item in resp.json()["documents"] if item["id"] == doc["id"]]
        _check(len(left) == 0, "列表里已经没有这篇文档")

        resp = await client.delete(f"{BASE_URL}/api/documents/{doc['id']}")
        _check(resp.status_code == 200 and resp.json()["deleted_chunks"] == 0, "重复删除幂等")

    print("\n== 6) 清理测试目录 ==")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    _check(not target_dir.exists(), f"已删除 {target_dir}")

    print("\n全部通过 ✅")


if __name__ == "__main__":
    asyncio.run(main())

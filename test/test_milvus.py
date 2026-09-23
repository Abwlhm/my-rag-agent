"""临时修复验证脚本：release/load 集合触发段重载，验证 503 是否消失（用完即删）。"""

import asyncio

from db.milvus.client import Embedding_DIM, collection_name, get_client, retrieve


async def main() -> None:
    client = get_client()

    # release + load：让 QueryNode 重新加载 sealed 段（符号链接已修好路径解析）
    await client.release_collection(collection_name=collection_name)
    print("released, 开始 load ...")
    await client.load_collection(collection_name=collection_name)

    # 轮询加载状态：预期从 Loading 50% 卡死变为 Loaded 100%
    for _ in range(30):
        state = await client.get_load_state(collection_name=collection_name)
        print("load_state:", state)
        if "Loaded" in str(state):
            break
        await asyncio.sleep(2)

    # 混合检索冒烟：与故障时同一调用路径，成功返回即证明 503 已消除
    try:
        hits = await retrieve([0.0] * Embedding_DIM, "套餐怎么收费", limit=3)
        print("retrieve OK, hits:", len(hits))
    except Exception as exc:
        print("retrieve FAILED:", type(exc).__name__, str(exc)[:300])


if __name__ == "__main__":
    asyncio.run(main())

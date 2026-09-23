"""client.py —— Milvus 向量库客户端：库/集合准备 + chunk 增删查与混合检索

混合检索 = 稠密向量（客户端写入）+ BM25 稀疏向量（BM25 Function 从 text 自动生成，
客户端不写入也不能读出），检索时双路 AnnSearchRequest + RRFRanker 融合，见 retrieve()。
配置来自 .env：MILVUS_URI / MILVUS_DB_NAME / MILVUS_COLLECTION_NAME / MILVUS_EMBEDDING_DIM。
本模块只依赖 pymilvus，绝不 import services / agent_core / rag_component（依赖方向单向）。
"""

import asyncio
import logging
import os

from dotenv import load_dotenv
from pymilvus import (
    AnnSearchRequest,
    AsyncMilvusClient,
    DataType,
    Function,
    FunctionType,
    RRFRanker,
)

load_dotenv(override=True)

# ---- Milvus 连接与集合配置 ----
MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
db_name = os.getenv("MILVUS_DB_NAME", "rag_demo")
collection_name = os.getenv("MILVUS_COLLECTION_NAME", "docs")
Embedding_DIM = int(os.getenv("MILVUS_EMBEDDING_DIM", "4096"))  # 建集合时校验向量维度用

logger = logging.getLogger(__name__)

# ---- 异步 Milvus 客户端（模块级单例）----
# create_rag_db() 会**替换**这个实例，禁止 `from db.milvus.client import client` 按值导入，
# 一律用 get_client() 现取。构造时不建连，首次请求才懒加载连接。
_client = AsyncMilvusClient(uri=MILVUS_URI, db_name=db_name)


def get_client() -> AsyncMilvusClient:
    """取当前生效的 Milvus 客户端（同步函数）。"""
    return _client


# 集合就绪标记（懒加载）：ensure_docs_collection() 成功一次后不再发 RPC；失败保持 False 可重试
_ready = False
_ready_lock = asyncio.Lock()


async def create_rag_db():
    """建库（不存在时）+ 切库。

    注意：它会**换掉模块级单例 _client**（带 db_name 的客户端无法创建不存在的库），
    调用后必须把 _ready 复位，让下次 ensure_docs_collection() 重新检查集合。
    """
    global _client, _ready
    _client = AsyncMilvusClient(uri=MILVUS_URI)
    milvus_client = get_client()  # 刚换上的那个实例

    existed_databases = await milvus_client.list_databases()

    logger.info("================== Existed_databases ==================")
    for db in existed_databases:
        logger.info(db)
    logger.info("================== Existed_databases ==================")

    if db_name not in existed_databases:
        await milvus_client.create_database(db_name=db_name)

    logger.info("================== Existed_databases ==================")
    for db in await milvus_client.list_databases():
        logger.info(db)
    logger.info("================== Existed_databases ==================")

    await milvus_client.use_database(db_name=db_name)

    # 客户端换了：集合就绪标记失效，下次 ensure 会重新查一次
    _ready = False


async def _create_docs_collection(milvus_client: AsyncMilvusClient):
    """创建 docs 集合：显式 schema（双向量混合检索）+ 两路索引，建完自动 load。

    字段：id（VARCHAR 主键="<doc_id>_<chunk序号>"）、text（BM25 Function 输入）、
    dense_vector（客户端写入，COSINE）、sparse_vector（BM25 Function 输出，服务端生成）；
    source/metadata 走动态字段。不能用快速建集合：它只支持单向量字段。
    """
    schema = milvus_client.create_schema(auto_id=False, enable_dynamic_field=True)
    # 主键用字符串 f"{doc_id}_{序号}"：跨文档/多次上传天然不撞主键
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=512)
    # 原文字段：BM25 分词来源。language_identifier 自动检测语言并路由分析器。
    # 注意：analyzers 键名必须与检测引擎输出的语言名精确匹配（lingua 中文="Chinese"，
    #       whatlang 则="Mandarin"）；不匹配会静默落进 default 分词器（中文召回变差但不报错）。
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,
        analyzer_params={
            "tokenizer": {
                "type": "language_identifier",
                "identifier": "lingua",  # lingua 对短文本/查询更准；键名与它配套
                "analyzers": {
                    "Chinese": {"tokenizer": "jieba"},  # 中文 → jieba 分词
                    "English": {"type": "english"},  # 英文 → 内置 english
                    "default": {"tokenizer": "jieba"},  # 兜底：jieba 中英都能切
                },
            },
        },
    )
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=Embedding_DIM)
    # 稀疏向量是 Function 输出字段：写入时服务端自动算，客户端不要传值
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

    # BM25 Function：写入时 text → sparse_vector；检索时查询文本同样转查询稀疏向量
    schema.add_function(
        Function(
            name="text_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
        )
    )

    # 两路索引；稀疏路必须 metric_type="BM25" 才启用全文检索打分
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE"
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
    )

    # 传 index_params 后 create_collection 会自动建索引并 load
    await milvus_client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


async def ensure_docs_collection():
    """幂等：保证库与 docs 集合存在并切库（不删已有集合，是上传链路的入口）。

    动态字段可以直接写进 filter 表达式（source == "assets/a.pdf"），无需 $meta["source"]。
    """
    global _ready
    if _ready:
        return

    async with _ready_lock:
        if _ready:
            return

        milvus_client = get_client()

        # 1) 库不存在就先建（用临时客户端，见 create_rag_db 的说明）
        existed_databases = await milvus_client.list_databases()
        if db_name not in existed_databases:
            temp_client = AsyncMilvusClient(uri=MILVUS_URI)
            try:
                await temp_client.create_database(db_name=db_name)
            finally:
                await temp_client.close()

        # 2) 切库（每个客户端都要切一次）
        await milvus_client.use_database(db_name=db_name)

        # 3) 集合不存在才建
        if not await milvus_client.has_collection(collection_name=collection_name):
            await _create_docs_collection(milvus_client)

        _ready = True


async def recreate_docs_collec():
    """删掉 docs 集合再新建：仅供手动重置数据使用（test/reset_docs_db.py），上传链路禁用。"""
    global _ready
    milvus_client = get_client()
    if await milvus_client.has_collection(collection_name=collection_name):
        await milvus_client.drop_collection(collection_name=collection_name)

    await _create_docs_collection(milvus_client)
    _ready = True
    await _list_collections()


async def _list_collections():
    """打印当前库中所有集合名。"""
    milvus_client = get_client()
    existed_collections = await milvus_client.list_collections()
    logger.info("================== Existed_collections ==================")
    for collec in existed_collections:
        logger.info(collec)
    logger.info("================== Existed_collections ==================")


def _filter_string_literal(value: str) -> str:
    """把 Python 字符串转成 Milvus filter 表达式的字符串字面量（转义反斜杠与双引号，兜底）。"""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


async def upsertToMilvus(data: list[dict]):
    """把 chunk 数据按主键 upsert 进 docs 集合（只追加/覆盖，不重建集合），最后 flush。

    data 里不要放 sparse_vector：BM25 Function 输出字段，服务端按 text 自动重算。
    """
    await ensure_docs_collection()
    milvus_client = get_client()
    upsert_res = await milvus_client.upsert(
        collection_name=collection_name,
        data=data,
    )
    logger.info(f"Upsert result : {upsert_res}")
    logger.info("-" * 50)
    # flush 让刚写入的数据落盘，否则紧接着检索可能查不到
    await milvus_client.flush(collection_name=collection_name)

    stats = await milvus_client.get_collection_stats(collection_name=collection_name)
    logger.info(f"{collection_name} stats: {stats}")
    logger.info("-" * 50)


async def delete_chunks_by_source(source: str) -> int:
    """按 source 删除某文档的全部 chunk（覆盖上传 / 删除文档时调用），返回删除条数（不存在返回 0）。"""
    await ensure_docs_collection()
    milvus_client = get_client()
    res = await milvus_client.delete(
        collection_name=collection_name,
        filter=f"source == {_filter_string_literal(source)}",
    )
    # 删除也要 flush：否则紧接着的 count/query 可能仍看到快照里的旧数据
    await milvus_client.flush(collection_name=collection_name)
    return int(res.get("delete_count", 0)) if isinstance(res, dict) else 0


async def count_chunks(source: str | None = None) -> int:
    """统计 chunk 数：source=None 统计整个集合，否则只统计某个文档。"""
    await ensure_docs_collection()
    milvus_client = get_client()
    expr = "" if source is None else f"source == {_filter_string_literal(source)}"
    result = await milvus_client.query(
        collection_name=collection_name,
        filter=expr,
        output_fields=["count(*)"],
    )
    if not result:
        return 0
    return int(result[0].get("count(*)", 0))


async def aclose():
    """关闭 Milvus 客户端（Web 端经 services.document_service.aclose() 转发调用）。"""
    await get_client().close()


async def retrieve(query_vector: list[float], query_text: str, limit: int = 10) -> list[dict]:
    """混合检索：稠密向量 + BM25 稀疏向量，RRF 融合后返回 top-limit。

    稠密路传 query 的 embedding（COSINE）；稀疏路**直接传查询文本**（BM25 Function
    的输出字段由服务端分词打分，客户端不算 BM25）。
    返回项形如 {"id", "distance", "entity": {...}}；distance 是 RRF 融合分，
    只用于排序、不可跨查询比较。
    """
    milvus_client = get_client()

    dense_request = AnnSearchRequest(
        data=[query_vector],
        anns_field="dense_vector",
        param={"metric_type": "COSINE"},
        limit=30,
    )
    sparse_request = AnnSearchRequest(
        data=[query_text],
        anns_field="sparse_vector",
        param={"metric_type": "BM25"},
        limit=30,
    )

    results = await milvus_client.hybrid_search(
        collection_name=collection_name,
        reqs=[dense_request, sparse_request],
        ranker=RRFRanker(k=60),
        limit=limit,
        output_fields=["id", "text", "source", "metadata"],
    )
    return results[0]

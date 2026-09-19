import asyncio
import logging

from rag_component.loader import load_txt
from rag_component.splitter import split_docs
from rag_component.embedding import embedding_model
from rag_component.milvus import create_rag_db, upsertToMilvus, retrieve

logger = logging.getLogger(__name__)


def _print_retrieve_chunks(retrieve_chunks: list[dict]):
    logger.debug("================== 检索结果 ==================")
    for i, chunk in enumerate(retrieve_chunks, start=1):
        id = chunk["entity"]["id"]
        score = chunk["distance"]
        text = chunk["entity"]["text"]
        source = chunk["entity"].get("source", "unknown")

        logger.debug(
            f"Trunk[{i}] | id={id} | score={score:.5f} | source={source} \n{text}\n"
        )

    logger.debug("================== 检索结果 ==================")


async def upload_new_file(file_path: str):
    documents = load_txt(file_path)

    chunks = split_docs(documents)
    logger.info(f"{file_path}文档共切分为{len(chunks)}个chunk")

    texts = [chunk.page_content for chunk in chunks]
    vectors = await embedding_model.aembed_documents(texts)
    data = [
        {
            "id": i,
            "vector": vectors[i],
            "text": texts[i],
            "source": file_path,
            "metadata": chunks[i].metadata,
        }
        for i in range(len(texts))
    ]

    await upsertToMilvus(data)


async def retrieve_from_vector_db(query: str, limit: int = 10) -> list[dict]:
    query_vector = await embedding_model.aembed_query(query)
    retrieve_chunks = await retrieve(query_vector, limit=limit)
    _print_retrieve_chunks(retrieve_chunks)
    return retrieve_chunks


async def main():
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s:%(lineno)d:%(message)s"
    )
    await create_rag_db()
    await upload_new_file("assets/demo_txt_1.txt")


if __name__ == "__main__":
    asyncio.run(main())

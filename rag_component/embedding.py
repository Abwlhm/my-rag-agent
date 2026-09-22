import os
from dotenv import load_dotenv
from langchain.embeddings import init_embeddings

load_dotenv(override=True)

embedding_model = init_embeddings(
    base_url=os.getenv("SiliconFlow_BASE_URL"),
    api_key=os.getenv("SiliconFlow_API_KEY"),
    model=os.getenv("SiliconFlow_Embedding_MODEL"),
    provider="openai",
)


if __name__ == "__main__":
    
    texts = ["你好", "hello"]

    for text in texts:
        embedded_vector = embedding_model.embed_query(text)
        print(text, "长度：", len(embedded_vector), "\n向量：", embedded_vector[:10])

    embedded_vectors = embedding_model.embed_documents(texts)
    for i, vec in enumerate(embedded_vectors):
        print(texts[i], "长度：", len(vec), "\n向量：", vec[:10])

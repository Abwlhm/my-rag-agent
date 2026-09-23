from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import Iterable

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50,
    separators=[
        "\n\n",
        "\n",
        "。",
        "；",
        ";",
        "！",
        "!",
        "？",
        "?",
        "，",
        ",",
        " ",
        "",
    ],
    add_start_index=True,
)


def split_docs(documents: Iterable[Document]) -> list[Document]:
    return text_splitter.split_documents(documents)


if __name__ == "__main__":
    text = "aaaaaaaaaa\n\nbbbbbbbbbbbbbb\n\ncccccccccccccccccc"

    paragraphs = text_splitter.split_text(text)

    for i, chunk in enumerate(paragraphs):
        print("-" * 50)
        print(f"块{i + 1}：长度: {len(chunk)}")
        print(chunk)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import Iterable

"""
# 方法1：
split_text(self, text: str) -> list[str]:
传入的参数类型：文本内容（或字符串），返回值类型：字符串列表
此方法是抽象方法，具体的实现细节由子类来决定

# 方法2：
create_documents(self, texts: list[str],...) -> list[Document]:
传入的参数类型：字符串列表，返回值类型：Document对象列表
此方法的底层调用了split_text()，即将参数中的每一个字符串都传入split_text()中执行，得到的字符串列表中，将字符串封装为Document对象，就构成了list[Document]。

# 方法3：
split_documents(self, documents: Iterable[Document]) -> list[Document]:
传入的参数类型：Document对象列表，返回值类型：Document对象列表
此方法的底层调用了create_documents()，将参数中的每一个Document对象，提取其page_content字段，则构成了字符串列表，然后调用方法2即可。
"""

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
    text = "aaaaaaaaaaaaaaa\n\nbbbbbbbbbbbbbbbbbbbb\n\nccccccccccccccccccccccc"

    paragraphs = text_splitter.split_text(text)

    for i, chunk in enumerate(paragraphs):
        print("-" * 50)
        print(f"块{i + 1}：长度: {len(chunk)}")
        print(chunk)

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# TODO: use MinerU

def load_txt(file_path: str, encoding: str = "utf-8") -> list[Document]:
    loader = TextLoader(file_path=file_path, encoding=encoding)
    return loader.load()


def load_csv(file_path: str) -> list[Document]:
    loader = CSVLoader(file_path=file_path)
    return loader.load()


def load_json(
    file_path: str, jq_schema: str = ".", text_content: bool = True
) -> list[Document]:
    loader = JSONLoader(
        file_path=file_path, jq_schema=jq_schema, text_content=text_content
    )
    return loader.load()


def load_pdf(file_path: str, extraction_mode: str = "plain") -> list[Document]:
    loader = PyPDFLoader(file_path=file_path, extraction_mode=extraction_mode)
    return loader.load()


def load_word(file_path: str) -> list[Document]:
    loader = UnstructuredWordDocumentLoader(file_path=file_path)
    return loader.load()


def load_markdown(file_path: str) -> list[Document]:
    loader = UnstructuredMarkdownLoader(file_path=file_path)
    return loader.load()


if __name__ == "__main__":
    documents = load_txt("assets/demo_txt_1.txt")
    print("document 数量：", len(documents))
    print(documents)

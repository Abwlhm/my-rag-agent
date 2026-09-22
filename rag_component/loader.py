import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_mineru import MinerULoader

load_dotenv(override=True)


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


def load_markdown(file_path: str) -> list[Document]:
    loader = UnstructuredMarkdownLoader(file_path=file_path)
    return loader.load()


LOCAL_LOADERS: dict[str, Callable[..., list[Document]]] = {
    ".txt": load_txt,
    ".csv": load_csv,
    ".json": load_json,
    ".md": load_markdown,
    ".markdown": load_markdown,
}

MINERU_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".xls",
        ".xlsx",
        ".png",
        ".jpg",
        ".jpeg",
        ".jp2",
        ".webp",
        ".gif",
        ".bmp",
        ".html",
        ".htm",
    }
)

token = os.getenv("MinerU_API_KEY", "")


async def aload_document(
    file_path: str | Path,
    *,
    mode: Literal["precision", "flash"] = "precision",
    ocr: bool = False,
    formula: bool = True,
    table: bool = True,
    language: str = "ch",
    pages: str | None = None,
    timeout: int = 1200,
    **local_kwargs: Any,
) -> list[Document]:
    source = str(file_path)
    is_remote = source.startswith(("http://", "https://"))

    clean_source = source.split("?")[0].split("#")[0] if is_remote else source
    suffix = os.path.splitext(clean_source)[1].lower()

    if suffix not in MINERU_EXTENSIONS and suffix not in LOCAL_LOADERS:
        raise ValueError(f"不支持的文件类型：{suffix or '无扩展名'}（{source}）")
    if is_remote and suffix in LOCAL_LOADERS:
        raise ValueError(f"远程 URL 只能交给 MinerU 解析，但 MinerU 不支持 {suffix} 类型（{source}）")
    if not is_remote and not Path(source).is_file():
        raise FileNotFoundError(f"文件不存在或不是常规文件：{source}")

    if suffix in MINERU_EXTENSIONS:
        if local_kwargs:
            raise ValueError(
                f"未知参数 {list(local_kwargs)}：MinerU 一侧只接受 "
                f"mode / ocr / formula / table / language / pages / timeout（{source}）"
            )

        if mode == "precision" and not token:
            raise ValueError("mode='precision' 需要 token，请在项目根目录的 .env 中配置 MinerU_API_KEY")

        def _load_with_mineru() -> list[Document]:
            loader = MinerULoader(
                source=source,
                mode=mode,
                token=token,
                language=language,
                pages=pages,
                timeout=timeout,
                ocr=ocr,
                formula=formula,
                table=table,
            )
            return loader.load()

        return await asyncio.to_thread(_load_with_mineru)

    return LOCAL_LOADERS[suffix](file_path, **local_kwargs)


if __name__ == "__main__":
    documents = load_txt("assets/demo_txt_1.txt")
    print("document 数量：", len(documents))
    print(documents)

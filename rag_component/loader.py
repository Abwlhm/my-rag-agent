"""文档加载模块：按扩展名分发到 MinerU 在线解析或本地 loader。

文档类（pdf/office/图片/html）→ MinerU 在线解析；纯文本类（txt/csv/json/md）→ 本地 loader。
"""

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

# 与项目其它模块保持一致：加载 .env，且后加载的覆盖进程里已有的同名变量
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


# 纯文本类扩展名 → 本地 loader（MinerU 不支持这些类型，继续本地读）
LOCAL_LOADERS: dict[str, Callable[..., list[Document]]] = {
    ".txt": load_txt,
    ".csv": load_csv,
    ".json": load_json,
    ".md": load_markdown,
    ".markdown": load_markdown,
}

MINERU_EXTENSIONS: frozenset[str] = frozenset(
    {
        # 文档
        ".pdf",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".xls",
        ".xlsx",
        # 图片
        ".png",
        ".jpg",
        ".jpeg",
        ".jp2",
        ".webp",
        ".gif",
        ".bmp",
        # 网页（langchain-mineru 会自动切到 MinerU-HTML 模型）
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
    """
    统一加载入口：按扩展名分发到 MinerU 在线解析或本地 loader。

    参数命名约定（重要）：显式列出的参数都属于 MinerU，其余具名参数一律透传给本地 loader
    （显式参数名刻意与 MinerULoader 同名，避免被误判进 local_kwargs）。
    page_content 是 MinerU 输出的 Markdown（不保留页码等结构化元数据）。
    """
    source = str(file_path)
    is_remote = source.startswith(("http://", "https://"))

    # URL 先剥掉查询串 / 锚点再取扩展名；用 splitext 兼容 Windows 反斜杠路径
    clean_source = source.split("?")[0].split("#")[0] if is_remote else source
    suffix = os.path.splitext(clean_source)[1].lower()

    # 前置校验统一放在分发之前
    if suffix not in MINERU_EXTENSIONS and suffix not in LOCAL_LOADERS:
        raise ValueError(f"不支持的文件类型：{suffix or '无扩展名'}（{source}）")
    if is_remote and suffix in LOCAL_LOADERS:
        raise ValueError(f"远程 URL 只能交给 MinerU 解析，但 MinerU 不支持 {suffix} 类型（{source}）")
    if not is_remote and not Path(source).is_file():
        raise FileNotFoundError(f"文件不存在或不是常规文件：{source}")

    # ---- 文档类：MinerU 在线解析
    if suffix in MINERU_EXTENSIONS:
        if local_kwargs:
            raise ValueError(
                f"未知参数 {list(local_kwargs)}：MinerU 一侧只接受 "
                f"mode / ocr / formula / table / language / pages / timeout（{source}）"
            )

        # langchain-mineru 只认 MINERU_TOKEN 环境变量，必须显式传 token
        if mode == "precision" and not token:
            raise ValueError("mode='precision' 需要 token，请在项目根目录的 .env 中配置 MinerU_API_KEY")

        def _load_with_mineru() -> list[Document]:
            """在线解析是同步阻塞的（SDK 内部轮询），放在线程里跑。"""
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
                # split_pages 必须默认 False：True 会逐页调用 API，浪费额度
            )
            return loader.load()

        return await asyncio.to_thread(_load_with_mineru)

    # ---- 纯文本类：本地 loader（同步、毫秒级，无需 offload）
    return LOCAL_LOADERS[suffix](file_path, **local_kwargs)


if __name__ == "__main__":
    documents = load_txt("assets/demo_txt_1.txt")
    print("document 数量：", len(documents))
    print(documents)

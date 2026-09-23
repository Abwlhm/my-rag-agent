import os
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.constants import TAG_NOSTREAM
from pydantic import BaseModel, Field

load_dotenv(override=True)

llm = init_chat_model(
    model=os.getenv("DeepSeek_MODEL"),
    model_provider="deepseek",
    api_key=os.getenv("DeepSeek_API_KEY"),
    base_url=os.getenv("DeepSeek_BASE_URL"),
    temperature=0.7,
    reasoning_effort="low",
    configurable_fields=("extra_body",),
    model_kwargs={"tools": []},
    extra_body={
        "thinking": {"type": "disabled"}
    },  # 关闭 DeepSeek 深度思考，让输出更简单直接
)


class RouteDecision(BaseModel):
    """路由分类结果（交给模型以结构化输出的形式填写）"""

    need_retrieve: bool = Field(
        description=(
            "true = 必须查询企业知识库才能回答（套餐、资费、政策、规则等业务事实）；"
            "false = 不需要查库（寒暄、闲聊、通用常识、写作/编程等）"
        )
    )


# route_llm：结构化输出约束成 RouteDecision；TAG_NOSTREAM 防止分类 token 混进回答流。
route_llm = llm.with_structured_output(RouteDecision).with_config(
    {
        "tags": [TAG_NOSTREAM],
        "configurable": {"extra_body": {"thinking": {"type": "disabled"}}},
    }
)

# condense_llm：幕后改写调用，同样 TAG_NOSTREAM 不推流 + 关思考（省钱提速）
condense_llm = llm.with_config(
    {
        "tags": [TAG_NOSTREAM],
        "configurable": {"extra_body": {"thinking": {"type": "disabled"}}},
    }
)

_jev_client: Any = None


def get_jev_client() -> Any:
    global _jev_client
    if _jev_client is None:
        from typesafe_sdk import AsyncTypeSafeClient

        # 自动读 TYPESAFE_API_KEY，默认 model="jev-latest"
        _jev_client = AsyncTypeSafeClient()
    return _jev_client

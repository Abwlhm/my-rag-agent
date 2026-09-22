import os

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
    },
)


class RouteDecision(BaseModel):

    need_retrieve: bool = Field(
        description=(
            "true = 必须查询企业知识库才能回答（套餐、资费、政策、规则等业务事实）；"
            "false = 不需要查库（寒暄、闲聊、通用常识、写作/编程等）"
        )
    )


route_llm = llm.with_structured_output(RouteDecision).with_config(
    {
        "tags": [TAG_NOSTREAM],
        "configurable": {"extra_body": {"thinking": {"type": "disabled"}}},
    }
)

condense_llm = llm.with_config(
    {
        "tags": [TAG_NOSTREAM],
        "configurable": {"extra_body": {"thinking": {"type": "disabled"}}},
    }
)

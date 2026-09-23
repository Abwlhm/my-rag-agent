"""constants.py —— 跨层共享常量。

STEP_PERCENT 同时被 rag_component / services / backend_api 使用，放顶层保持依赖单向。
数值属于"前端进度协议"，改动会影响进度条，请勿随意调整。
"""

# 各阶段完成时的总进度百分比
STEP_PERCENT: dict[str, int] = {
    "save": 10,
    "parse": 30,
    "split": 50,
    "embed": 75,
    "write": 95,
    "done": 100,
}

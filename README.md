# 对话 Agent 原型（React + FastAPI + LangChain/LangGraph）

前后端分离的流式对话 Agent（含 RAG 检索链路）原型：

```
浏览器（React 聊天界面）
    │  /api/*（开发期由 Vite 代理转发到后端）
    ▼
FastAPI 接口（backend_api）—— SSE 流式回答
    │
    ▼
Agent（agent_core，LangGraph 图：追问改写 → 路由 → 检索/直答 → 回答）
    │                                   │
    ▼                                   ▼
DeepSeek（LLM）                  PostgreSQL（会话记忆 + 侧栏会话列表）
                                        ▲
                              Milvus + SiliconFlow（检索链路）
```

## 目录结构

- `agent_core/llm_client.py`：LLM / 路由模型的构造（DeepSeek 官方 API）。
- `agent_core/rag_agent.py`：LangGraph 图（condense → route → retrieve/rerank → 回答），
  通过 `persistence` 层提供的 Postgres checkpointer 持久化每个会话（`thread_id`）的记忆。
- `persistence/`：基础设施层——Postgres 连接池与 checkpointer（懒加载单例）、会话列表查询与删除；
  依赖方向单向：`agent_core` → `persistence`（后者不反向 import）。
- `agent_core/chat_agent.py`：Web 层唯一依赖的业务入口
  （流式回答、会话列表、历史消息、删除会话、关闭连接池）。
- `backend_api/main.py`：FastAPI，提供 SSE 流式问答与会话管理接口。
- `frontend/`：React 前端（Vite + TypeScript），界面仿 DeepSeek：
  左侧会话列表（标题暂为 `thread_id`，含更新时间、悬浮"..."删除菜单），
  右侧聊天区（流式打字机 + Markdown 渲染）。
- `rag_component/`：检索组件（加载、切分、向量化、Milvus、重排）。

## 运行前提

- 根目录 `.env` 已配置 `DeepSeek_*`（LLM）与 `POSTGRESQL_DB_URL`（会话记忆）；
  检索链路另需 `SiliconFlow_*`，且本机 Milvus 已启动（`http://127.0.0.1:19530`）。
- 后端依赖：`pip install -r requirements.txt`
- 前端依赖：进入 `frontend/` 后执行 `npm install`（需要 Node.js 18+）。

## 启动步骤（两个终端）

终端 1 —— 启动后端（默认 8000 端口）：

```
python -m uvicorn backend_api.main:app --host 127.0.0.1 --port 8000 --loop asyncio:SelectorEventLoop
```

（Windows 上 psycopg 异步必须使用 SelectorEventLoop；直接 `python -m backend_api.main` 也可以。）

终端 2 —— 启动前端（默认 5173 端口）：

```
cd frontend
npm run dev

cd frontend; if ($?) { npm run dev }
```

浏览器打开 http://127.0.0.1:5173 即可对话。

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat/stream` | 流式问答（SSE）。请求体 `{"query": "...", "thread_id": "..."}`；`thread_id` 省略时后端自动生成 uuid4（等价于无记忆的单轮问答） |
| GET | `/api/sessions` | 列出所有历史会话（直接读 PostgreSQL 的 `checkpoints` 表），按最近活跃倒序，含 `updated_at` |
| GET | `/api/sessions/{thread_id}/messages` | 读取某个会话的聊天记录（不含 system 提示词） |
| DELETE | `/api/sessions/{thread_id}` | 删除某个会话：清空 `checkpoints` / `checkpoint_blobs` / `checkpoint_writes` 三张表中该 thread 的数据；幂等 |

### SSE 事件格式

| type | 含义 |
|---|---|
| `content` | 一段回答文本，字段 `text` |
| `error`   | 出错信息，字段 `message` |
| `done`    | 回答结束 |

## 前端行为说明

- **多轮记忆**：不再由前端传 history，而是后端 checkpointer 按 `thread_id` 读写 PostgreSQL；
  前端只负责生成（`crypto.randomUUID()`）并透传 `thread_id`。
- **会话列表**：直接读 checkpointer 的 `checkpoints` 表，所以测试脚本产生的线程
  （如 `thread_test00`、`demo-thread-001`）也会出现在列表里，不需要时用界面删除即可。
- **删除会话**：鼠标悬停会话 → 点右侧"..." → 删除 → 确认；后端会同时清空数据库中该会话的全部数据。
- **流式期间**：输入框禁用；侧栏的"切换会话 / 新建对话 / 删除"操作被忽略（简化处理）。
- **主题**：侧栏左下角可切换浅色 / 深色（深色仿 DeepSeek 深色界面），选择持久化在
  localStorage，刷新后保持；首屏由 `index.html` 内联脚本提前应用，不会闪白屏。
- **开发期跨域**：Vite 把 `/api` 代理到 `http://127.0.0.1:8000`，后端无需配置 CORS。


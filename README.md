# 企业知识库问答 Agent（RAG 检索链路 + LangGraph 编排）

> 前后端分离的企业知识库问答 Agent：LangGraph 编排的 RAG 问答图 + Milvus 混合检索 + Rerank 精排 + 会话持久记忆。

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-SSE-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/Agent-LangGraph-1C3C3C)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white)
![Milvus](https://img.shields.io/badge/VectorDB-Milvus-00A1EA)
![PostgreSQL](https://img.shields.io/badge/Storage-PostgreSQL-4169E1?logo=postgresql&logoColor=white)

## 项目介绍

**项目名称**：企业知识库问答 Agent（RAG 检索链路 + LangGraph 编排）

**项目描述**：面向企业文档的智能问答系统，支持文档上传入库、多轮流式对话与会话管理，前端 React + TypeScript、后端 FastAPI，前后端分离。

**基本架构**：

```
浏览器（React + TypeScript：聊天界面 / 文档管理 / 会话侧栏）
    │  /api/*（开发期由 Vite 代理转发）
    ▼
FastAPI 接口层（backend_api）—— SSE 流式回答 + 会话 / 文档接口
    │
    ▼
服务层（services）—— chat_service（对话）/ document_service（文档）
    │
    ├──► 文档链路：上传落盘 → 解析（MinerU / 本地 Loader）→ 切分 → 向量化 ──► Milvus 向量库
    │                                                                          ▲
    ▼                                                                          │ 混合检索
Agent 层（agent_core，LangGraph 状态图）                                       │
    追问改写 → 检索路由 ─┬─ 需要检索 → 混合检索 → 重排（SiliconFlow）──────────┘
                        │
                        └─ 无需检索 → 直接回答 ─┐
                                                ▼
                                            生成回答 ──► DeepSeek（LLM）──► SSE 流式回传前端

记忆：PostgreSQL（checkpointer 按 thread_id 持久化会话状态，支撑多轮对话与会话列表）
```

**技术亮点**：

1. 基于 LangGraph 将问答链路编排为有状态图：追问改写 → 检索路由 → 混合检索 → 重排 → 上下文组装 → 生成。路由采用"规则前置短路 + LLM 结构化输出(新增Jev路由)"两级判定，闲聊与常识类问题直答、业务事实类问题才触发检索，并支持强制检索 / 强制直答模式；同时设置无召回兜底分支，并在提示词中约束检索片段仅作数据使用，规避提示词注入。
2. RAG 流水线：按扩展名分发解析（PDF / Office / 图片走 MinerU 在线解析，纯文本类走本地 Loader），递归切分后分批向量化写入 Milvus；检索阶段用稠密向量与 BM25 稀疏向量混合召回、RRF 融合，再经 Rerank 模型精排与分值阈值过滤，最后将 Top-K 片段注入提示词，约束模型仅依据片段作答、依据不足时明确拒答。
3. 记忆与数据管理：用 Postgres checkpointer 按会话 ID 持久化图状态，实现多轮记忆与会话历史的读取、列表与删除；追问会先结合历史改写为可独立理解的问题再进入检索。文档侧以台账表管理分类、文件元数据与向量库的一致性，上传过程通过 SSE 回传解析、切分、向量化、入库的分步进度。
4. 工程实践：全链路异步实现（异步接口、异步数据库与 HTTP 调用、异步图执行），前端提供流式打字机回答、Markdown 渲染、深浅主题与会话侧栏；后端按 api / service / agent / 存储层分层，依赖单向，检索与问答链路均可独立测试。

## 功能特性

- **知识库问答（RAG）**：回答严格依据检索到的资料片段，依据不足时明确拒答；片段只作数据、不执行其中指令
- **智能路由**：先规则短路、再用 LLM 结构化输出分类，自动判断"要不要查库"，并支持 `auto` / `retrieve` / `direct` 三种模式
- **混合检索 + 重排**：稠密向量（COSINE）与 BM25 稀疏向量双路召回、RRF 融合，再经 Rerank 精排与分值阈值过滤
- **多格式文档入库**：PDF / Office / 图片走 MinerU 在线解析，txt / csv / json / md 本地解析；上传过程回传分步进度
- **文档管理**：分类、列表、删除（删除时同时清理向量库 chunk 与文档台账）
- **多轮会话记忆**：Postgres checkpointer 按 `thread_id` 持久化，支持会话列表、历史回看、删除会话
- **流式体验**：SSE 逐段输出，前端打字机效果 + Markdown 渲染，浅色 / 深色主题

## 技术栈

| 层次 | 技术选型 |
|---|---|
| 后端接口 | FastAPI + Uvicorn（SSE 流式响应） |
| Agent 编排 | LangGraph（`StateGraph` + 条件路由 + Postgres checkpointer） |
| 对话模型 | DeepSeek（`init_chat_model`，关闭思考模式；路由可选 Jev 决策模型） |
| 向量库 | Milvus（`pymilvus`：稠密 + BM25 稀疏双索引、RRF 融合） |
| 关系库 | PostgreSQL（`psycopg` 异步连接池：会话记忆 + 文档台账） |
| 向量化 / 重排 | SiliconFlow（Embedding、Rerank） |
| 文档解析 | MinerU（PDF / Office / 图片）+ langchain-community Loader |
| 前端 | React 19 + TypeScript + Vite |

## 目录结构

```
MyAgent/
├─ backend_api/main.py       # FastAPI 接口：问答 SSE、会话管理、文档管理
├─ services/                 # 应用服务层（Web 层唯一依赖）
│  ├─ chat_service.py        #   对话用例：流式问答、会话列表 / 历史 / 删除
│  └─ document_service.py    #   文档用例：上传落盘、入库、删除
├─ agent_core/
│  ├─ rag_agent.py           # LangGraph 问答图：追问改写 → 路由 → 检索 / 直答 → 回答
│  └─ llm_client.py          # 对话模型与路由模型的构造
├─ rag_component/
│  ├─ loader.py              # 文档解析分发（MinerU / 本地 Loader）
│  ├─ splitter.py            # 递归切分
│  ├─ embedding.py           # 向量化
│  ├─ rerank.py              # 重排
│  └─ pipeline.py            # 入库流水线 + 混合检索入口
├─ db/                       # 存储层（不反向依赖上层）
│  ├─ postgres/              #   连接池 + checkpointer / 会话 / 文档台账
│  └─ milvus/client.py       #   Milvus 客户端：集合准备、chunk 增删查、混合检索
├─ frontend/                 # React 前端（聊天 / 文档 / 会话侧栏）
├─ test/                     # 冒烟与端到端脚本
├─ constants.py              # 跨层共享常量（上传进度百分比）
└─ requirements.txt
```

## 快速开始

### 1. 环境要求

- Python 3.14（本项目开发运行环境）
- Node.js 20.19+ / 22.12+（Vite 8 要求）
- PostgreSQL（会话记忆与文档台账）
- Milvus（默认 `http://127.0.0.1:19530`；服务端建议 ≥ v3.0.2）
- DeepSeek、SiliconFlow 的 API Key；解析 PDF / Office / 图片还需 MinerU API Key

### 2. 安装依赖

```bash
# 后端（在工作区根目录执行）
pip install -r requirements.txt

# 前端
cd frontend
npm install
```

### 3. 配置 `.env`

在项目根目录创建 `.env`（已被 `.gitignore` 忽略）：

| 变量 | 说明 |
|---|---|
| `DeepSeek_API_KEY` / `DeepSeek_MODEL` / `DeepSeek_BASE_URL` | 对话模型（默认 `https://api.deepseek.com`） |
| `POSTGRESQL_DB_URL` | PostgreSQL 连接串（会话记忆与文档台账） |
| `SiliconFlow_API_KEY` / `SiliconFlow_BASE_URL` | Embedding 与 Rerank 服务 |
| `SiliconFlow_Embedding_MODEL` | 向量模型，维度需与 `MILVUS_EMBEDDING_DIM` 一致 |
| `SiliconFlow_Reranker_MODEL` | 重排模型 |
| `TOP_N_RERANK` / `SCORE_THRESHOLD_RERANK` | 重排保留条数（默认 5）/ 最低分（默认 0.1） |
| `MinerU_API_KEY` | 文档解析（仅 PDF / Office / 图片需要） |
| `MILVUS_URI` / `MILVUS_DB_NAME` / `MILVUS_COLLECTION_NAME` | Milvus 地址与库 / 集合名（默认 `rag_demo` / `docs`） |
| `MILVUS_EMBEDDING_DIM` | 稠密向量维度（默认 4096，建集合时校验） |
| `ROUTE_BACKEND` / `Jev_ROUTE_THRESHOLD` | 路由实现（`llm` / `jev`）与判定阈值（默认 0.5） |

### 4. 启动服务（两个终端）

终端 1 —— 启动后端（默认 8000 端口）：

```
python -m uvicorn backend_api.main:app --host 127.0.0.1 --port 8000 --loop asyncio:SelectorEventLoop
```

（直接 `python -m backend_api.main` 也可以。）

终端 2 —— 启动前端（默认 5173 端口）：

```
cd frontend
npm run dev


```

浏览器打开 http://127.0.0.1:5173 即可对话。

## 接口说明

### 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat/stream` | 流式问答（SSE）；请求体 `{"query", "thread_id"?, "mode"?}`，`mode` 取 `auto` / `retrieve` / `direct` |
| GET | `/api/sessions` | 会话列表（按最近活跃倒序） |
| GET | `/api/sessions/{thread_id}/messages` | 会话聊天记录（不含 system 提示词） |
| DELETE | `/api/sessions/{thread_id}` | 删除会话（幂等） |
| POST | `/api/documents/upload` | 上传文档（multipart，≤ 50 MB），返回 SSE 进度流 |
| GET | `/api/documents` | 文档列表（按最近更新倒序） |
| GET | `/api/documents/categories` | 已有分类（前端输入候选） |
| DELETE | `/api/documents/{doc_id}` | 删除文档：清 Milvus chunk + 删台账行（幂等） |

命令行验证（不依赖前端）：

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"你好","thread_id":"demo-1"}'
```

### SSE 事件格式

| type | 含义 |
|---|---|
| `content` | 一段回答文本，字段 `text` |
| `error`   | 出错信息，字段 `message` |
| `done`    | 回答结束 |


## 常见问题

- **Windows 上启动后端要用 SelectorEventLoop**：psycopg 异步与默认 Proactor 事件循环不兼容，
  启动加 `--loop asyncio:SelectorEventLoop`（直接 `python -m backend_api.main` 也可以）。
- **改过集合 schema 后要重建**：向量维度、分词等配置变更后旧集合不可用，执行 `python -m test.reset_docs_db`。
- **Milvus 建议 ≥ v3.0.2**：v3.0.0 的 `hybrid_search` 在零命中时会报 `unsupported ID type`。
- **启动目录必须是工作区根目录**：`assets/` 等路径按相对路径解析，请在根目录启动后端。
- **长会话 token 线性增长**：`llm_node` 刻意不裁剪历史以保留完整记忆，后续可用滚动摘要兜底。

## 已知限制与后续计划

- 会话标题暂用 `thread_id`，计划新增会话元数据表支持自动生成标题
- 流式回答期间前端的会话切换 / 新建 / 删除会被忽略（简化处理）
- 历史消息未做窗口裁剪或摘要，超长会话的 token 成本会线性上升
- 暂未实现用户鉴权与多租户隔离


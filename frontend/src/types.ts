/**
 * types.ts —— 前端与后端交互用到的数据类型
 * 与 backend_api/main.py 里的接口返回结构一一对应。
 */

/** 一条聊天消息（对应 /api/sessions/{thread_id}/messages 返回的元素） */
export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

/** 检索模式（与后端 ChatRequest.mode 的取值一致） */
export type ChatMode = 'auto' | 'retrieve' | 'direct'

/** 会话摘要（对应 /api/sessions 返回的 sessions 数组元素） */
export interface SessionInfo {
  /** 会话编号（当前直接作为标题显示） */
  thread_id: string
  /** 最近活跃时间，ISO8601 字符串，如 "2026-09-17T08:00:00+00:00" */
  updated_at: string
}

/** 后端 SSE 流里的单个事件（对应 main.py 中 yield 的 JSON） */
export type SseEvent =
  | { type: 'content'; text: string } // 一段回答文本
  | { type: 'error'; message: string } // 出错信息
  | { type: 'done' } // 回答结束

/** 主题模式：浅色 / 深色（切换开关位于侧栏左下角） */
export type Theme = 'light' | 'dark'

/** 侧栏顶部的三个视图（导航项） */
export type ViewKey = 'chat' | 'upload' | 'docs'

/** 文档入库状态（与 db/postgres/documents.py 里的常量对应） */
export type DocumentStatus = 'parsing' | 'ready' | 'failed'

/** 一篇文档的台账信息（对应 GET /api/documents 返回的元素） */
export interface DocumentInfo {
  id: string
  /** 文件名（展示用；也是 assets 下那一级目录/文件的最后一节） */
  file_name: string
  /** 分类；空串表示未分类（文件放在 assets 根目录） */
  category: string
  /** 相对路径（posix 风格），同时是 Milvus chunk 里的 source 字段 */
  source: string
  stored_path: string
  /** 原始文件字节数 */
  file_size: number
  /** 切分出的 chunk 数（失败时为 0） */
  chunk_count: number
  status: DocumentStatus
  /** 失败原因（status 为 failed 时有值） */
  error: string | null
  created_at: string
  updated_at: string
}

/** 上传流程的五个阶段（与后端进度事件的 step 字段一致） */
export type UploadStep = 'save' | 'parse' | 'split' | 'embed' | 'write'

/** 后端的入库进度事件（POST /api/documents/upload 的 SSE 流） */
export type UploadEvent =
  | {
      type: 'progress'
      step: UploadStep
      status: 'running' | 'done'
      message: string
      /** 该阶段完成时的总进度百分比（running 事件一般不带） */
      percent?: number
    }
  | { type: 'done'; document: DocumentInfo; chunk_count: number }
  | { type: 'error'; message: string }

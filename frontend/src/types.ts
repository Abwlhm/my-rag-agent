/**
 * types.ts —— 前端与后端交互用到的数据类型
 * 与 backend_api/main.py 里的接口返回结构一一对应。
 */

/** 一条聊天消息（对应 /api/sessions/{thread_id}/messages 返回的元素） */
export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

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

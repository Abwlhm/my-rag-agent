/**
 * api.ts —— 后端接口封装（前端唯一与 FastAPI 通信的地方）
 *
 * 开发环境下所有 /api 请求都由 Vite 代理到 http://127.0.0.1:8000
 * （见 vite.config.ts），浏览器视角是同源请求，因此后端不需要配置 CORS。
 */
import type { ChatMessage, SessionInfo, SseEvent } from './types'

/** 列出所有历史会话（后端按最近活跃时间倒序返回） */
export async function listSessions(): Promise<SessionInfo[]> {
  const resp = await fetch('/api/sessions')
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
  const data = (await resp.json()) as { sessions: SessionInfo[] }
  return data.sessions
}

/** 读取某个会话的聊天记录（不含 system 提示词） */
export async function fetchMessages(threadId: string): Promise<ChatMessage[]> {
  const resp = await fetch(`/api/sessions/${encodeURIComponent(threadId)}/messages`)
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
  const data = (await resp.json()) as { messages: ChatMessage[] }
  return data.messages
}

/** 删除某个会话（后端会清空 PostgreSQL 中该 thread 的全部持久化数据） */
export async function deleteSession(threadId: string): Promise<void> {
  const resp = await fetch(`/api/sessions/${encodeURIComponent(threadId)}`, {
    method: 'DELETE',
  })
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
}

interface StreamChatParams {
  query: string
  threadId: string
  /** 每收到一段增量文本就回调一次（用于实现"打字机"效果） */
  onDelta: (text: string) => void
}

/**
 * 流式问答：POST /api/chat/stream，逐段读取后端的 SSE 流。
 *
 * 为什么不用浏览器原生 EventSource？
 *   EventSource 只支持 GET，而本接口是 POST + JSON 请求体；
 *   所以改用 fetch + ReadableStream 手动解析 SSE 格式（每行 "data: {json}"）。
 * 遇到后端 error 事件时：先取消读取，再把错误抛给调用方展示。
 */
export async function streamChat({ query, threadId, onDelta }: StreamChatParams): Promise<void> {
  const resp = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, thread_id: threadId }),
  })
  if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`)

  const reader = resp.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = '' // 跨网络分片的行缓冲

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    // SSE 每条事件恰好是一行（以换行结尾）；按行切分后逐条处理
    let newlineIndex = buffer.indexOf('\n')
    while (newlineIndex >= 0) {
      const line = buffer.slice(0, newlineIndex).trim()
      buffer = buffer.slice(newlineIndex + 1)
      newlineIndex = buffer.indexOf('\n')

      if (!line.startsWith('data:')) continue // 跳过空行 / 注释行

      let event: SseEvent
      try {
        event = JSON.parse(line.slice('data:'.length).trim()) as SseEvent
      } catch {
        continue // 非 JSON 内容（理论上不会出现）直接忽略
      }

      if (event.type === 'content') {
        onDelta(event.text)
      } else if (event.type === 'error') {
        await reader.cancel() // 提前断开连接，不再读取后续流
        throw new Error(event.message)
      } else if (event.type === 'done') {
        return // 回答正常结束
      }
    }
  }
}

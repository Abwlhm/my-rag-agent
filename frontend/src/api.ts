/**
 * api.ts —— 后端接口封装（前端唯一与 FastAPI 通信的地方）
 *
 * 开发环境下所有 /api 请求都由 Vite 代理到 http://127.0.0.1:8000
 * （见 vite.config.ts），浏览器视角是同源请求，因此后端不需要配置 CORS。
 */
import type {
  ChatMessage,
  ChatMode,
  DocumentInfo,
  SessionInfo,
  SseEvent,
  UploadEvent,
} from './types'

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
  /** 检索模式（透传给后端 ChatRequest.mode） */
  mode: ChatMode
  /** 每收到一段增量文本就回调一次（用于实现"打字机"效果） */
  onDelta: (text: string) => void
}

/**
 * 通用的 SSE 流读取器（异步生成器）：每读到一条 `data: {...}` 就 yield 解析后的对象。
 *
 * 为什么不用浏览器原生 EventSource？
 *   EventSource 只支持 GET，而本项目的两个流式接口都是 POST（一个是 JSON 请求体，
 *   一个是 multipart 上传），所以统一用 fetch + ReadableStream 手动解析。
 * 无论正常读完还是调用方提前退出（return / 抛错），finally 里都会 cancel 掉 reader。
 */
export async function* readSse<T>(resp: Response): AsyncGenerator<T> {
  if (!resp.body) return

  const reader = resp.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = '' // 跨网络分片的行缓冲

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) return

      buffer += decoder.decode(value, { stream: true })

      // SSE 每条事件恰好是一行（以换行结尾）；按行切分后逐条处理
      let newlineIndex = buffer.indexOf('\n')
      while (newlineIndex >= 0) {
        const line = buffer.slice(0, newlineIndex).trim()
        buffer = buffer.slice(newlineIndex + 1)
        newlineIndex = buffer.indexOf('\n')

        if (!line.startsWith('data:')) continue // 跳过空行 / 注释行

        try {
          yield JSON.parse(line.slice('data:'.length).trim()) as T
        } catch {
          // 非 JSON 内容（理论上不会出现）直接忽略
        }
      }
    }
  } finally {
    // 提前退出（例如收到 error 事件主动中断）时也要释放读取锁
    await reader.cancel().catch(() => undefined)
  }
}

/**
 * 流式问答：POST /api/chat/stream，逐段读取后端的 SSE 流。
 * 遇到后端 error 事件时把错误抛给调用方展示（生成器的 finally 会断开连接）。
 */
export async function streamChat({ query, threadId, mode, onDelta }: StreamChatParams): Promise<void> {
  const resp = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, thread_id: threadId, mode }),
  })
  if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`)

  for await (const event of readSse<SseEvent>(resp)) {
    if (event.type === 'content') {
      onDelta(event.text)
    } else if (event.type === 'error') {
      throw new Error(event.message)
    } else if (event.type === 'done') {
      return // 回答正常结束
    }
  }
}

/* ==================== 知识库文档接口 ==================== */

/** 上传体积上限（MB）：与后端 agent_core/document_service.py 的 MAX_UPLOAD_MB 保持一致 */
export const MAX_UPLOAD_MB = 50
export const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

/** 允许上传的扩展名（与后端 rag_component/loader.py 的分发表一致） */
export const ALLOWED_SUFFIXES = [
  '.pdf',
  '.doc',
  '.docx',
  '.ppt',
  '.pptx',
  '.xls',
  '.xlsx',
  '.png',
  '.jpg',
  '.jpeg',
  '.jp2',
  '.webp',
  '.gif',
  '.bmp',
  '.html',
  '.htm',
  '.txt',
  '.csv',
  '.json',
  '.md',
  '.markdown',
]

/** 从错误响应里取人话提示（后端统一放在 detail 字段；取不到就退回状态码） */
async function readErrorDetail(resp: Response): Promise<string> {
  try {
    const data = (await resp.json()) as { detail?: string }
    return data.detail ?? `HTTP ${resp.status}`
  } catch {
    return `HTTP ${resp.status}`
  }
}

/** 列出已入库的文档（后端按最近更新倒序返回） */
export async function listDocuments(): Promise<DocumentInfo[]> {
  const resp = await fetch('/api/documents')
  if (!resp.ok) throw new Error(await readErrorDetail(resp))
  const data = (await resp.json()) as { documents: DocumentInfo[] }
  return data.documents
}

/** 已有分类（"提交入库"页分类输入框的 datalist 候选） */
export async function listCategories(): Promise<string[]> {
  const resp = await fetch('/api/documents/categories')
  if (!resp.ok) throw new Error(await readErrorDetail(resp))
  const data = (await resp.json()) as { categories: string[] }
  return data.categories
}

/** 删除一篇文档：清掉它在 Milvus 里的全部 chunk，并删除台账行；返回清掉的 chunk 数 */
export async function deleteDocument(docId: string): Promise<number> {
  const resp = await fetch(`/api/documents/${encodeURIComponent(docId)}`, {
    method: 'DELETE',
  })
  if (!resp.ok) throw new Error(await readErrorDetail(resp))
  const data = (await resp.json()) as { deleted_chunks: number }
  return data.deleted_chunks
}

interface UploadDocumentParams {
  file: File
  /** 文件名（默认取 file.name，用户可改）；后端据此决定落盘名与"覆盖"关系 */
  fileName: string
  /** 分类（可留空）：后端把它当成 assets 下的一级目录名 */
  category: string
  /** 每收到一条进度事件就回调一次 */
  onEvent: (event: UploadEvent) => void
}

/**
 * 上传并入库一个文档：POST /api/documents/upload（multipart + SSE 进度流）。
 *
 * 返回入库成功后的台账信息；失败时抛错（调用方 catch 后展示 message）。
 * 注意：浏览器 → 后端这段传输期间拿不到字节级进度（fetch 没有 upload 事件，
 * 换 XHR 才能做），所以"保存文件"阶段只能显示进行中，进度条停在上一档。
 */
export async function uploadDocument({
  file,
  fileName,
  category,
  onEvent,
}: UploadDocumentParams): Promise<DocumentInfo> {
  const form = new FormData()
  form.append('file', file)
  form.append('file_name', fileName)
  form.append('category', category)

  const resp = await fetch('/api/documents/upload', { method: 'POST', body: form })
  // 参数校验类错误（类型不支持、分类非法…）发生在返回流之前，是 400 + JSON
  if (!resp.ok) throw new Error(await readErrorDetail(resp))

  let document: DocumentInfo | null = null
  for await (const event of readSse<UploadEvent>(resp)) {
    onEvent(event)
    if (event.type === 'error') throw new Error(event.message)
    if (event.type === 'done') document = event.document
  }

  if (!document) throw new Error('入库未完成：连接提前结束')
  return document
}

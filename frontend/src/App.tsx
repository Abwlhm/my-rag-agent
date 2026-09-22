/**
 * App.tsx —— 页面根组件（状态与流程编排）
 *
 * 三个视图（由侧栏顶部导航切换）：
 *   - 知识问答（chat）：原有聊天界面 + 会话列表；
 *   - 提交入库（upload）：上传文档到 Milvus，带五阶段进度；
 *   - 文档列表（docs）：查看 / 删除已入库文档。
 *
 * 问答视图的流程：
 *   1) 打开页面：拉取所有历史会话（GET /api/sessions）显示在左侧；
 *   2) "开启新对话"：前端生成新的 thread_id（crypto.randomUUID），清空消息区；
 *   3) 发送消息：先追加"用户消息 + 空的助手占位"，再用 SSE 流式填充助手内容；
 *      结束后刷新会话列表（新会话首次出现、更新时间与排序刷新）；
 *   4) 点击侧栏会话：拉取该会话历史消息（GET .../messages）渲染；
 *   5) 删除会话：调 DELETE 接口清空数据库数据；若删的是当前会话则回到空白态；
 *   6) 主题：侧栏左下方可在浅色 / 深色之间切换，选择持久化在 localStorage。
 *
 * 文档相关：上传成功、删除文档后把 docsRefreshKey +1，文档列表据此重新拉取。
 *
 * 交互约定：流式回答期间忽略"切换会话 / 新建对话 / 删除 / 切视图"等操作（简化处理）。
 */
import { useCallback, useEffect, useState } from 'react'

import { deleteSession, fetchMessages, listSessions, streamChat } from './api'
import ChatView from './components/ChatView'
import DocumentListView from './components/DocumentListView'
import Sidebar from './components/Sidebar'
import UploadView from './components/UploadView'
import type { ChatMessage, SessionInfo, Theme, ViewKey } from './types'

export default function App() {
  /** 当前视图（侧栏导航切换） */
  const [view, setView] = useState<ViewKey>('chat')

  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [streaming, setStreaming] = useState(false)
  const [loadingHistory, setLoadingHistory] = useState(false)
  /** 文档列表的刷新信号：上传成功 / 删除文档后 +1 */
  const [docsRefreshKey, setDocsRefreshKey] = useState(0)

  // 主题：默认浅色；用户的切换选择持久化到 localStorage。
  // （index.html 里有一段内联脚本会在首屏渲染前应用已保存的主题，避免闪屏。）
  const [theme, setTheme] = useState<Theme>(() =>
    localStorage.getItem('theme') === 'dark' ? 'dark' : 'light',
  )

  useEffect(() => {
    // 把当前主题写到 <html data-theme="..."> 上，CSS 变量据此切换配色
    document.documentElement.dataset.theme = theme
    localStorage.setItem('theme', theme)
  }, [theme])

  /** 切换浅色 / 深色主题 */
  const toggleTheme = () => setTheme((prev) => (prev === 'light' ? 'dark' : 'light'))

  /** 切换视图；流式回答期间忽略（避免把正在生成的回答"藏"起来） */
  const handleSelectView = (next: ViewKey) => {
    if (streaming) return
    setView(next)
  }

  /** 刷新侧栏会话列表（挂载时、每轮回答结束后调用） */
  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await listSessions())
    } catch (err) {
      console.error('加载会话列表失败', err)
    }
  }, [])

  useEffect(() => {
    void refreshSessions()
  }, [refreshSessions])

  /** "开启新对话"：生成一个尚未落库的 thread_id，界面回到空白态 */
  const handleNewChat = () => {
    if (streaming) return
    setCurrentThreadId(crypto.randomUUID())
    setMessages([])
  }

  /** 点击侧栏会话：加载它在服务端的聊天记录 */
  const handleSelectSession = async (threadId: string) => {
    if (streaming) return // 约定：流式期间忽略切换
    if (threadId === currentThreadId) return // 点击当前会话无需重复加载
    setCurrentThreadId(threadId)
    setMessages([])
    setLoadingHistory(true)
    try {
      setMessages(await fetchMessages(threadId))
    } catch (err) {
      console.error('加载聊天记录失败', err)
    } finally {
      setLoadingHistory(false)
    }
  }

  /** 删除会话：删除数据库数据；若删的是当前会话，界面回到空白态 */
  const handleDeleteSession = async (threadId: string) => {
    if (streaming) return // 约定：流式期间忽略删除
    try {
      await deleteSession(threadId)
    } catch (err) {
      console.error('删除会话失败', err)
      return
    }
    if (threadId === currentThreadId) {
      setCurrentThreadId(null)
      setMessages([])
    }
    void refreshSessions()
  }

  /** 发送消息并接收流式回答 */
  const handleSend = async (text: string) => {
    if (streaming) return

    // 还没有会话编号时（如刷新页面后直接输入）就地生成一个，
    // 后端会按这个 thread_id 把整轮对话存进 PostgreSQL
    const threadId = currentThreadId ?? crypto.randomUUID()
    setCurrentThreadId(threadId)

    // 1) 先追加用户消息 + 空的助手占位消息（流式增量会写进最后一条）
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: text },
      { role: 'assistant', content: '' },
    ])
    setStreaming(true)

    /** 把一段增量文本追加到"最后一条助手消息"上（打字机效果的核心） */
    const appendDelta = (delta: string) => {
      setMessages((prev) => {
        const next = [...prev]
        const last = next[next.length - 1]
        next[next.length - 1] = { ...last, content: last.content + delta }
        return next
      })
    }

    try {
      await streamChat({ query: text, threadId, onDelta: appendDelta })
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err)
      appendDelta(`\n\n[请求出错] ${reason}`)
    } finally {
      setStreaming(false)
      // 刷新侧栏：新会话首次出现 / 更新时间与排序刷新。
      // 立即刷一次 + 稍后再刷一次：checkpointer 的落库是异步持久化，
      // 偶尔会比流结束稍晚一点，双保险避免新会话"迟到"。
      void refreshSessions()
      window.setTimeout(() => void refreshSessions(), 700)
    }
  }

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        currentThreadId={currentThreadId}
        streaming={streaming}
        view={view}
        theme={theme}
        onToggleTheme={toggleTheme}
        onSelectView={handleSelectView}
        onNewChat={handleNewChat}
        onSelect={handleSelectSession}
        onDelete={handleDeleteSession}
      />

      {/* 主内容区：三个视图互斥显示（不卸载聊天状态，切回来还在） */}
      {view === 'chat' && (
        <ChatView
          messages={messages}
          streaming={streaming}
          loadingHistory={loadingHistory}
          onSend={handleSend}
        />
      )}

      {view === 'upload' && (
        <UploadView
          // 入库成功：让文档列表下次进入时重新拉取
          onUploaded={() => setDocsRefreshKey((key) => key + 1)}
          onGoDocs={() => setView('docs')}
        />
      )}

      {view === 'docs' && (
        <DocumentListView
          refreshKey={docsRefreshKey}
          onChanged={() => setDocsRefreshKey((key) => key + 1)}
          onGoUpload={() => setView('upload')}
        />
      )}
    </div>
  )
}

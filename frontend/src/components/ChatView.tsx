/**
 * ChatView.tsx —— 右侧聊天区
 *
 * 由两块组成：
 *   1) 消息列表：空会话时显示居中欢迎语；点击历史会话时显示"加载中"；
 *   2) 底部输入区：多行文本框 + 发送按钮（Enter 发送 / Shift+Enter 换行）。
 *
 * 细节：
 *   - 输入法（中文拼音）组合期间按 Enter 是"上屏选词"，不能当作发送；
 *   - 流式回答期间禁用输入与发送按钮（与"流式期间忽略其他操作"的约定一致）；
 *   - 每次消息变化后自动滚动到底部。
 */
import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, KeyboardEvent } from 'react'

import type { ChatMessage } from '../types'
import MessageItem from './MessageItem'

interface ChatViewProps {
  messages: ChatMessage[]
  streaming: boolean
  loadingHistory: boolean
  onSend: (text: string) => void
}

export default function ChatView({ messages, streaming, loadingHistory, onSend }: ChatViewProps) {
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  // 新消息 / 流式增量更新时，保持视图停在消息区底部
  useEffect(() => {
    bottomRef.current?.scrollIntoView()
  }, [messages])

  /** 发送：把文本交给上层（App），清空输入框并把焦点留在输入区 */
  const send = () => {
    const text = input.trim()
    if (!text || streaming) return
    onSend(text)
    setInput('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto' // 高度复位
      textareaRef.current.focus()
    }
  }

  /** 键盘策略：Enter 发送；Shift+Enter 换行；输入法组合期间不发送 */
  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || event.shiftKey) return
    if (event.nativeEvent.isComposing) return // 中文输入法候选框确认键
    event.preventDefault()
    send()
  }

  /** 输入框随内容自动增高（上限约 6 行，超出后内部滚动） */
  const handleChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    setInput(event.target.value)
    const el = event.target
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 144)}px`
  }

  return (
    <main className="chat-view">
      <div className="messages">
        {/* 空会话：居中欢迎语 */}
        {messages.length === 0 && !loadingHistory && (
          <div className="welcome">
            <div className="welcome-title">有什么可以帮你的？</div>
            <div className="welcome-sub">Enter 发送 · Shift + Enter 换行</div>
          </div>
        )}

        {loadingHistory && <div className="history-loading">正在加载聊天记录…</div>}

        {messages.map((message, index) => (
          // 消息只做"追加/替换最后一条"，用下标做 key 足够稳定
          <MessageItem key={index} message={message} />
        ))}

        {/* 滚动锚点：始终位于最底部 */}
        <div ref={bottomRef} />
      </div>

      <div className="composer">
        <textarea
          ref={textareaRef}
          className="composer-input"
          value={input}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={streaming ? '正在回答中…' : '给对话 Agent 发送消息'}
          rows={1}
          disabled={streaming}
        />
        <button
          type="button"
          className="send-btn"
          onClick={send}
          disabled={streaming || input.trim() === ''}
          title="发送"
        >
          <SendIcon />
        </button>
      </div>
    </main>
  )
}

/** 发送按钮的"向上箭头"图标（内联 SVG，避免引入图标库） */
function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="2.2">
      <path d="M12 19V5m0 0-6 6m6-6 6 6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

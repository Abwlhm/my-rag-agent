import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import type { ChatMessage } from '../types'

interface MessageItemProps {
  message: ChatMessage
}

/**
 * MessageItem —— 单条消息的展示：
 *   - 用户消息：蓝色气泡、右对齐（保留输入里的换行）
 *   - 助手消息：Markdown 渲染、左对齐
 *   - 助手消息内容为空时（流式刚开始、还没有第一个 token）显示闪烁光标
 */
export default function MessageItem({ message }: MessageItemProps) {
  if (message.role === 'user') {
    return (
      <div className="message-row user">
        <div className="bubble">{message.content}</div>
      </div>
    )
  }

  return (
    <div className="message-row assistant">
      <div className="markdown-body">
        {message.content ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
        ) : (
          <span className="typing-cursor" />
        )}
      </div>
    </div>
  )
}

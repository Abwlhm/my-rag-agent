/**
 * Sidebar.tsx —— 左侧边栏
 *
 * 组成：
 *   1) 侧栏底部视图导航：文档列表 / 提交入库 / 知识问答（切换右侧主内容区，排在主题开关上方）；
 *   2) 只有"知识问答"视图下才显示"开启新对话"按钮 + 会话列表；
 *   3) 会话悬浮交互：鼠标悬在会话上时右侧出现"..."按钮，点击弹出菜单——只有"删除"一项；
 *   4) 删除确认弹窗：点击"删除"后先确认，再调后端清空该会话的数据库数据；
 *   5) 最底部：浅色/深色主题切换——显示"当前主题"（浅色=小太阳图标/浅色模式，
 *      深色=月亮图标/深色模式），点击即切换。
 *
 * 交互约定：streaming=true（正在流式回答）时禁用切换 / 删除 / 切视图（点击被忽略）。
 */
import { useEffect, useState } from 'react'
import type { MouseEvent, ReactNode } from 'react'

import type { SessionInfo, Theme, ViewKey } from '../types'
import { formatTime } from '../utils'

interface SidebarProps {
  sessions: SessionInfo[]
  currentThreadId: string | null
  streaming: boolean
  /** 当前视图：决定导航高亮，也决定要不要显示会话列表 */
  view: ViewKey
  /** 当前主题（底部"深色模式"开关的显示状态） */
  theme: Theme
  /** 切换浅色 / 深色主题 */
  onToggleTheme: () => void
  /** 切换视图（知识问答 / 提交入库 / 文档列表） */
  onSelectView: (view: ViewKey) => void
  onNewChat: () => void
  onSelect: (threadId: string) => void
  onDelete: (threadId: string) => void
}

/** "..."菜单的定位信息（用 fixed 定位，避免被滚动容器裁剪） */
interface MenuState {
  threadId: string
  x: number
  y: number
}

/** 三个视图的导航项（顺序对齐原型图：文档列表 / 提交入库 / 知识问答） */
const NAV_ITEMS: { key: ViewKey; label: string; icon: ReactNode }[] = [
  { key: 'docs', label: '文档列表', icon: <ListIcon /> },
  { key: 'upload', label: '提交入库', icon: <UploadIcon /> },
  { key: 'chat', label: '知识问答', icon: <ChatIcon /> },
]

export default function Sidebar({
  sessions,
  currentThreadId,
  streaming,
  view,
  theme,
  onToggleTheme,
  onSelectView,
  onNewChat,
  onSelect,
  onDelete,
}: SidebarProps) {
  const [menu, setMenu] = useState<MenuState | null>(null)
  const [confirmThreadId, setConfirmThreadId] = useState<string | null>(null)

  // 点击页面任意其它位置时关闭"..."菜单（按钮与菜单自身的点击已阻止冒泡）
  useEffect(() => {
    if (!menu) return
    const closeMenu = () => setMenu(null)
    document.addEventListener('click', closeMenu)
    return () => document.removeEventListener('click', closeMenu)
  }, [menu])

  /** 打开/收起某个会话的"..."菜单；菜单出现在按钮右侧（仿 DeepSeek） */
  const toggleMenu = (event: MouseEvent<HTMLButtonElement>, threadId: string) => {
    event.stopPropagation() // 不要触发"选择会话"
    if (menu?.threadId === threadId) {
      setMenu(null)
      return
    }
    const rect = event.currentTarget.getBoundingClientRect()
    setMenu({
      threadId,
      x: rect.right + 6,
      // 靠近窗口底部时向上收一点，避免菜单超出屏幕
      y: Math.min(rect.top - 2, window.innerHeight - 64),
    })
  }

  return (
    <aside className="sidebar">
      {/* 会话列表只在"知识问答"视图出现：上传/文档列表面板不需要它 */}
      {view === 'chat' && (
        <>
          <button
            type="button"
            className="new-chat-btn"
            onClick={onNewChat}
            disabled={streaming}
          >
            <PlusIcon />
            开启新对话
          </button>

          {/* 会话列表（流式期间滚动也会关闭菜单） */}
          <nav className="session-list" onScroll={() => setMenu(null)}>
            {sessions.length === 0 && <div className="session-empty">暂无历史会话</div>}

            {sessions.map((session) => (
              <div
                key={session.thread_id}
                className={`session-item${session.thread_id === currentThreadId ? ' active' : ''}`}
                title={session.thread_id}
                onClick={() => {
                  if (!streaming) onSelect(session.thread_id)
                }}
              >
                <div className="session-text">
                  {/* 标题当前直接显示 thread_id，后续可换成自动生成的会话标题 */}
                  <div className="session-title">{session.thread_id}</div>
                  <div className="session-time">{formatTime(session.updated_at)}</div>
                </div>

                <button
                  type="button"
                  className="session-more"
                  title="更多操作"
                  disabled={streaming}
                  onClick={(event) => toggleMenu(event, session.thread_id)}
                >
                  <DotsIcon />
                </button>
              </div>
            ))}
          </nav>
        </>
      )}

      {/* "..."展开的悬浮菜单：只需要"删除"一个选项 */}
      {menu && (
        <div
          className="session-menu"
          style={{ left: menu.x, top: menu.y }}
          onClick={(event) => event.stopPropagation()}
        >
          <button
            type="button"
            className="session-menu-item danger"
            onClick={() => {
              setConfirmThreadId(menu.threadId)
              setMenu(null)
            }}
          >
            <TrashIcon />
            删除
          </button>
        </div>
      )}

      {/* 删除确认弹窗：删除会清空数据库里的会话数据，避免误点 */}
      {confirmThreadId && (
        <div className="modal-overlay" onClick={() => setConfirmThreadId(null)}>
          <div className="modal" onClick={(event) => event.stopPropagation()}>
            <div className="modal-title">删除会话</div>
            <div className="modal-desc">该会话的全部聊天记录将从数据库永久删除，且无法恢复。</div>
            <div className="modal-actions">
              <button
                type="button"
                className="btn ghost"
                onClick={() => setConfirmThreadId(null)}
              >
                取消
              </button>
              <button
                type="button"
                className="btn danger"
                onClick={() => {
                  onDelete(confirmThreadId)
                  setConfirmThreadId(null)
                }}
              >
                删除
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 底部：视图导航（文档列表 / 提交入库 / 知识问答），排在主题开关上方 */}
      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.key}
            type="button"
            className={`nav-item${view === item.key ? ' active' : ''}`}
            disabled={streaming}
            onClick={() => onSelectView(item.key)}
          >
            {item.icon}
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      {/* 最底部：浅色/深色主题切换（显示当前主题，点击切换；选择由 App 持久化到 localStorage） */}
      <div className="sidebar-footer">
        <button
          type="button"
          className="theme-toggle"
          onClick={onToggleTheme}
          title="切换深色 / 浅色主题"
        >
          {theme === 'light' ? <SunIcon /> : <MoonIcon />}
          <span className="theme-label">{theme === 'light' ? '浅色模式' : '深色模式'}</span>
        </button>
      </div>
    </aside>
  )
}

/* ---------- 内联小图标（避免引入图标库） ---------- */

/** 加号：开启新对话 */
function PlusIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 5v14M5 12h14" strokeLinecap="round" />
    </svg>
  )
}

/** 三个点：更多操作（仿 DeepSeek 截图中的 "…"） */
function DotsIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
      <circle cx="5" cy="12" r="1.7" />
      <circle cx="12" cy="12" r="1.7" />
      <circle cx="19" cy="12" r="1.7" />
    </svg>
  )
}

/** 垃圾桶：删除 */
function TrashIcon() {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path
        d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2m-9 0 1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

/** 月亮：深色模式（切换按钮图标） */
function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path
        d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79Z"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

/** 小太阳：浅色模式（与月亮图标相对应） */
function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8">
      <circle cx="12" cy="12" r="4" />
      <path
        d="M12 2v2m0 16v2M4.93 4.93l1.41 1.41m11.32 11.32 1.41 1.41M2 12h2m16 0h2M4.93 19.07l1.41-1.41m11.32-11.32 1.41-1.41"
        strokeLinecap="round"
      />
    </svg>
  )
}

/** 列表：视图导航"文档列表" */
function ListIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path
        d="M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

/** 上传云朵：视图导航"提交入库" */
function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path
        d="M7 17a4 4 0 0 1-.6-7.96A5.5 5.5 0 0 1 17.3 8.2A3.9 3.9 0 0 1 17 17"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M12 12v8m0-8-2.8 2.8M12 12l2.8 2.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

/** 对话气泡：视图导航"知识问答" */
function ChatIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path
        d="M21 12a8 8 0 0 1-11.6 7.1L4 20.5l1.4-5.2A8 8 0 1 1 21 12Z"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

/**
 * DocumentListView.tsx —— "文档列表"视图：查看已入库文档、按文件名/分类筛选、删除
 *
 * 数据来源是后端的 documents 台账表（PostgreSQL），不是 Milvus：
 * 台账里有分类、分块数、状态、上传时间这些"文档级"信息，向量库里只有 chunk 级数据。
 * 删除一篇文档 = 清掉它在 Milvus 里的全部 chunk + 删掉台账行（磁盘原始文件保留）。
 */
import { useCallback, useEffect, useMemo, useState } from 'react'

import { deleteDocument, listDocuments } from '../api'
import type { DocumentInfo, DocumentStatus } from '../types'
import { formatBytes, formatTime } from '../utils'

/** 状态文案与样式类名 */
const STATUS_TEXT: Record<DocumentStatus, string> = {
  parsing: '处理中',
  ready: '已入库',
  failed: '失败',
}

interface DocumentListViewProps {
  /** 数值变化时重新拉列表（上传成功 / 删除后由上层 +1） */
  refreshKey: number
  /** 列表数据变化后通知上层（例如删除完成后） */
  onChanged: () => void
  /** 库为空时引导去上传 */
  onGoUpload: () => void
}

export default function DocumentListView({
  refreshKey,
  onChanged,
  onGoUpload,
}: DocumentListViewProps) {
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [keyword, setKeyword] = useState('')
  const [categoryFilter, setCategoryFilter] = useState('')
  /** 待确认删除的文档（非空时显示确认弹窗） */
  const [confirming, setConfirming] = useState<DocumentInfo | null>(null)
  const [deleting, setDeleting] = useState(false)

  /** 拉取文档列表 */
  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      setDocuments(await listDocuments())
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  // 首次进入 + 上层要求刷新（refreshKey 变化）时重新拉取
  useEffect(() => {
    void load()
  }, [load, refreshKey])

  /** 分类下拉的候选：从当前数据里现算，不必再调接口 */
  const categories = useMemo(() => {
    const set = new Set<string>()
    for (const doc of documents) {
      if (doc.category) set.add(doc.category)
    }
    return [...set].sort()
  }, [documents])

  /** 前端过滤：文件名关键字 + 分类 */
  const visible = useMemo(() => {
    const key = keyword.trim().toLowerCase()
    return documents.filter((doc) => {
      if (categoryFilter && doc.category !== categoryFilter) return false
      if (key && !doc.file_name.toLowerCase().includes(key)) return false
      return true
    })
  }, [documents, keyword, categoryFilter])

  /** 执行删除：先清 Milvus chunk，再删台账行 */
  const handleDelete = async (doc: DocumentInfo) => {
    setDeleting(true)
    try {
      await deleteDocument(doc.id)
      setConfirming(null)
      // 通知上层把 refreshKey +1：本组件监听到变化后会重新拉列表（只发一次请求）
      onChanged()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setDeleting(false)
    }
  }

  return (
    <main className="page-view">
      <header className="page-header">
        <h1 className="page-title">文档列表</h1>
        <p className="page-desc">
          已入库文档（{documents.length} 篇）。删除会清掉该文档在 Milvus 里的全部 chunk，
          磁盘上的原始文件保留。
        </p>
      </header>

      <section className="card">
        {/* 工具栏：文件名搜索 + 分类筛选 + 刷新 */}
        <div className="toolbar">
          <input
            className="field-input toolbar-search"
            value={keyword}
            placeholder="按文件名搜索"
            onChange={(event) => setKeyword(event.target.value)}
          />
          <select
            className="field-input toolbar-select"
            value={categoryFilter}
            onChange={(event) => setCategoryFilter(event.target.value)}
          >
            <option value="">全部分类</option>
            {categories.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
          <button type="button" className="btn ghost" onClick={() => void load()} disabled={loading}>
            {loading ? '刷新中…' : '刷新'}
          </button>
        </div>

        {error && <div className="alert error">{error}</div>}

        {loading && documents.length === 0 && <div className="list-hint">正在加载文档列表…</div>}

        {!loading && documents.length === 0 && !error && (
          <div className="list-empty">
            <div className="list-empty-title">知识库还是空的</div>
            <div className="list-empty-desc">先上传一份文档，入库后就会出现在这里。</div>
            <button type="button" className="btn primary" onClick={onGoUpload}>
              去提交入库
            </button>
          </div>
        )}

        {documents.length > 0 && (
          <table className="doc-table">
            <thead>
              <tr>
                <th>文件名</th>
                <th>分类</th>
                <th className="num">分块数</th>
                <th>大小</th>
                <th>上传时间</th>
                <th>状态</th>
                <th className="actions">操作</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((doc) => (
                <tr key={doc.id}>
                  <td>
                    <div className="doc-name" title={doc.source}>
                      {doc.file_name}
                    </div>
                    <div className="doc-source">{doc.source}</div>
                  </td>
                  <td>{doc.category || <span className="muted">未分类</span>}</td>
                  <td className="num">{doc.chunk_count}</td>
                  <td>{formatBytes(doc.file_size)}</td>
                  <td>{formatTime(doc.updated_at)}</td>
                  <td>
                    <span
                      className={`badge ${doc.status}`}
                      title={doc.error ?? undefined}
                    >
                      {STATUS_TEXT[doc.status]}
                    </span>
                  </td>
                  <td className="actions">
                    <button
                      type="button"
                      className="btn danger ghost-danger"
                      onClick={() => setConfirming(doc)}
                    >
                      删除
                    </button>
                  </td>
                </tr>
              ))}
              {visible.length === 0 && (
                <tr>
                  <td colSpan={7} className="list-hint">
                    没有匹配的文档（试试清空搜索或切换分类）
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        )}
      </section>

      {/* 删除确认弹窗（复用会话删除的样式） */}
      {confirming && (
        <div className="modal-overlay" onClick={() => setConfirming(null)}>
          <div className="modal" onClick={(event) => event.stopPropagation()}>
            <div className="modal-title">删除文档</div>
            <div className="modal-desc">
              将从向量库删除「{confirming.file_name}」的全部 {confirming.chunk_count} 个 chunk，
              且无法恢复。磁盘上的原始文件会保留。
            </div>
            <div className="modal-actions">
              <button type="button" className="btn ghost" onClick={() => setConfirming(null)}>
                取消
              </button>
              <button
                type="button"
                className="btn danger"
                onClick={() => void handleDelete(confirming)}
                disabled={deleting}
              >
                {deleting ? '删除中…' : '删除'}
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  )
}

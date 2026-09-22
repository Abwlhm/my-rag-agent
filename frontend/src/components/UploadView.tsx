/**
 * UploadView.tsx —— "提交入库"视图：把本地文档上传到 Milvus 知识库
 *
 * 流程：
 *   1) 选文件（拖拽或点击）+ 确认文件名 / 分类；
 *   2) 点"提交入库"，后端返回 SSE 进度流，界面按五个阶段实时显示状态；
 *   3) 成功后展示结果，可一键跳到"文档列表"。
 *
 * 两点实现说明：
 *   - 浏览器 → 后端这段传输拿不到字节级进度（fetch 没有 upload 事件，换 XHR 才行），
 *     所以"保存文件"阶段只显示进行中，进度条停在上一档；
 *   - 前端校验（扩展名 / 50MB）只是为了快速反馈，后端还会再校验一次兜底。
 */
import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, ReactNode } from 'react'

import {
  ALLOWED_SUFFIXES,
  MAX_UPLOAD_BYTES,
  MAX_UPLOAD_MB,
  listCategories,
  uploadDocument,
} from '../api'
import type { DocumentInfo, UploadEvent, UploadStep } from '../types'
import { formatBytes } from '../utils'

/** 五个阶段的固定顺序与显示名（key 与后端进度事件的 step 一一对应） */
const STEPS: { key: UploadStep; label: string }[] = [
  { key: 'save', label: '保存文件' },
  { key: 'parse', label: '解析文档' },
  { key: 'split', label: '切分文本' },
  { key: 'embed', label: '向量化' },
  { key: 'write', label: '写入向量库' },
]

type StepStatus = 'pending' | 'running' | 'done'

/** 全部阶段都处于"待处理"状态 */
function initialStatuses(): Record<UploadStep, StepStatus> {
  return { save: 'pending', parse: 'pending', split: 'pending', embed: 'pending', write: 'pending' }
}

/** 前端即时校验：返回错误提示，空串表示通过 */
function validate(file: File): string {
  const dotIndex = file.name.lastIndexOf('.')
  const suffix = dotIndex >= 0 ? file.name.slice(dotIndex).toLowerCase() : ''
  if (!ALLOWED_SUFFIXES.includes(suffix)) return `不支持的文件类型：${suffix || '无扩展名'}`
  if (file.size > MAX_UPLOAD_BYTES) return `文件超过 ${MAX_UPLOAD_MB}MB 上限`
  return ''
}

interface UploadViewProps {
  /** 入库成功后通知上层（刷新文档列表的 refreshKey） */
  onUploaded: () => void
  /** 切到"文档列表"视图 */
  onGoDocs: () => void
}

export default function UploadView({ onUploaded, onGoDocs }: UploadViewProps) {
  const [file, setFile] = useState<File | null>(null)
  const [fileName, setFileName] = useState('')
  const [category, setCategory] = useState('')
  const [categories, setCategories] = useState<string[]>([])
  const [dragging, setDragging] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [statuses, setStatuses] = useState<Record<UploadStep, StepStatus>>(initialStatuses)
  const [message, setMessage] = useState('') // 当前阶段的说明文字
  const [percent, setPercent] = useState(0)
  const [error, setError] = useState('')
  const [result, setResult] = useState<DocumentInfo | null>(null)

  const inputRef = useRef<HTMLInputElement | null>(null)

  /** 拉取已有分类（datalist 候选）；进页面拉一次，上传成功后再拉一次 */
  const refreshCategories = async () => {
    try {
      setCategories(await listCategories())
    } catch (err) {
      // 候选拉不到不影响上传（分类可以手输），只记日志
      console.error('加载分类候选失败', err)
    }
  }

  useEffect(() => {
    void refreshCategories()
  }, [])

  /** 选中一个文件（点击选择或拖拽都会走到这里） */
  const pickFile = (next: File | null) => {
    if (!next) return
    const reason = validate(next)
    setError(reason)
    setResult(null)
    setStatuses(initialStatuses())
    setPercent(0)
    setMessage('')
    if (reason) {
      setFile(null)
      setFileName('')
      return
    }
    setFile(next)
    setFileName(next.name) // 默认用原文件名，用户可以在输入框里改
  }

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const picked = event.target.files?.[0] ?? null
    pickFile(picked)
    // 清掉 input 的值，否则"再次选择同一个文件"不会触发 change
    event.target.value = ''
  }

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    if (submitting) return
    pickFile(event.dataTransfer.files?.[0] ?? null)
  }

  /** 清空表单，回到初始状态 */
  const handleClear = () => {
    setFile(null)
    setFileName('')
    setCategory('')
    setError('')
    setMessage('')
    setResult(null)
    setPercent(0)
    setStatuses(initialStatuses())
  }

  /** 提交入库：边收 SSE 进度事件边更新界面 */
  const handleSubmit = async () => {
    if (!file || submitting) return

    const reason = validate(file)
    if (reason) {
      setError(reason)
      return
    }

    setSubmitting(true)
    setError('')
    setResult(null)
    setStatuses(initialStatuses())
    setPercent(0)
    setMessage('正在把文件传给后端…')

    /** 每条进度事件：更新对应阶段状态 + 文案 + 总进度 */
    const applyEvent = (event: UploadEvent) => {
      if (event.type !== 'progress') return
      setStatuses((prev) => ({
        ...prev,
        [event.step]: event.status === 'done' ? 'done' : 'running',
      }))
      setMessage(event.message)
      if (typeof event.percent === 'number') setPercent(event.percent)
    }

    try {
      const doc = await uploadDocument({
        file,
        fileName: fileName.trim() || file.name,
        category: category.trim(),
        onEvent: applyEvent,
      })
      setPercent(100)
      setMessage('入库完成')
      setResult(doc)
      onUploaded() // 通知上层刷新文档列表
      void refreshCategories()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setMessage('')
      // 把"进行中"的阶段退回"待处理"，避免看起来还在跑
      setStatuses((prev) => {
        const next = { ...prev }
        for (const step of STEPS) {
          if (next[step.key] === 'running') next[step.key] = 'pending'
        }
        return next
      })
    } finally {
      setSubmitting(false)
    }
  }

  const showPanel = submitting || result !== null || error !== ''

  return (
    <main className="page-view">
      <header className="page-header">
        <h1 className="page-title">提交文档入库</h1>
        <p className="page-desc">
          上传 PDF / DOCX / PPTX / TXT 等文档，填写分类后提交；后端会解析、切分并写入向量库。
          单文件上限 {MAX_UPLOAD_MB}MB。
        </p>
      </header>

      <section className="card">
        {/* 文件选择区（点击或拖拽） */}
        <div
          className={`dropzone${dragging ? ' dragging' : ''}${file ? ' has-file' : ''}`}
          onClick={() => {
            if (!submitting) inputRef.current?.click()
          }}
          onDragOver={(event) => {
            event.preventDefault()
            if (!submitting) setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <UploadCloudIcon />
          <div className="dropzone-text">
            {file ? file.name : '把 PDF / DOCX / PPTX / TXT 等文件拖到这里，或点击选择'}
          </div>
          <div className="dropzone-hint">
            {file ? formatBytes(file.size) : `单文件不超过 ${MAX_UPLOAD_MB}MB`}
          </div>
          <input
            ref={inputRef}
            className="file-input"
            type="file"
            accept={ALLOWED_SUFFIXES.join(',')}
            onChange={handleInputChange}
            disabled={submitting}
          />
        </div>

        {/* 文件名：决定落盘名，也是"覆盖"判定的依据之一 */}
        <div className="field">
          <label className="field-label" htmlFor="doc-name">
            文件名
          </label>
          <input
            id="doc-name"
            className="field-input"
            value={fileName}
            placeholder="默认使用所选文件的文件名"
            disabled={submitting}
            onChange={(event) => setFileName(event.target.value)}
          />
          <div className="field-hint">
            落盘位置：assets/&lt;分类&gt;/&lt;文件名&gt;；同分类同名再次上传 = 覆盖旧版本
          </div>
        </div>

        {/* 分类：input + datalist（可选已有分类，也可自由输入） */}
        <div className="field">
          <label className="field-label" htmlFor="doc-category">
            分类
          </label>
          <input
            id="doc-category"
            className="field-input"
            list="category-options"
            value={category}
            placeholder="如：制度文件（留空则直接放在 assets 根目录）"
            disabled={submitting}
            onChange={(event) => setCategory(event.target.value)}
          />
          {/* datalist：浏览器原生的"输入 + 候选"控件，零依赖 */}
          <datalist id="category-options">
            {categories.map((item) => (
              <option key={item} value={item} />
            ))}
          </datalist>
          <div className="field-hint">分类会成为 assets 下的一级目录名，可自由输入</div>
        </div>

        <div className="form-actions">
          <button
            type="button"
            className="btn primary"
            onClick={handleSubmit}
            disabled={submitting || !file}
          >
            {submitting ? '入库中…' : '提交入库'}
          </button>
          <button type="button" className="btn ghost" onClick={handleClear} disabled={submitting}>
            清空
          </button>
        </div>
      </section>

      {/* 进度 / 结果面板 */}
      {showPanel && (
        <section className="card">
          <div className="progress-head">
            <span className="progress-title">入库进度</span>
            <span className="progress-percent">{percent}%</span>
          </div>
          <div className="progress-bar">
            <div className="progress-bar-fill" style={{ width: `${percent}%` }} />
          </div>

          <ul className="step-list">
            {STEPS.map((step) => (
              <li key={step.key} className={`step-item ${statuses[step.key]}`}>
                <span className="step-dot" />
                <span className="step-label">{step.label}</span>
                {statuses[step.key] === 'running' && <span className="step-state">进行中…</span>}
                {statuses[step.key] === 'done' && <span className="step-state">完成</span>}
              </li>
            ))}
          </ul>

          {message && <div className="progress-message">{message}</div>}
          {error && <div className="alert error">{error}</div>}

          {result && (
            <div className="alert success">
              <div className="alert-title">入库成功：{result.file_name}</div>
              <div className="alert-line">
                分类：{result.category || '未分类'} · 分块数：{result.chunk_count} · 大小：
                {formatBytes(result.file_size)}
              </div>
              <div className="alert-line">落盘位置：{result.source}</div>
              <div className="alert-actions">
                <button type="button" className="btn ghost" onClick={onGoDocs}>
                  查看文档列表
                </button>
              </div>
            </div>
          )}
        </section>
      )}
    </main>
  )
}

/* ---------- 内联小图标 ---------- */

/** 上传云朵：dropzone 的装饰图标 */
function UploadCloudIcon(): ReactNode {
  return (
    <svg viewBox="0 0 24 24" width="34" height="34" fill="none" stroke="currentColor" strokeWidth="1.6">
      <path
        d="M7 18a4 4 0 0 1-.6-7.96A5.5 5.5 0 0 1 17.3 9.2A3.9 3.9 0 0 1 17 18"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M12 12.5V21m0-8.5-3 3m3-3 3 3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

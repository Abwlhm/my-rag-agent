/**
 * main.tsx —— 应用入口：把 <App/> 挂载到 index.html 的 #root 节点。
 * 全局样式统一在 styles.css 中维护。
 */
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import App from './App.tsx'
import './styles.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

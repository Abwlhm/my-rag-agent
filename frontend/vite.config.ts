import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1', // 显式绑定本机地址，浏览器用 http://127.0.0.1:5173 访问
    port: 5173,
    // 开发环境把 /api 开头的请求代理到后端 FastAPI：
    // 浏览器视角是"同源"请求，因此后端不需要配置 CORS；SSE 流也会原样透传。
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})

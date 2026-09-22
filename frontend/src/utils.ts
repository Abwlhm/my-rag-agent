/**
 * utils.ts —— 通用小工具
 */

/**
 * 把后端的 ISO8601 时间格式化成侧栏用的中文时间：
 *   - 1 分钟内：刚刚
 *   - 1 小时内：N 分钟前
 *   - 24 小时内：N 小时前
 *   - 更早：MM-DD HH:mm（跨年时显示 YYYY-MM-DD）
 */
export function formatTime(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return '' // 非法时间字符串直接留空

  const MINUTE = 60 * 1000
  const HOUR = 60 * MINUTE
  const DAY = 24 * HOUR
  const diffMs = Date.now() - date.getTime()

  if (diffMs < MINUTE) return '刚刚'
  if (diffMs < HOUR) return `${Math.floor(diffMs / MINUTE)} 分钟前`
  if (diffMs < DAY) return `${Math.floor(diffMs / HOUR)} 小时前`

  // 两位数补零用的小函数
  const pad = (n: number) => String(n).padStart(2, '0')
  const monthDay = `${pad(date.getMonth() + 1)}-${pad(date.getDate())}`
  const hourMinute = `${pad(date.getHours())}:${pad(date.getMinutes())}`

  return date.getFullYear() === new Date().getFullYear()
    ? `${monthDay} ${hourMinute}`
    : `${date.getFullYear()}-${monthDay}`
}

/**
 * 把字节数格式化成好读的大小（文档列表用）：
 *   0 → 0 B；1536 → 1.5 KB；1048576 → 1.0 MB
 */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B'

  const units = ['B', 'KB', 'MB', 'GB']
  let value = bytes
  let unitIndex = 0
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024
    unitIndex += 1
  }

  // 字节数不带小数，其它单位保留一位
  return `${unitIndex === 0 ? value : value.toFixed(1)} ${units[unitIndex]}`
}

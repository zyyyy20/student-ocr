import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

const DEFAULT_THRESHOLD = 0.85
const EMPTY_TABLE = {
  headers: [],
  rows: [],
  title: '',
}

const statusStyles = {
  idle: 'text-muted-foreground',
  ok: 'text-primary',
  error: 'text-destructive',
}

const formatBytes = (bytes) => {
  if (!Number.isFinite(bytes) || bytes <= 0) return '-'
  const units = ['B', 'KB', 'MB', 'GB']
  let idx = 0
  let value = bytes
  while (value >= 1024 && idx < units.length - 1) {
    value /= 1024
    idx += 1
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[idx]}`
}

export default function App() {
  const fileInputRef = useRef(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [isDragOver, setIsDragOver] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isExporting, setIsExporting] = useState(false)
  const [downloadUrl, setDownloadUrl] = useState('')
  const [status, setStatus] = useState({
    text: '请选择 PNG/JPG/SVG/XLSX 文件',
    type: 'idle',
  })
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD)
  const [riskColumn, setRiskColumn] = useState('all')
  const [tableData, setTableData] = useState(EMPTY_TABLE)

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl('')
      return
    }
    const url = URL.createObjectURL(selectedFile)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [selectedFile])

  const lowCount = useMemo(() => {
    if (!tableData.rows?.length) return 0
    let count = 0
    tableData.rows.forEach((row) => {
      const confidences = row.confidences || {}
      Object.values(confidences).forEach((value) => {
        if (Number(value) < threshold) count += 1
      })
    })
    return count
  }, [tableData.rows, threshold])

  const totalRows = tableData.rows?.length || 0
  const totalColumns = tableData.headers?.length || 0

  const riskOptions = useMemo(() => {
    if (!tableData.rows?.length) return []
    const counts = new Map()
    tableData.headers.forEach((h) => counts.set(h, 0))
    tableData.rows.forEach((row) => {
      Object.entries(row.confidences || {}).forEach(([key, value]) => {
        if (Number(value) < threshold) {
          counts.set(key, (counts.get(key) || 0) + 1)
        }
      })
    })
    return [...counts.entries()]
      .filter(([, value]) => value > 0)
      .sort((a, b) => b[1] - a[1])
  }, [tableData.headers, tableData.rows, threshold])

  const visibleHeaders = useMemo(() => {
    if (!tableData.headers?.length) return []
    if (riskColumn === 'all') return tableData.headers
    return tableData.headers.filter((h) => h === riskColumn)
  }, [tableData.headers, riskColumn])

  const onFileChange = (file) => {
    setSelectedFile(file)
    setTableData(EMPTY_TABLE)
    setRiskColumn('all')
    setDownloadUrl('')
    setStatus({
      text: file ? '已选择文件，点击“开始识别”进行处理' : '请选择 PNG/JPG/SVG/XLSX 文件',
      type: 'idle',
    })
  }

  const handleFileInput = (event) => {
    const file = event.target.files?.[0] || null
    onFileChange(file)
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragOver(false)
    const file = event.dataTransfer?.files?.[0] || null
    if (!file) return
    const name = file.name.toLowerCase()
    if (!/(png|jpe?g|svg|xlsx)$/.test(name)) {
      setStatus({ text: '不支持的文件类型：仅支持 PNG/JPG/SVG/XLSX', type: 'error' })
      return
    }
    onFileChange(file)
  }

  const handleUpload = async () => {
    if (!selectedFile || isLoading) return
    setIsLoading(true)
    setStatus({ text: '识别中，请稍候…（首次加载 OCR 模型可能较慢）', type: 'idle' })
    try {
      const form = new FormData()
      form.append('file', selectedFile)
      const res = await fetch('/upload', { method: 'POST', body: form })
      if (!res.ok) {
        const detail = await res.json().catch(() => null)
        const msg = detail?.detail || `HTTP ${res.status}`
        throw new Error(msg)
      }
      const data = await res.json()
      setTableData({
        headers: data.headers || [],
        rows: data.rows || [],
        title: data.title || '',
      })
      setRiskColumn('all')
      setDownloadUrl('')
      setStatus({ text: '识别完成，可在右侧表格中修改后导出', type: 'ok' })
    } catch (error) {
      setTableData(EMPTY_TABLE)
      setStatus({ text: `识别失败：${error.message || error}`, type: 'error' })
    } finally {
      setIsLoading(false)
    }
  }

  const handleExport = async () => {
    if (!tableData.headers.length || isExporting) return
    setIsExporting(true)
    setStatus({ text: '导出中…', type: 'idle' })
    try {
      const payload = {
        title: tableData.title || '',
        headers: tableData.headers,
        rows: tableData.rows,
      }
      const res = await fetch('/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        const detail = await res.json().catch(() => null)
        const msg = detail?.detail || `HTTP ${res.status}`
        throw new Error(msg)
      }
      const data = await res.json()
      const url = data.download_url
      if (!url) throw new Error('后端未返回下载链接')
      setDownloadUrl(url)
      setStatus({ text: '导出完成，可点击下载链接', type: 'ok' })
    } catch (error) {
      setStatus({ text: `导出失败：${error.message || error}`, type: 'error' })
    } finally {
      setIsExporting(false)
    }
  }

  const handleClear = () => {
    if (fileInputRef.current) fileInputRef.current.value = ''
    setSelectedFile(null)
    setTableData(EMPTY_TABLE)
    setRiskColumn('all')
    setDownloadUrl('')
    setStatus({ text: '请选择 PNG/JPG/SVG/XLSX 文件', type: 'idle' })
  }

  const updateCell = (rowIndex, header, value) => {
    setTableData((prev) => {
      const rows = [...prev.rows]
      const row = { ...rows[rowIndex] }
      const values = { ...(row.values || {}) }
      const confidences = { ...(row.confidences || {}) }
      values[header] = value
      confidences[header] = 1
      rows[rowIndex] = { ...row, values, confidences }
      return { ...prev, rows }
    })
  }

  const addRow = () => {
    if (!tableData.headers.length) return
    const values = {}
    const confidences = {}
    tableData.headers.forEach((header) => {
      values[header] = ''
      confidences[header] = 1
    })
    setTableData((prev) => ({
      ...prev,
      rows: [...prev.rows, { values, confidences }],
    }))
  }

  const removeRow = (rowIndex) => {
    setTableData((prev) => ({
      ...prev,
      rows: prev.rows.filter((_, index) => index !== rowIndex),
    }))
  }

  return (
    <div className="app-shell theme min-h-screen bg-background text-foreground">
      <div className="absolute inset-0 grain" />
      <div className="relative z-10 mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-6 px-4 py-8">
        <header className="fade-rise relative flex flex-col gap-4 overflow-hidden rounded-2xl border border-border bg-card/80 px-6 py-5 shadow-sm backdrop-blur">
          <div className="hero-sheen" />
          <div className="hero-grid" />
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <h1 className="font-display text-2xl font-semibold tracking-tight">
                班级成绩单自动识别系统
              </h1>
              <p className="text-sm text-muted-foreground">
                上传成绩单，自动识别结构，校对后导出 Excel
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge variant="secondary">CPU 推理</Badge>
              <Badge variant="secondary">多行表格</Badge>
              <Badge variant="secondary">可编辑</Badge>
            </div>
          </div>
          <div className="status-pill">
            <span className="status-dot" />
            <span className="font-medium text-foreground">流程</span>
            <span className="text-muted-foreground">上传 → 识别 → 校对 → 导出</span>
          </div>
        </header>

        <main className="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
          <div className="flex flex-col gap-6">
            <Card className="fade-rise" style={{ animationDelay: '80ms' }}>
              <CardHeader>
                <CardTitle>上传文件</CardTitle>
                <CardDescription>支持 PNG/JPG/SVG/XLSX</CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                <div
                  className={cn(
                    'dropzone relative flex flex-col gap-2 rounded-xl border border-dashed border-border bg-background/70 px-4 py-6 text-sm text-muted-foreground transition',
                    isDragOver && 'border-primary bg-primary/10 text-foreground',
                  )}
                  onDragOver={(event) => {
                    event.preventDefault()
                    setIsDragOver(true)
                  }}
                  onDragLeave={() => setIsDragOver(false)}
                  onDrop={handleDrop}
                  role="button"
                  tabIndex={0}
                  onClick={() => fileInputRef.current?.click()}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault()
                      fileInputRef.current?.click()
                    }
                  }}
                >
                  <div className="text-base font-medium text-foreground">拖拽文件到这里</div>
                  <div>或点击选择文件</div>
                  {selectedFile ? (
                    <div className="text-xs text-foreground/80">
                      已选择：{selectedFile.name}
                    </div>
                  ) : null}
                  <Input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    accept=".png,.jpg,.jpeg,.svg,.xlsx"
                    onChange={handleFileInput}
                  />
                </div>

                <div className="meta-grid">
                  <div className="meta-item">
                    <span className="meta-label">文件名称</span>
                    <span className="meta-value">
                      {selectedFile ? selectedFile.name : '-'}
                    </span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">文件类型</span>
                    <span className="meta-value">
                      {selectedFile ? selectedFile.type || 'unknown' : '-'}
                    </span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">文件大小</span>
                    <span className="meta-value">
                      {selectedFile ? formatBytes(selectedFile.size) : '-'}
                    </span>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  <Button onClick={handleUpload} disabled={!selectedFile || isLoading}>
                    {isLoading ? '识别中…' : '开始识别'}
                  </Button>
                  <Button variant="outline" onClick={handleClear} disabled={!selectedFile}>
                    清空
                  </Button>
                </div>

                <div className="flex flex-col gap-2 rounded-xl border border-border bg-muted/40 px-4 py-3">
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>标红阈值</span>
                    <span className="font-medium text-foreground">{threshold.toFixed(2)}</span>
                  </div>
                  <Slider
                    value={[threshold]}
                    min={0.5}
                    max={0.99}
                    step={0.01}
                    onValueChange={(value) => setThreshold(value[0])}
                  />
                </div>

                <p className={cn('text-sm', statusStyles[status.type] || statusStyles.idle)}>
                  {status.text}
                </p>
                {downloadUrl ? (
                  <a
                    href={downloadUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm font-medium text-primary underline-offset-4 hover:underline"
                  >
                    点击下载导出的 Excel
                  </a>
                ) : null}
              </CardContent>
            </Card>

            <Card className="fade-rise" style={{ animationDelay: '140ms' }}>
              <CardHeader>
                <CardTitle>原始文件预览</CardTitle>
                <CardDescription>识别前可预览文件</CardDescription>
              </CardHeader>
              <CardContent className="flex min-h-[280px] items-center justify-center rounded-xl border border-dashed border-border bg-background/70 p-4">
                {!selectedFile ? (
                  <span className="text-sm text-muted-foreground">等待上传</span>
                ) : selectedFile.name.toLowerCase().endsWith('.xlsx') ? (
                  <span className="text-sm text-muted-foreground">XLSX 无法预览，将直接解析表格</span>
                ) : selectedFile.name.toLowerCase().endsWith('.svg') ? (
                  <object data={previewUrl} type="image/svg+xml" className="h-full w-full" />
                ) : (
                  <img src={previewUrl} alt="预览" className="max-h-[360px] w-full object-contain" />
                )}
              </CardContent>
            </Card>
          </div>

          <Card className="fade-rise flex h-full flex-col" style={{ animationDelay: '200ms' }}>
            <CardHeader className="flex flex-col gap-4">
              <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                <div>
                  <CardTitle>识别结果（可编辑）</CardTitle>
                  <CardDescription>
                    低置信度单元格将显示为无效状态（{lowCount} 处）
                  </CardDescription>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <Select value={riskColumn} onValueChange={setRiskColumn}>
                    <SelectTrigger className="w-[180px]">
                      <SelectValue placeholder="高风险列" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        <SelectItem value="all">全部列</SelectItem>
                        {riskOptions.map(([header, count]) => (
                          <SelectItem key={header} value={header}>
                            {header}（{count}）
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                  <Button variant="outline" onClick={addRow} disabled={!tableData.headers.length}>
                    添加行
                  </Button>
                  <Button onClick={handleExport} disabled={!tableData.headers.length || isExporting}>
                    {isExporting ? '导出中…' : '导出 Excel'}
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="flex flex-1 flex-col gap-4">
              <div className="title-chip rounded-xl border border-border bg-muted/30 px-4 py-3 text-center text-sm text-muted-foreground">
                {tableData.title ? tableData.title : '识别标题会显示在这里'}
              </div>

              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">行数：{totalRows}</Badge>
                <Badge variant="secondary">列数：{totalColumns}</Badge>
                <Badge variant="secondary">低置信度：{lowCount}</Badge>
              </div>

              <div className="table-shell flex-1 overflow-auto rounded-xl border border-border bg-background/70">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[60px] text-center">序号</TableHead>
                      {visibleHeaders.map((header) => (
                        <TableHead key={header}>{header}</TableHead>
                      ))}
                      <TableHead className="w-[100px] text-center">操作</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {!tableData.headers.length ? (
                      <TableRow>
                        <TableCell colSpan={3} className="text-center text-sm text-muted-foreground">
                          尚无数据，请先上传并识别
                        </TableCell>
                      </TableRow>
                    ) : tableData.rows.length === 0 ? (
                      <TableRow>
                        <TableCell
                          colSpan={visibleHeaders.length + 2}
                          className="text-center text-sm text-muted-foreground"
                        >
                          未识别到数据行
                        </TableCell>
                      </TableRow>
                    ) : (
                      tableData.rows.map((row, rowIndex) => (
                        <TableRow key={`${rowIndex}-${visibleHeaders.join('-')}`}>
                          <TableCell className="text-center text-sm text-muted-foreground">
                            {rowIndex + 1}
                          </TableCell>
                          {visibleHeaders.map((header) => {
                            const value = row.values?.[header] ?? ''
                            const confidence = Number(row.confidences?.[header] ?? 1)
                            return (
                              <TableCell key={`${rowIndex}-${header}`}>
                                <Input
                                  value={String(value)}
                                  onChange={(event) => updateCell(rowIndex, header, event.target.value)}
                                  aria-invalid={confidence < threshold}
                                />
                              </TableCell>
                            )
                          })}
                          <TableCell className="text-center">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => removeRow(rowIndex)}
                            >
                              删除
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))
                    )}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
            <CardFooter className="flex items-center justify-between text-xs text-muted-foreground">
              <span>提示：编辑后该单元格默认视为高置信度。</span>
              <span>低置信度阈值：{threshold.toFixed(2)}</span>
            </CardFooter>
          </Card>
        </main>
      </div>
    </div>
  )
}

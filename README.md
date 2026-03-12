# 学生成绩单自动识别系统（班级版）

本项目采用前后端分离架构：
- 后端：FastAPI + PaddleOCR（CPU 推理）
- 前端：原生 HTML/CSS/JS（由后端在 `/` 直接提供页面）

支持上传班级成绩单截图/文件，自动识别为可编辑表格，并导出 Excel。

## 功能特性

- 支持多种输入：PNG/JPG/SVG/XLSX
- 表格识别：优先表格结构解析，必要时自动降级为文本兜底
- 识别结果可编辑：低置信度单元格标红，可调整标红阈值
- 标题识别：支持“带大标题”的成绩单，自动提取标题并展示/导出
- 一键导出：生成 Excel 并提供下载链接（含标题行）

## 环境要求

- 操作系统：Windows 10/11（Linux/macOS 亦可运行，但本项目已在 Windows 验证）
- Python：3.9+（建议 3.9/3.10）
- 依赖：见 requirements.txt（已固定 PaddlePaddle 版本）
- CPU 推理：默认使用 CPU

## 关键依赖版本

- paddleocr==3.4.0
- paddlepaddle==3.2.0
- fastapi==0.115.6
- uvicorn[standard]==0.30.6
- opencv-python==4.10.0.84
- openpyxl==3.1.5
- beautifulsoup4==4.12.3
- lxml==5.3.0
- cairosvg==2.7.1

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

## 快速开始（Windows PowerShell）

可选（加速 PaddleX 模型源检查；建议首次安装后开启）：

```powershell
set PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
```

## 启动服务

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

浏览器访问：

```
http://localhost:8000/
```

## 使用流程

1. 打开页面 `http://localhost:8000/`
2. 拖拽文件到上传区（或点击选择）
3. 点击“开始识别”
4. 在右侧表格中校对/编辑（低置信度单元格会标红；可通过“标红阈值”滑块调整）
5. 点击“导出 Excel”，按提示下载生成的文件

导出文件默认存放在 `backend/static/exports/`，并通过 `/downloads/<filename>` 提供下载。

## 接口一览（开发/联调）

- `GET /`：前端页面
- `POST /upload`：上传文件并识别，返回 `{ headers, rows }`
- `POST /export`：接收前端编辑后的 JSON（含可选 `title`），生成 Excel，返回 `{ download_url }`
- `GET /health`：健康检查

## 目录结构

- `backend/`：FastAPI 后端
  - `backend/main.py`：应用入口、路由
- `backend/services/ocr_service.py`：OCR 与表格识别封装
- `backend/services/parser_service.py`：文件解析入口（PNG/JPG/SVG/XLSX）
- `backend/services/excel_service.py`：Excel 导出
- `frontend/`：前端静态资源（由后端挂载在 `/static`）

## 阶段性说明（当前能力）

- 支持带大标题的成绩单（如“学生成绩表”），标题会显示在页面并写入导出的 Excel
- 支持倾斜截图纠偏（桌面截图/旋转图片可自动校正）
- 表头与数据行识别稳定，合并单元格导致的“空列”会自动剔除

## 图像预处理（当前实现）

- 透明/棋盘背景处理：如 PNG 带透明通道，先合成白底，避免背景干扰
- 倾斜纠偏：优先从“纸面区域”估计倾斜角；失败则使用 Hough 直线与投影法估计
- 纸面裁切：纠偏后在安全条件下裁切到纸面区域，避免裁掉关键列
- 小图增强：对较小截图加白边并放大，提高文字/数字识别稳定性

## 常见问题

1. 若看到 `ConvertPirAttribute2RuntimeAttribute not support` 报错，请确认已使用：

```
paddlepaddle==3.2.0
```

2. 首次识别很慢/卡住？

- Paddle/PaddleX 可能需要首次下载/初始化模型（时间较长属于正常现象）
- 建议开启环境变量以跳过模型源检查：

```bash
set PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
```

3. 识别结果缺字、数字漏位（例如 99 变成 9）？

- 截图分辨率过低或文字贴边时更容易发生，建议：尽量截取清晰、包含完整边框的区域
- 项目已对小图做了加白边与放大预处理，以提高稳定性；仍有问题可提供样例进一步优化


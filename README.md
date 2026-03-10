# 学生成绩单自动识别系统（班级版）

本项目采用前后端分离架构，后端 FastAPI + PaddleOCR（CPU），前端原生 HTML/CSS/JS。

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

## 启动服务

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

浏览器访问：

```
http://localhost:8000/
```

## 常见问题

1. 若看到 `ConvertPirAttribute2RuntimeAttribute not support` 报错，请确认已使用：

```
paddlepaddle==3.2.0
```

2. 如果提示模型源检查耗时较长，可设置环境变量加速：

```bash
set PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
```


# 学生成绩单自动识别系统

本项目是一个面向学生成绩单/班级成绩表场景的 OCR 识别系统，采用前后端分离架构：

- 后端：FastAPI + PaddleOCR/PaddleX + OpenCV
- 前端：原生 HTML/CSS/JS，由 FastAPI 直接提供静态页面

系统支持上传 `PNG / JPG / SVG / XLSX` 文件，自动识别成绩表结构、提取元信息与成绩字段，支持前端校对编辑、Excel 导出，以及识别结果与原图/预处理图之间的高亮联动。

## 1. 当前能力概览

- 支持图片、SVG、XLSX 多种输入格式
- 支持标题、姓名、学号、班级、学院、学期等元信息提取
- 支持表格识别、表头归一化、字段清洗与结构化输出
- 支持低置信度标红、右侧表格人工校对编辑
- 支持 Excel 导出，包含标题、表头样式、冻结首行、自动筛选
- 支持两种高亮联动模式
  - `原图反投影`：在原始图像上高亮选中单元格
  - `预处理图高亮`：在预处理后的识别图上高亮选中单元格

## 2. 技术路线

整体流程：

```text
文件输入
  -> 图像解码 / SVG 解析 / XLSX 读取
  -> 图像预处理
  -> 透视校正 / 旋转纠偏 / 裁切 / 增强
  -> OCR 文本识别
  -> 表格结构恢复
  -> 元信息提取与字段归一化
  -> 前端编辑与高亮联动
  -> Excel 导出
```

图片主路径：

```text
原图
  -> 输入归一化
  -> 文档轮廓检测
  -> 透视校正
  -> 旋转纠偏
  -> 表格/标题区域裁切
  -> 小图增强
  -> OCR + 表格识别
  -> 结构化结果 + 高亮框坐标
```

## 3. 本次改造总结

本次主要围绕“更符合任务书/开题要求”和“提升答辩展示效果”做了以下改动：

### 3.1 图像预处理链路增强

- 将预处理拆为更清晰的流水线，而不是单一黑盒步骤
- 增加文档轮廓检测与透视校正，补齐“扭曲文档几何校正”能力
- 增加旋转纠偏、纸面裁切、标题区域裁切、小图增强
- 对透明 PNG 截图场景单独处理，避免误用透视校正
- 增加 `debug_preprocess()`，可输出各阶段图像用于论文与答辩展示

### 3.2 表格结构恢复增强

- 保留 OCR 框聚类路径，优先适配截图类成绩表
- 增加表格线检测与表格线过滤，避免短孤立竖线干扰
- 针对规则表格，新增单元格区域估计，提升高亮与整格映射稳定性

### 3.3 结果结构化增强

- 增加元信息提取：标题、姓名、学号、班级、学院、学期
- 增加表头别名归一化与多行表头合并
- 增加字段级清洗：学号、学分、成绩等格式化

### 3.4 前端交互增强

- 增加元信息展示区
- 增加高亮模式切换
- 支持点击右侧表头/单元格，左侧原图或预处理图同步高亮
- 高亮改为 `SVG polygon` 叠加，而不是简单矩形框

### 3.5 导出与测试增强

- 修复 Excel 列宽换算问题
- 增加导出样式、冻结首行、自动筛选
- 增加 API / 解析 / Excel 单元测试
- 增加样本评测脚本 `tools/evaluate_samples.py`

## 4. 图像预处理实现细节

这一部分是本次修改的重点，也是答辩时最需要展开说明的部分。

### 4.1 输入归一化

文件位置：

- `backend/services/ocr_service.py`

主要处理：

- 如果输入是灰度图，统一转成 3 通道 BGR
- 如果输入是带透明通道的 PNG，先与白底合成
- 统一后续 OpenCV 处理的输入格式

目的：

- 避免透明背景、棋盘底、灰度输入导致 OCR 和轮廓检测不稳定

### 4.2 文档轮廓检测与透视校正

主要函数：

- `_detect_document_quad()`
- `_order_quad_points()`
- `_perspective_correct_with_matrix()`

处理流程：

1. 灰度化
2. 高斯模糊
3. `Canny` 边缘检测
4. 轮廓查找与面积排序
5. 提取近似四边形轮廓
6. 使用 `cv2.getPerspectiveTransform` 与 `cv2.warpPerspective` 做透视变换

说明：

- 这一步主要面向拍照、扫描、轻度扭曲的成绩单
- 对于透明背景截图场景，系统会跳过透视校正，避免把纯旋转截图误判成透视拍照

### 4.3 旋转纠偏

主要函数：

- `_deskew_rotate_with_matrix()`
- `_estimate_skew_paper()`
- `_estimate_skew_hough()`
- `_estimate_skew_projection()`

处理策略：

- 优先根据纸面区域估计旋转角
- 如果纸面估计不稳定，则退化到 Hough 直线法或投影法
- 最终通过旋转矩阵纠偏，并保留完整几何变换上下文

目的：

- 解决截图旋转、拍照倾斜导致的 OCR 框与表格列线不稳定问题

### 4.4 裁切与增强

主要函数：

- `_crop_to_table_or_title_with_matrix()`
- `_estimate_table_bbox()`
- `_estimate_title_crop_y()`
- `_enhance_for_ocr_with_matrix()`

处理逻辑：

- 优先检测表格主体区域
- 若检测不到稳定表格，则尝试去掉顶部标题区域干扰
- 对小图增加白边并放大，提高数字与中文识别稳定性

### 4.5 几何变换记录

这是高亮联动能成立的关键。

主要函数：

- `preprocess_with_context()`

系统会记录：

- `forward_matrix`
- `inverse_matrix`
- `original_size`
- `processed_size`

作用：

- 能把识别过程中得到的框从预处理图坐标系反投影回原图
- 也能把原图框重新投影到预处理图
- 这使得“原图反投影”和“预处理图高亮”两种模式都可以共存

## 5. 高亮选择联动实现细节

这是本次另一块重点改造内容。

### 5.1 目标

用户在右侧识别结果表格中点击任意表头或单元格时，左侧预览同步高亮对应区域。

支持两种模式：

- `原图反投影`
  - 左侧主预览显示原始文件
  - 高亮框为原图坐标系下的 polygon
- `预处理图高亮`
  - 左侧额外显示预处理后的识别图
  - 高亮框为预处理图坐标系下的 polygon

### 5.2 为什么不能只用简单矩形

如果图片存在旋转、透视、裁切，OCR 返回的内容框在几何上通常不是与屏幕平行的矩形。

因此系统改成：

- 后端返回四边形 `points`
- 前端使用 `SVG polygon` 绘制高亮

好处：

- 更适合旋转图和拍照图
- 不会因为图像变换而出现明显的轴对齐误差

### 5.3 后端如何生成高亮区域

文件位置：

- `backend/services/parser_service.py`

核心思路：

1. 先根据 OCR 匹配到表头和单元格的原始框
2. 估计整张表格的主方向与单元格排列关系
3. 重建单元格区域，而不是只高亮文字本身
4. 同时输出：
   - `points`：原图归一化 polygon
   - `processed_points`：预处理图归一化 polygon

涉及函数：

- `_normalize_polygon()`
- `_estimate_cell_regions()`
- `_estimate_table_axes()`
- `_positions_to_bounds()`
- `_refine_col_bounds()`
- `_refine_row_bounds()`
- `_bounds_to_polygon()`
- `_project_points_to_processed()`
- `_project_points_to_original()`

### 5.4 为什么要从“文字框高亮”升级成“单元格区域高亮”

最初只用 OCR 文本框会出现这些问题：

- 框只包住文字，不包住整格
- 重复值场景容易匹配到别的位置
- 数字列、短姓名列偏差更明显
- 经过旋转/裁切后误差被放大

现在系统会估计：

- 列中心和列边界
- 行中心和行边界
- 表格整体方向和原点

再生成整格 polygon，因此效果更接近用户认知中的“选中单元格”。

### 5.5 两种高亮模式的前端实现

文件位置：

- `frontend/index.html`
- `frontend/script.js`
- `frontend/style.css`

前端增加了：

- 高亮模式选择框 `#previewMode`
- 原图预览框 `#preview`
- 预处理图预览框 `#processedPreview`
- `SVG` 叠加层 `preview-overlay`

点击右侧表头/单元格时：

1. 读取单元格携带的 `data-box`
2. 根据当前模式决定取 `points` 还是 `processed_points`
3. 根据图片实际显示尺寸和 `object-fit: contain` 的偏移量换算 polygon 位置
4. 用 `SVG polygon` 在对应预览框中绘制高亮

### 5.6 当前调参策略

为了让高亮更接近整格而不是只贴文字，当前做了这些调整：

- 首列左边界额外放宽
- 数字列右边界额外放宽
- 行高上下增加 padding
- 预处理图模式优先直接在预处理图坐标系中重建单元格，再映射回原图

这使得：

- 原图模式更适合展示“几何反投影能力”
- 预处理图模式更适合展示“识别与表格结构恢复效果”

## 6. 目录结构

```text
backend/
  main.py                      FastAPI 入口
  services/
    ocr_service.py             OCR、预处理、透视校正、表格线检测
    parser_service.py          文件解析、结构化提取、单元格区域估计
    excel_service.py           Excel 导出
  static/
    exports/                   导出文件目录

frontend/
  index.html                   前端页面
  script.js                    前端交互与高亮联动
  style.css                    页面与高亮样式

tests/
  test_api.py
  test_parser_service.py
  test_excel_service.py

tools/
  evaluate_samples.py          样本批量评测脚本
```

## 7. 环境要求

- Python 3.9+
- Windows 10/11 已验证，Linux/macOS 理论可运行
- 默认 CPU 推理

关键依赖：

- `paddleocr==3.4.0`
- `paddlepaddle==3.2.0`
- `fastapi==0.115.6`
- `uvicorn[standard]==0.30.6`
- `opencv-python==4.10.0.84`
- `openpyxl==3.1.5`

## 8. 安装与启动

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

建议设置：

```powershell
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True"
```

启动服务：

```powershell
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

浏览器访问：

[http://localhost:8000/](http://localhost:8000/)

## 9. 使用流程

1. 上传 `PNG / JPG / SVG / XLSX`
2. 点击“开始识别”
3. 在右侧查看识别结果与低置信度标红
4. 点击表头或单元格，查看左侧对应区域高亮
5. 可切换 `原图反投影` / `预处理图高亮`
6. 必要时在右侧手工修改
7. 点击“导出 Excel”

## 10. 接口说明

- `GET /`
  - 返回前端页面
- `POST /upload`
  - 上传文件并返回结构化识别结果
  - 返回字段包含：
    - `title`
    - `meta`
    - `headers`
    - `rows`
    - `header_boxes`
    - `processed_preview`
- `POST /export`
  - 接收前端编辑后的 JSON 并导出 Excel
- `GET /health`
  - 健康检查

## 11. 测试与评测

运行单元测试：

```powershell
python -m unittest discover -s tests -v
```

运行批量评测：

```powershell
python tools/evaluate_samples.py .\samples --debug-dir .\debug_outputs --output .\report.json
```

建议样本目录结构：

```text
samples/
  sample01.png
  sample01.json
  sample02.jpg
  sample02.json
```

评测输出包括：

- 平均识别耗时
- 字段级准确率
- 完全匹配率
- 单样本摘要
- 预处理阶段调试图

## 12. 当前局限

- 复杂合并单元格、极弱线表格仍可能出现边界估计偏差
- 预处理图高亮模式在特殊样本上仍可能需要继续微调列/行 padding
- 当前重点仍是成绩表识别与结构化输出，不包含结果持久化管理

## 13. 后续可继续增强的方向

- 增加真实样本集与更系统的准确率报告
- 增加更多成绩单模板与字段规则
- 继续细化不同列类型的单元格边界估计
- 增加历史识别结果管理与数据库存储

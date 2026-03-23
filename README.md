# 学生成绩单自动识别系统

本项目用于识别学生成绩单/班级成绩表，支持：

- 上传 `PNG / JPG / SVG / XLSX`
- 自动提取标题、表头、成绩数据
- 前端校对与高亮联动
- Excel 导出

技术栈：

- 后端：FastAPI + PaddleOCR / PaddleX + OpenCV
- 前端：原生 HTML / CSS / JS

## 启动

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

访问：

[http://localhost:8000/](http://localhost:8000/)

运行测试：

```powershell
python -m unittest discover -s tests -v
```

## 项目环境说明

运行环境：

- 操作系统：Windows 10 / 11 已验证
- Python：3.9+
- 推理方式：默认 CPU
- 浏览器：Chrome / Edge 最新版

核心依赖：

- `fastapi==0.115.6`
- `uvicorn[standard]==0.30.6`
- `paddleocr==3.4.0`
- `paddlepaddle==3.2.0`
- `opencv-python==4.10.0.84`
- `openpyxl==3.1.5`
- `beautifulsoup4==4.12.3`
- `lxml==5.3.0`
- `cairosvg==2.7.1`

项目目录中与运行最相关的部分：

```text
backend/
  main.py
  services/
    ocr_service.py
    parser_service.py
    excel_service.py

frontend/
  index.html
  script.js
  style.css

tests/
tools/
```

## 当前处理流程

系统当前处理成绩单的主流程如下：

```text
图片输入
-> 输入归一化
-> 白纸区域检测与裁切
-> 透视校正
-> 旋转纠偏
-> 表格/内容区域聚焦裁切
-> OCR 文本识别
-> 表格结构恢复
-> 碎行合并与数字补识别
-> 字段清洗与结构化输出
-> 前端高亮联动
-> Excel 导出
```

## 1. 文件入口

入口在 `backend/main.py`。

核心接口：

- `POST /upload`
- `POST /export`
- `GET /health`

核心代码：

```python
@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    result = parser_service.parse(
        filename=file.filename or f"upload-{uuid.uuid4().hex}",
        content_type=file.content_type or "",
        data=raw,
    )
    return JSONResponse(content=result)
```

## 2. 图像预处理

核心代码在 `backend/services/ocr_service.py` 的 `preprocess_with_context()`。

它负责：

- 输入归一化
- 白纸区域裁切
- 透视校正
- 旋转纠偏
- 表格/标题区域裁切
- 小图增强
- 记录几何变换矩阵

核心代码：

```python
def preprocess_with_context(self, image_bgr: np.ndarray) -> Dict[str, Any]:
    image_bgr = self._normalize_input_image(image_bgr)

    if alpha is None:
        paper_mask = self._extract_paper_mask(image_bgr)
        crop_bbox = self._safe_crop_bbox_from_mask(image_bgr, paper_mask)
        if crop_bbox is not None:
            image_bgr, crop_matrix = self._crop_by_bbox(image_bgr, crop_bbox)
            transform = crop_matrix @ transform

    if alpha is None:
        quad = self._detect_document_quad(image_bgr)
        if quad is not None:
            image_bgr, matrix = self._perspective_correct_with_matrix(image_bgr, quad)
            transform = matrix @ transform

    image_bgr, deskew_matrix = self._deskew_rotate_with_matrix(image_bgr, alpha_mask=alpha)
    transform = deskew_matrix @ transform

    image_bgr, crop_matrix = self._crop_to_table_or_title_with_matrix(image_bgr)
    transform = crop_matrix @ transform

    image_bgr, enhance_matrix = self._enhance_for_ocr_with_matrix(image_bgr)
    transform = enhance_matrix @ transform
```

### 2.1 输入归一化

目的：

- 灰度图统一转 BGR
- 带透明通道的 PNG 统一合成白底

### 2.2 白纸区域检测与裁切

目的：

- 从木纹桌面、复杂背景中先裁出纸张主体

当前做法：

- 使用 `HSV + LAB` 组合阈值提取亮白区域
- 再用形态学操作过滤噪声
- 选择最像纸张的外轮廓

### 2.3 透视校正

目的：

- 处理拍照造成的梯形变形

### 2.4 旋转纠偏

目的：

- 处理轻微拍歪、截图旋转

当前做法：

- 优先纸面角度
- 失败后退化到 Hough 直线和投影法

### 2.5 表格/内容区域裁切

目的：

- 不把整页空白表格和页脚说明一起送去识别

当前做法：

- 先按表格线检测大表格区域
- 再用文字连通域估计真正有内容的区域
- 二次收紧裁切框

### 2.6 小图增强

目的：

- 提升小数字、小中文的识别率

## 3. OCR 处理详解

OCR 是整个系统的核心。当前实现不是“整图跑一次 OCR 就结束”，而是分成“整图 OCR、表格结构恢复、单元格补识别”三个层次。

### 3.1 整图 OCR 的作用

整图 OCR 由 `backend/services/ocr_service.py` 中的 `recognize_text()` 完成。

它的作用不是只拿最终文本，而是同时提供三类信息：

- 文本内容 `text`
- 置信度 `confidence`
- 检测框 `box`

这些信息后续会同时用于：

- 标题和元信息提取
- 表格结构恢复
- 单元格高亮定位
- 低置信度标红
- 缺失数字的补识别

核心代码：

```python
def recognize_text(self, image_bgr: np.ndarray, *, preprocess: bool = True) -> List[OCRItem]:
    if preprocess:
        image_bgr = self.preprocess_image(image_bgr)

    ocr = self._get_ocr()
    results = ocr.ocr(image_bgr, cls=True)
    ...
    items.append(OCRItem(text=text, confidence=conf, box=box_pts))
    return items
```

### 3.2 PaddleOCR 在这里做了什么

PaddleOCR 在这里承担的是“文本检测 + 文本识别”两件事：

1. 文本检测  
   找出图上每一块文字所在区域，输出 polygon/box。

2. 文本识别  
   对每个文字区域进行字符识别，输出字符串和置信度。

这也是为什么系统后面不仅能得到 `张三 / 95 / 6.4` 这类文本，还能知道它们在图上的位置。

### 3.3 为什么不能只靠 OCR 文本顺序恢复表格

单纯拿 OCR 输出的文本顺序直接拼表格，问题很多：

- 拍照后同一行文字的 y 坐标会有波动
- 表格线和文字会互相干扰
- 同一列的短数字可能会错落
- 有的行会被拆成两段
- 有的数字会漏识别

所以当前系统不是“按 OCR 顺序直接拼”，而是先把 OCR 当作原始观测，再恢复表格结构。

## 4. OCR 之后的表格结构恢复

核心代码在 `backend/services/ocr_service.py` 的 `recognize_table()`。

当前使用三段策略：

1. 规则截图优先 OCR 框聚类
2. 表格线足够稳定时走线结构恢复
3. 不够稳定时回退到 PaddleX 表格识别

核心代码：

```python
def recognize_table(self, image_bgr, *, ocr_items=None, preprocess=True):
    grid_from_items = self._items_to_grid(ocr_items)
    score_items = self._grid_score(grid_from_items)

    cells = self._grid_mask_to_cells(grid_mask) if line_confident else None
    if cells and score_items < 40:
        grid_from_lines = self._line_cells_to_grid(cells, ocr_items)

    if score_items >= 20:
        return grid_from_items

    pred = engine.predict(image_bgr)
```

### 4.1 路径一：OCR 框聚类

适用场景：

- 规则截图
- 边框不强但排版整齐

原理：

- 先按 y 坐标把 OCR 框聚成“行”
- 再按 x 坐标把 OCR 框聚成“列”
- 最终重建二维表格

核心代码：

```python
def _items_to_grid(self, items: Sequence[OCRItem]) -> List[List[str]]:
    enriched.sort(key=lambda t: t[3])  # by y_center
    ...
    if abs(yc - rc) <= y_tol:
        rows[idx].append((it, x1, y1, x2, y2))
    ...
    if abs(xc - col_centers[-1]) <= x_tol:
        ...
```

### 4.2 路径二：表格线结构恢复

适用场景：

- 纸面表格
- 横线/竖线比较明显

原理：

- 先对图像二值化
- 用形态学操作分别提取横线和竖线
- 根据线交点和单元格轮廓恢复表格网格
- 再把 OCR 文本分配到单元格中

这一步的优势是“结构更强”，缺点是对弱线、断线、卷曲表格不一定稳定。

### 4.3 路径三：PaddleX 表格识别兜底

适用场景：

- 前两种路径都不够稳定时

原理：

- 调用 PaddleX `table_recognition` pipeline
- 尝试从更重的结构模型里拿 HTML 表格结果

这个路径不是主路径，而是兜底路径。

## 5. OCR 二次聚焦

核心代码在 `backend/services/parser_service.py` 的 `_parse_image()`。

当前流程：

1. 预处理整图
2. 先做一次整图 OCR
3. 提取标题等元信息
4. 根据 OCR 结果再聚焦到表格有效区域
5. 对聚焦后的图重新做 OCR 和表格识别

核心代码：

```python
ocr_items = self.ocr_service.recognize_text(image_bgr, preprocess=False)
meta = self._extract_transcript_meta(ocr_items)
image_bgr, transform_context, focused = self._focus_table_region(
    image_bgr, ocr_items, transform_context
)
if focused:
    ocr_items = self.ocr_service.recognize_text(image_bgr, preprocess=False)
```

这样做的原因是：

- 第一次 OCR 负责“找区域”
- 第二次 OCR 负责“在更聚焦的区域里提高识别质量”

对拍照成绩单很有效。

## 6. 表格解析与字段结构化

核心代码在 `backend/services/parser_service.py` 的 `_extract_class_grid()`。

当前做法：

- 自动寻找标题行和表头行
- 表头别名归一化
- 抽取数据行
- 空列剔除
- 碎行合并
- 单元格区域估计
- 数字补识别
- 字段清洗

核心代码：

```python
headers_raw = self._merge_multirow_headers(grid, header_row_idx)
headers = [self._normalize_header_name(h.strip()) for h in headers_raw]
...
rows_out = self._merge_fragmented_rows(headers, rows_out)
header_boxes, rows_out = self._estimate_cell_regions(headers, header_boxes, rows_out, transform_context)
rows_out = self._recover_numeric_cells(headers, rows_out, ocr_items, image_bgr, transform_context)
rows_out = self._normalize_transcript_rows(headers, rows_out)
```

### 6.1 碎行合并

解决的问题：

- 拍照表格中，一条记录被错误拆成两行

核心代码：

```python
def _should_merge_fragmented_rows(headers, current_non_empty, next_non_empty):
    overlap = set(current_non_empty) & set(next_non_empty)
    if overlap:
        return False
    current_max = max(headers.index(h) for h in current_non_empty if h in headers)
    next_min = min(headers.index(h) for h in next_non_empty if h in headers)
    return next_min >= current_max
```

### 6.2 单元格区域估计

解决的问题：

- 右侧点击后，左侧高亮不能只框文字，要尽量框整格

原理：

- 先估计表格主方向
- 再估计每列中心、每行中心
- 推导列边界和行边界
- 生成整格 polygon

核心代码：

```python
origin, x_axis, y_axis = self._estimate_table_axes([...], coord_key)
col_bounds = self._positions_to_bounds(col_positions)
row_bounds = self._positions_to_bounds(row_positions)
new_boxes[header] = self._bounds_to_polygon(
    origin, x_axis, y_axis, left, right, top, bottom, transform_context, coord_key
)
```

### 6.3 数字补识别

解决的问题：

- `6.4 / 6.6 / 6.7` 这类短数字、小数值容易漏识别

当前做法：

- 先尝试利用已有 OCR 框补值
- 如果空数字格仍为空，则按单元格区域裁块
- 加白边、放大
- 单独再跑一次局部 OCR

核心代码：

```python
def _recover_numeric_cells(...):
    if not candidate and image_bgr is not None:
        candidate = self._run_local_cell_ocr(
            image_bgr, cell_box, coord_key, transform_context
        )

def _run_local_cell_ocr(self, image_bgr, cell_box, coord_key, transform_context):
    crop = image_bgr[y1:y2 + 1, x1:x2 + 1]
    crop = cv2.copyMakeBorder(crop, 8, 8, 8, 8, ...)
    crop = cv2.resize(crop, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
    items = self.ocr_service.recognize_text(crop, preprocess=True)
```

## 7. 元信息提取

当前支持提取：

- 标题
- 姓名
- 学号
- 班级
- 学院
- 学期

核心代码：

```python
field_patterns = {
    "姓名": r"姓名[:：]?\s*([^\s:：]+)",
    "学号": r"学号[:：]?\s*([A-Za-z0-9]+)",
    "班级": r"班级[:：]?\s*([^\s]+)",
    "学院": r"学院[:：]?\s*([^\s]+)",
    "学期": r"(?:学期|学年学期)[:：]?\s*([^\s]+)",
}
```

## 8. 前端高亮联动

前端文件：

- `frontend/index.html`
- `frontend/script.js`
- `frontend/style.css`

当前支持两种模式：

- `原图反投影`
- `预处理图高亮`

核心做法：

- 后端返回 `points` 和 `processed_points`
- 前端点击右侧表头或单元格时，读取 `data-box`
- 按当前模式在对应预览区绘制 `SVG polygon`

核心代码：

```javascript
function showPreviewHighlight(shape) {
  const pointsKey = previewMode === "processed" ? "processed_points" : "points";
  const points = Array.isArray(shape?.[pointsKey]) ? shape[pointsKey] : null;
  drawPolygonOnPreview(targetContainer, targetImage, targetOverlay, points);
}
```

## 9. Excel 导出

核心代码在 `backend/services/excel_service.py`。

当前导出包含：

- 标题
- 表头样式
- 冻结首行
- 自动筛选
- 列宽自适应

核心代码：

```python
def export_transcript(self, payload: Dict[str, Any]) -> str:
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    self._apply_table_style(ws, row_offset, headers, rows)
    self._autosize_columns(ws, headers, rows)
```

## 10. 测试

当前测试包括：

- `tests/test_api.py`
- `tests/test_parser_service.py`
- `tests/test_excel_service.py`

运行：

```powershell
python -m unittest discover -s tests -v
```

## 11. 当前已知边界

- 拍照图优于之前，但仍明显弱于规则截图
- 复杂卷曲、强反光、严重透视时仍可能识别不稳
- 数值列的小数识别已补强，但极端模糊样本仍可能漏识别

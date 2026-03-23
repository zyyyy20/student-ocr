# 学生成绩单自动识别系统

本项目用于识别学生成绩单/班级成绩表，支持：

- 上传 `PNG / JPG / SVG / XLSX`
- 自动提取标题、表头、成绩数据
- 前端校对与高亮联动
- Excel 导出

技术栈：

- 后端：FastAPI + PaddleOCR/PaddleX + OpenCV
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

下面按步骤说明当前实现。

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

核心代码：

```python
def _normalize_input_image(self, image_bgr: np.ndarray) -> np.ndarray:
    if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 4:
        bgr = image_bgr[:, :, :3]
        alpha = image_bgr[:, :, 3]
        return self._alpha_composite_white(bgr, alpha)
    if len(image_bgr.shape) == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    return image_bgr
```

### 2.2 白纸区域检测与裁切

目的：

- 从木纹桌面、复杂背景中先裁出纸张主体

当前做法：

- 使用 `HSV + LAB` 组合阈值提取亮白区域
- 再用形态学操作过滤噪声
- 选择最像“纸张”的外轮廓

核心代码：

```python
def _extract_paper_mask(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    mask_hsv = cv2.inRange(hsv, (0, 0, 210), (179, 70, 255))
    mask_lab = cv2.inRange(lab[:, :, 0], 205, 255)
    mask = cv2.bitwise_or(mask_hsv, mask_lab)
    ...
    largest = max(contours, key=contour_score)
    cv2.drawContours(paper, [largest], -1, 255, thickness=-1)
```

### 2.3 透视校正

目的：

- 处理拍照造成的梯形变形

核心代码：

```python
def _perspective_correct_with_matrix(self, image_bgr, quad):
    rect = self._order_quad_points(quad)
    matrix = cv2.getPerspectiveTransform(rect, dst)
    corrected = cv2.warpPerspective(image_bgr, matrix, (max_width, max_height))
    return corrected, matrix
```

### 2.4 旋转纠偏

目的：

- 处理轻微拍歪、截图旋转

当前做法：

- 优先纸面角度
- 失败后退化到 Hough 直线和投影法

核心代码：

```python
def _deskew_rotate_with_matrix(self, image_bgr, alpha_mask=None):
    if alpha_mask is not None:
        angle = self._estimate_skew_min_area(alpha_mask)
    if angle is None and paper_mask is not None:
        angle = self._estimate_skew_paper(paper_mask)
    if angle is None:
        angle = self._estimate_skew_angle(image_bgr)
```

### 2.5 表格/内容区域裁切

目的：

- 不把整页空白表格和页脚说明一起送去识别

当前做法：

- 先按表格线检测大表格区域
- 再用文字连通域估计真正有内容的区域
- 二次收紧裁切框

核心代码：

```python
def _estimate_table_bbox(self, image_bgr):
    bw = self._binarize_for_table(gray)
    _horiz, _vert, grid = self._extract_grid_lines(bw)
    x, y, ww, hh = cv2.boundingRect(max(contours, key=cv2.contourArea))

    content_box = self._estimate_text_content_bbox(image_bgr)
    if content_box is not None:
        cx1, cy1, cx2, cy2 = content_box
        x = min(x, cx1)
        y = min(y, cy1)
        right = max(x + ww, cx2)
        bottom = min(y + hh, cy2)
```

### 2.6 小图增强

目的：

- 提升小数字、小中文的识别率

核心代码：

```python
def _enhance_for_ocr_with_matrix(self, image_bgr):
    padded = cv2.copyMakeBorder(image_bgr, pad, pad, pad, pad, ...)
    scaled = cv2.resize(padded, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return scaled, matrix
```

## 3. OCR 与二次聚焦

核心代码在 `backend/services/parser_service.py` 的 `_parse_image()`。

当前流程：

1. 预处理整图
2. 做一次整图 OCR
3. 提取标题等元信息
4. 根据 OCR 结果再聚焦到表格有效区域
5. 对聚焦后的图重新做 OCR 和表格识别

核心代码：

```python
def _parse_image(self, data: bytes) -> Dict[str, Any]:
    image_bgr = self.ocr_service.decode_image(data)
    preprocess_result = self.ocr_service.preprocess_with_context(image_bgr)
    image_bgr = preprocess_result["image"]
    transform_context = preprocess_result["context"]

    ocr_items = self.ocr_service.recognize_text(image_bgr, preprocess=False)
    meta = self._extract_transcript_meta(ocr_items)
    image_bgr, transform_context, focused = self._focus_table_region(
        image_bgr, ocr_items, transform_context
    )
    if focused:
        ocr_items = self.ocr_service.recognize_text(image_bgr, preprocess=False)
```

## 4. 表格结构恢复

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

## 5. 表格解析与字段结构化

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

### 5.1 碎行合并

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

### 5.2 单元格区域估计

解决的问题：

- 右侧点击后，左侧高亮不能只框文字，要尽量框整格

核心代码：

```python
origin, x_axis, y_axis = self._estimate_table_axes([...], coord_key)
col_bounds = self._positions_to_bounds(col_positions)
row_bounds = self._positions_to_bounds(row_positions)
new_boxes[header] = self._bounds_to_polygon(
    origin, x_axis, y_axis, left, right, top, bottom, transform_context, coord_key
)
```

### 5.3 数字补识别

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

## 6. 元信息提取

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

## 7. 前端高亮联动

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

## 8. Excel 导出

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

## 9. 测试

当前测试包括：

- `tests/test_api.py`
- `tests/test_parser_service.py`
- `tests/test_excel_service.py`

运行：

```powershell
python -m unittest discover -s tests -v
```

## 10. 目录结构

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

## 11. 当前已知边界

- 拍照图优于之前，但仍明显弱于规则截图
- 复杂卷曲、强反光、严重透视时仍可能识别不稳
- 数值列的小数识别已补强，但极端模糊样本仍可能漏识别

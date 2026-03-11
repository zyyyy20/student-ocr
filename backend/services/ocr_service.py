from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class OCRItem:
    text: str
    confidence: float
    box: List[List[float]]


class OCRService:
    """
    PaddleOCR 统一封装：
    - 普通文字 OCR（用于元信息提取、兜底解析、置信度匹配）
    - 表格识别（优先使用 PaddleX TableRecognition；若不可用则用检测框做简单重建）
    """

    def __init__(self, *, use_gpu: bool = False, lang: str = "ch") -> None:
        self.use_gpu = use_gpu
        self.lang = lang
        self._ocr = None
        self._table_engine = None
        # 小图（截图/裁剪图）在 PaddleOCR 中容易漏字/漏数字。
        # 通过“加白边 + 放大”可显著提升识别稳定性。
        self._small_image_min_side = 600
        self._small_image_pad = 20
        self._small_image_scale = 2.0

    @staticmethod
    def _create_with_compat(factory, kwargs: Dict[str, Any]):
        """
        PaddleOCR/PPStructure 在不同版本中参数可能不完全一致。
        这里采用“尝试初始化 → 若提示 Unknown argument 则剔除该参数 → 重试”的策略，
        以确保在 CPU 环境下尽可能稳定运行。
        """
        k = dict(kwargs)
        for _ in range(20):
            try:
                return factory(**k)
            except Exception as e:
                msg = str(e)
                m = re.search(r"Unknown argument:\s*([A-Za-z_][A-Za-z0-9_]*)", msg)
                if m:
                    bad = m.group(1)
                    if bad in k:
                        k.pop(bad, None)
                        continue
                raise
        return factory(**k)

    def _get_ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR

            self._ocr = self._create_with_compat(
                PaddleOCR,
                {
                    "use_angle_cls": True,
                    "lang": self.lang,
                    "use_gpu": self.use_gpu,
                    "show_log": False,
                },
            )
        return self._ocr

    def _get_table_engine(self):
        if self._table_engine is None:
            import paddlex

            self._table_engine = paddlex.create_pipeline("table_recognition")
        return self._table_engine

    @staticmethod
    def decode_image(data: bytes) -> np.ndarray:
        """
        将图片 bytes 解码为 OpenCV BGR ndarray。
        支持 PNG/JPG 等常见格式。
        """
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解码图片：请确认文件为 PNG/JPG")
        return img

    def preprocess_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        OCR 预处理：
        - 对较小截图加白边，避免边缘文字被截断/漏检
        - 对较小截图放大，提高字符可分辨率（如 99->9 的问题）
        """
        if image_bgr is None:
            return image_bgr

        h, w = image_bgr.shape[:2]
        if min(h, w) >= self._small_image_min_side:
            return image_bgr

        padded = cv2.copyMakeBorder(
            image_bgr,
            self._small_image_pad,
            self._small_image_pad,
            self._small_image_pad,
            self._small_image_pad,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        scaled = cv2.resize(
            padded,
            None,
            fx=self._small_image_scale,
            fy=self._small_image_scale,
            interpolation=cv2.INTER_CUBIC,
        )
        return scaled

    def recognize_text(self, image_bgr: np.ndarray, *, preprocess: bool = True) -> List[OCRItem]:
        """
        返回整图 OCR 的文本行信息（含置信度、检测框）。
        PaddleOCR 返回结构示例：[[box, (text, conf)], ...]
        """
        if preprocess:
            image_bgr = self.preprocess_image(image_bgr)

        ocr = self._get_ocr()
        try:
            results = ocr.ocr(image_bgr, cls=True)
        except Exception as e:
            msg = str(e)
            if "cls" in msg and ("unexpected keyword" in msg or "Unknown argument" in msg):
                results = ocr.ocr(image_bgr)
            else:
                raise
        items: List[OCRItem] = []

        if not results:
            return items

        # PaddleOCR 3.x 可能返回 list[dict]（rec_texts/rec_scores/rec_polys）
        if isinstance(results, list) and results and isinstance(results[0], dict):
            d = results[0]
            texts = d.get("rec_texts")
            if texts is None:
                texts = []
            scores = d.get("rec_scores")
            if scores is None:
                scores = []
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            polys = d.get("rec_polys")
            if polys is None:
                polys = d.get("dt_polys")
            if polys is None:
                polys = []
            boxes = d.get("rec_boxes")
            if boxes is None:
                boxes = []
            if hasattr(boxes, "tolist"):
                boxes = boxes.tolist()

            n = min(len(texts), len(scores))
            for i in range(n):
                text = str(texts[i]).strip()
                if not text:
                    continue
                conf = float(scores[i])
                box_pts = self._coerce_box(polys[i] if i < len(polys) else (boxes[i] if i < len(boxes) else None))
                items.append(OCRItem(text=text, confidence=conf, box=box_pts))
            return items

        # PaddleOCR 2.x 旧格式
        for line in results[0] if isinstance(results, list) else results:
            if not line or len(line) < 2:
                continue
            box, rec = line[0], line[1]
            if not rec or len(rec) < 2:
                continue
            text = str(rec[0]).strip()
            conf = float(rec[1])
            if not text:
                continue
            items.append(OCRItem(text=text, confidence=conf, box=self._coerce_box(box)))
        return items

    @staticmethod
    def _grid_score(grid: Optional[List[List[str]]]) -> int:
        """
        估算 grid “像表格”的程度，用于在多种识别路径间做轻量选择。
        分数越高越可信。
        """
        if not grid or len(grid) < 2:
            return 0

        # rows/cols 基础分
        rows = len(grid)
        cols = max((len(r) for r in grid if r), default=0)
        if cols < 2:
            return 0

        head = "".join(grid[0] or [])
        keywords = ("姓名", "班级", "学号", "科目", "成绩", "分数", "平时", "期中", "期末")
        kw_hits = sum(1 for k in keywords if k in head)

        # 空值过多惩罚：首行空单元格占比过高通常代表结构不稳
        head_cells = grid[0] or []
        empty = sum(1 for c in head_cells if not (c or "").strip())
        empty_penalty = int((empty / max(1, len(head_cells))) * 10)

        return int(kw_hits * 12 + cols * 3 + min(rows, 50) - empty_penalty)

    def recognize_table(
        self,
        image_bgr: np.ndarray,
        *,
        ocr_items: Optional[Sequence[OCRItem]] = None,
        preprocess: bool = True,
    ) -> Optional[List[List[str]]]:
        """
        表格识别：
        - 先尝试基于 OCR 检测框重建二维表格（对“截图类班级表”更快更轻量）
        - 若重建效果不佳，再尝试 PaddleX table_recognition pipeline（若可用）
        """
        if preprocess:
            image_bgr = self.preprocess_image(image_bgr)

        # 1) 轻量路径：OCR 检测框聚类重建表格
        if ocr_items is None:
            ocr_items = self.recognize_text(image_bgr, preprocess=False)

        grid_from_items = self._items_to_grid(ocr_items)
        score_items = self._grid_score(grid_from_items)
        if score_items >= 20:
            return grid_from_items

        # 2) 重路径：PaddleX TableRecognition
        try:
            engine = self._get_table_engine()
            # pipeline.predict 既支持图片路径，也支持 ndarray；这里直接传 ndarray
            pred = engine.predict(image_bgr)
            out = list(pred)
            if out and isinstance(out[0], dict):
                html = out[0].get("html")
                if html:
                    grid_from_html = self._html_table_to_grid(html)
                    score_html = self._grid_score(grid_from_html)
                    if score_html > score_items:
                        return grid_from_html
        except Exception:
            pass

        return grid_from_items if grid_from_items and len(grid_from_items) >= 2 else None

    @staticmethod
    def _coerce_box(raw: Any) -> List[List[float]]:
        """
        将不同来源的 box/poly 统一为 4 点坐标 [[x,y],...]
        支持：
        - [[x,y],...] 或 numpy array
        - [x1,y1,x2,y2]
        """
        if raw is None:
            return []
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if isinstance(raw, list) and raw and isinstance(raw[0], (int, float)) and len(raw) == 4:
            x1, y1, x2, y2 = raw
            return [[float(x1), float(y1)], [float(x2), float(y1)], [float(x2), float(y2)], [float(x1), float(y2)]]
        if isinstance(raw, list) and raw and isinstance(raw[0], (list, tuple)) and len(raw[0]) >= 2:
            pts = [[float(p[0]), float(p[1])] for p in raw[:4]]
            if len(pts) == 4:
                return pts
        return []

    @staticmethod
    def _xyxy(box: List[List[float]]) -> Optional[Tuple[float, float, float, float]]:
        if not box:
            return None
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return min(xs), min(ys), max(xs), max(ys)

    def _items_to_grid(self, items: Sequence[OCRItem]) -> List[List[str]]:
        """
        简易表格重建：按 y 聚类成行，再按 x 聚类成列。
        适用于截图类“有明显网格/对齐”的班级成绩表。
        """
        enriched = []
        for it in items:
            bb = self._xyxy(it.box)
            if not bb:
                continue
            x1, y1, x2, y2 = bb
            enriched.append((it, (x1, y1, x2, y2), (x1 + x2) / 2.0, (y1 + y2) / 2.0, (y2 - y1)))
        if not enriched:
            return []

        heights = sorted(h for *_rest, h in enriched)
        med_h = heights[len(heights) // 2] if heights else 12.0
        y_tol = max(8.0, med_h * 0.8)

        enriched.sort(key=lambda t: t[3])  # by y_center
        rows: List[List[Tuple[OCRItem, float, float, float, float]]] = []
        row_centers: List[float] = []
        for it, (x1, y1, x2, y2), xc, yc, _h in enriched:
            placed = False
            for idx, rc in enumerate(row_centers):
                if abs(yc - rc) <= y_tol:
                    rows[idx].append((it, x1, y1, x2, y2))
                    row_centers[idx] = (row_centers[idx] * 0.7) + (yc * 0.3)
                    placed = True
                    break
            if not placed:
                rows.append([(it, x1, y1, x2, y2)])
                row_centers.append(yc)

        for r in rows:
            r.sort(key=lambda t: (t[1] + t[3]) / 2.0)

        # 列中心不要只取表头（表头文字可能漏检），而是用全量检测框的 x 中心做聚类
        x_centers_all = sorted(((x1 + x2) / 2.0) for _it, x1, _y1, x2, _y2 in sum(rows, []))
        widths = sorted((x2 - x1) for _it, x1, _y1, x2, _y2 in sum(rows, []))
        med_w = widths[len(widths) // 2] if widths else 40.0
        x_tol = max(25.0, med_w * 0.6)

        col_centers: List[float] = []
        for xc in x_centers_all:
            if not col_centers:
                col_centers.append(xc)
                continue
            if abs(xc - col_centers[-1]) <= x_tol:
                col_centers[-1] = (col_centers[-1] + xc) / 2.0
            else:
                col_centers.append(xc)

        grid: List[List[str]] = []
        for r in rows:
            line = [""] * len(col_centers)
            for it, x1, y1, x2, y2 in r:
                xc = (x1 + x2) / 2.0
                best_i = min(range(len(col_centers)), key=lambda i: abs(col_centers[i] - xc))
                if line[best_i]:
                    line[best_i] = f"{line[best_i]} {it.text}".strip()
                else:
                    line[best_i] = it.text
            if any(c.strip() for c in line):
                grid.append(line)

        return grid

    @staticmethod
    def _html_table_to_grid(html: str) -> List[List[str]]:
        """
        将 PP-Structure 产出的 table HTML 转换为二维数组（行列）。
        """
        soup = BeautifulSoup(html, "lxml")
        rows: List[List[str]] = []
        for tr in soup.select("tr"):
            cells = tr.select("th,td")
            row = [c.get_text(" ", strip=True) for c in cells]
            if any(cell.strip() for cell in row):
                rows.append(row)
        return rows

    @staticmethod
    def normalize_text(s: str) -> str:
        return re.sub(r"\s+", "", s or "").strip()

    @staticmethod
    def best_match_confidence(query: str, items: Sequence[OCRItem]) -> Optional[float]:
        """
        在 OCR 行结果中，为指定文本找到一个近似匹配的置信度。
        - 优先：归一化后完全匹配
        - 其次：数字（例如 95、95.0）值相等匹配
        """
        qn = OCRService.normalize_text(query)
        if not qn:
            return None

        # 完全匹配（去空白）
        for it in items:
            if OCRService.normalize_text(it.text) == qn:
                return float(it.confidence)

        # 数字匹配：将文本抽取为数字后比较
        qnum = OCRService._extract_number(qn)
        if qnum is None:
            return None

        best: Optional[float] = None
        for it in items:
            inum = OCRService._extract_number(OCRService.normalize_text(it.text))
            if inum is None:
                continue
            if abs(inum - qnum) < 1e-6:
                best = max(best or 0.0, float(it.confidence))
        return best

    @staticmethod
    def _extract_number(s: str) -> Optional[float]:
        m = re.search(r"(\d+(?:\.\d+)?)", s or "")
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None


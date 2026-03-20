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
        # 旋转纠偏（deskew）参数
        self._deskew_max_angle = 30.0
        self._deskew_min_abs_angle = 0.2
        self._deskew_hough_min_lines = 6
        self._deskew_projection_step = 0.5
        self._deskew_estimate_max_side = 1200
        self._deskew_pad = 20
        # 标题/表格区域处理
        self._table_roi_min_ratio = 0.55
        self._title_max_rows = 3
        self._title_max_chars = 12
        self._title_min_width_ratio = 0.35
        self._document_min_area_ratio = 0.35
        self._cell_min_size = 12

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
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("无法解码图片：请确认文件为 PNG/JPG")
        return img

    def preprocess_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        OCR 预处理：
        - 对较小截图加白边，避免边缘文字被截断/漏检
        - 对较小截图放大，提高字符可分辨率（如 99->9 的问题）
        - 对轻微倾斜做纠偏，提升表格与文本稳定性
        """
        return self.preprocess_with_context(image_bgr)["image"]

    def preprocess_with_context(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        if image_bgr is None:
            return {"image": image_bgr, "context": None}

        alpha = None
        if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 4:
            alpha = image_bgr[:, :, 3].copy()

        image_bgr = self._normalize_input_image(image_bgr)
        original_h, original_w = image_bgr.shape[:2]
        transform = self._identity_transform()

        # 透明背景截图大多只是旋转，不是透视拍照；此时跳过透视校正，避免反投影失真。
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

        processed_h, processed_w = image_bgr.shape[:2]
        context = {
            "forward_matrix": transform,
            "inverse_matrix": np.linalg.inv(transform),
            "original_size": {"width": int(original_w), "height": int(original_h)},
            "processed_size": {"width": int(processed_w), "height": int(processed_h)},
        }
        return {"image": image_bgr, "context": context}

    def debug_preprocess(self, image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        if image_bgr is None:
            return {}

        stages: Dict[str, np.ndarray] = {}
        normalized = self._normalize_input_image(image_bgr)
        stages["01_normalized"] = normalized.copy()

        corrected = normalized
        quad = self._detect_document_quad(normalized)
        if quad is not None:
            corrected = self._perspective_correct(normalized, quad)
        stages["02_perspective"] = corrected.copy()

        deskewed = self._deskew_rotate(corrected)
        stages["03_deskew"] = deskewed.copy()

        cropped = self._crop_to_table_or_title(deskewed)
        stages["04_cropped"] = cropped.copy()

        enhanced = self._enhance_for_ocr(cropped)
        stages["05_enhanced"] = enhanced.copy()

        try:
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            stages["06_table_binary"] = self._binarize_for_table(gray)
        except Exception:
            pass

        return stages

    @staticmethod
    def _identity_transform() -> np.ndarray:
        return np.eye(3, dtype=np.float32)

    def _normalize_input_image(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None:
            return image_bgr
        if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 4:
            bgr = image_bgr[:, :, :3]
            alpha = image_bgr[:, :, 3]
            return self._alpha_composite_white(bgr, alpha)
        if len(image_bgr.shape) == 2:
            return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        return image_bgr

    def _deskew_rotate(self, image_bgr: np.ndarray) -> np.ndarray:
        image_bgr, _matrix = self._deskew_rotate_with_matrix(image_bgr)
        return image_bgr

    def _deskew_rotate_with_matrix(
        self, image_bgr: np.ndarray, alpha_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        transform = self._identity_transform()
        angle = None

        if alpha_mask is not None:
            angle = self._estimate_skew_min_area(alpha_mask)
            if angle is not None and abs(angle) < self._deskew_min_abs_angle:
                angle = None

        paper_mask = self._extract_paper_mask(image_bgr) if alpha_mask is None else None
        if angle is None and paper_mask is not None:
            angle = self._estimate_skew_paper(paper_mask)
            if angle is not None and abs(angle) < self._deskew_min_abs_angle:
                angle = None
        if angle is None:
            angle = self._estimate_skew_angle(image_bgr)
        if angle is not None and abs(angle) >= self._deskew_min_abs_angle:
            image_bgr, rotate_matrix = self._rotate_with_padding_with_matrix(
                image_bgr, angle, pad=self._deskew_pad
            )
            transform = rotate_matrix @ transform
            paper_mask_after = self._extract_paper_mask(image_bgr)
            crop_bbox = self._safe_crop_bbox_from_mask(image_bgr, paper_mask_after)
            if crop_bbox is not None:
                image_bgr, crop_matrix = self._crop_by_bbox(image_bgr, crop_bbox)
                transform = crop_matrix @ transform
        return image_bgr, transform

    def _enhance_for_ocr(self, image_bgr: np.ndarray) -> np.ndarray:
        image_bgr, _matrix = self._enhance_for_ocr_with_matrix(image_bgr)
        return image_bgr

    def _enhance_for_ocr_with_matrix(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image_bgr.shape[:2]
        if min(h, w) >= self._small_image_min_side:
            return image_bgr, self._identity_transform()
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
        matrix = np.array(
            [
                [self._small_image_scale, 0, self._small_image_scale * self._small_image_pad],
                [0, self._small_image_scale, self._small_image_scale * self._small_image_pad],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        return scaled, matrix

    def _detect_document_quad(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        h, w = gray.shape[:2]
        min_area = h * w * self._document_min_area_ratio
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            area = cv2.contourArea(approx)
            if area < min_area:
                continue
            return approx.reshape(4, 2).astype(np.float32)
        return None

    @staticmethod
    def _order_quad_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _perspective_correct(self, image_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
        image_bgr, _matrix = self._perspective_correct_with_matrix(image_bgr, quad)
        return image_bgr

    def _perspective_correct_with_matrix(
        self, image_bgr: np.ndarray, quad: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        rect = self._order_quad_points(quad)
        tl, tr, br, bl = rect
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_width = int(max(width_a, width_b))
        max_height = int(max(height_a, height_b))
        if max_width < 50 or max_height < 50:
            return image_bgr

        dst = np.array(
            [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        corrected = cv2.warpPerspective(
            image_bgr,
            matrix,
            (max_width, max_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return corrected, matrix

    def _estimate_skew_angle(self, image_bgr: np.ndarray) -> Optional[float]:
        """
        估计小角度倾斜（单位：度，正值为逆时针）。
        先尝试 Hough 直线，失败则用投影法兜底。
        """
        if image_bgr is None:
            return None

        # 降采样以加速估计
        h, w = image_bgr.shape[:2]
        scale = 1.0
        max_side = max(h, w)
        if max_side > self._deskew_estimate_max_side:
            scale = self._deskew_estimate_max_side / max_side
            image_small = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            image_small = image_bgr

        angle = self._estimate_skew_hough(image_small)
        if angle is not None:
            return float(angle)

        angle = self._estimate_skew_projection(image_small)
        return float(angle) if angle is not None else None

    def _estimate_skew_paper(self, mask: np.ndarray) -> Optional[float]:
        """
        基于纸面区域的最小外接矩形估计倾斜角。
        """
        if mask is None or len(mask.shape) != 2:
            return None

        ys, xs = np.where(mask > 0)
        if len(xs) < 200:
            return None

        coords = np.column_stack((xs, ys)).astype(np.float32)
        rect = cv2.minAreaRect(coords)
        angle = rect[2]
        width, height = rect[1]
        if width == 0 or height == 0:
            return None

        if width < height:
            angle = angle + 90.0

        if angle < -90:
            angle += 180
        if angle > 90:
            angle -= 180

        if abs(angle) > self._deskew_max_angle:
            return None

        return float(angle)

    def _estimate_skew_min_area(self, alpha: np.ndarray) -> Optional[float]:
        """
        通过 alpha 掩膜的最小外接矩形估计倾斜角。
        对透明背景截图更稳定。
        """
        if alpha is None:
            return None

        if len(alpha.shape) != 2:
            return None

        ys, xs = np.where(alpha > 0)
        if len(xs) < 50:
            return None

        coords = np.column_stack((xs, ys)).astype(np.float32)
        rect = cv2.minAreaRect(coords)
        angle = rect[2]
        width, height = rect[1]
        if width == 0 or height == 0:
            return None

        # OpenCV 角度范围为 [-90, 0)
        if width < height:
            angle = angle + 90.0

        # 归一化到 [-90, 90]
        if angle < -90:
            angle += 180
        if angle > 90:
            angle -= 180

        if abs(angle) > self._deskew_max_angle:
            return None

        return float(angle)

    def _estimate_skew_hough(self, image_bgr: np.ndarray) -> Optional[float]:
        """
        基于 Hough 直线的倾斜估计，偏向表格横线/文字行。
        """
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None

        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120, minLineLength=80, maxLineGap=10)
        if lines is None:
            return None

        angles: List[float] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            angle = np.degrees(np.arctan2(dy, dx))
            # 归一化到 [-90, 90]
            if angle < -90:
                angle += 180
            if angle > 90:
                angle -= 180
            # 只取接近水平的线，避免竖线干扰
            if abs(angle) <= self._deskew_max_angle:
                angles.append(angle)

        if len(angles) < self._deskew_hough_min_lines:
            return None

        return float(np.median(angles))

    def _estimate_skew_projection(self, image_bgr: np.ndarray) -> Optional[float]:
        """
        基于水平投影方差的倾斜估计（小角度扫描）。
        """
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None

        # 自适应二值化，文本为白色
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        best_angle = None
        best_score = -1.0
        step = self._deskew_projection_step
        max_angle = self._deskew_max_angle

        for angle in np.arange(-max_angle, max_angle + 1e-6, step):
            rotated = self._rotate_keep_size(bw, angle)
            # 计算水平投影方差
            proj = np.sum(rotated > 0, axis=1)
            score = float(np.var(proj))
            if score > best_score:
                best_score = score
                best_angle = angle

        return float(best_angle) if best_angle is not None else None

    @staticmethod
    def _extract_paper_mask(image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        估计“纸面区域”掩膜：低饱和、高亮度。
        对透明棋盘背景或灰底更稳。
        """
        if image_bgr is None:
            return None

        try:
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        except Exception:
            return None

        # 选择近白色区域（高亮度、低饱和），并通过形态学去掉棋盘噪声
        v = hsv[:, :, 2]
        v_blur = cv2.GaussianBlur(v, (5, 5), 0)
        mask = cv2.inRange(hsv, (0, 0, 235), (179, 60, 255))
        mask = cv2.bitwise_and(mask, cv2.threshold(v_blur, 235, 255, cv2.THRESH_BINARY)[1])
        if mask is None:
            return None

        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 2000:
            return mask

        paper = np.zeros_like(mask)
        cv2.drawContours(paper, [largest], -1, 255, thickness=-1)
        return paper

    @staticmethod
    def _crop_to_paper(image_bgr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        crop_bbox = OCRService._safe_crop_bbox_from_mask(image_bgr, mask)
        if crop_bbox is None:
            return image_bgr
        cropped, _matrix = OCRService._crop_by_bbox(image_bgr, crop_bbox)
        return cropped

    @staticmethod
    def _safe_crop_bbox_from_mask(
        image_bgr: np.ndarray, mask: Optional[np.ndarray]
    ) -> Optional[Tuple[int, int, int, int]]:
        if image_bgr is None or mask is None:
            return None

        ys, xs = np.where(mask > 0)
        if len(xs) < 200:
            return None

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        h, w = image_bgr.shape[:2]
        box_w = x2 - x1 + 1
        box_h = y2 - y1 + 1
        if box_w <= 0 or box_h <= 0:
            return None

        # 过于激进的裁切容易丢列，设置保守阈值
        if (box_w / w) < 0.8 or (box_h / h) < 0.8 or (box_w * box_h) < (0.6 * w * h):
            return None

        pad = 30
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad)
        y2 = min(h - 1, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _crop_by_bbox(
        image_bgr: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = bbox
        cropped = image_bgr[y1 : y2 + 1, x1 : x2 + 1]
        matrix = np.array([[1, 0, -x1], [0, 1, -y1], [0, 0, 1]], dtype=np.float32)
        return cropped, matrix

    def _crop_to_table_or_title(self, image_bgr: np.ndarray) -> np.ndarray:
        image_bgr, _matrix = self._crop_to_table_or_title_with_matrix(image_bgr)
        return image_bgr

    def _crop_to_table_or_title_with_matrix(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        优先裁切表格区域；若无法检测表格区域，则尝试裁切掉顶部标题。
        """
        if image_bgr is None:
            return image_bgr, self._identity_transform()

        # 1) 表格区域（基于水平/垂直线的交叉区域）
        table_box = self._estimate_table_bbox(image_bgr)
        if table_box is not None:
            return self._crop_by_bbox(image_bgr, table_box)

        # 2) 标题裁切（基于文本行高度/位置）
        title_crop = self._estimate_title_crop_y(image_bgr)
        if title_crop is not None:
            h, w = image_bgr.shape[:2]
            y = min(max(0, title_crop), h - 1)
            return self._crop_by_bbox(image_bgr, (0, y, w - 1, h - 1))

        return image_bgr, self._identity_transform()

    def _estimate_table_bbox(self, image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        基于表格线检测估计表格区域边界。
        """
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None

        # 二值化强调表格线
        bw = self._binarize_for_table(gray)
        h, w = bw.shape[:2]

        # 检测水平/垂直线
        _horiz, _vert, grid = self._extract_grid_lines(bw)

        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # 取最大区域
        x, y, ww, hh = cv2.boundingRect(max(contours, key=cv2.contourArea))
        if ww * hh < (self._table_roi_min_ratio * w * h):
            return None

        pad = 6
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w - 1, x + ww + pad)
        y2 = min(h - 1, y + hh + pad)
        return x1, y1, x2, y2

    @staticmethod
    def _binarize_for_table(gray: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
        )

    def _extract_grid_lines(self, binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = binary.shape[:2]
        horiz = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 25), 1)),
        )
        vert = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 25))),
        )
        horiz = self._filter_line_components(
            horiz,
            axis="horizontal",
            min_long_side=max(20, int(w * 0.18)),
            max_short_side=max(6, h // 12),
        )
        vert = self._filter_line_components(
            vert,
            axis="vertical",
            min_long_side=max(28, int(h * 0.35)),
            max_short_side=max(6, w // 12),
        )
        grid = cv2.bitwise_or(horiz, vert)
        return horiz, vert, grid

    @staticmethod
    def _filter_line_components(
        mask: np.ndarray, *, axis: str, min_long_side: int, max_short_side: int
    ) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask

        filtered = np.zeros_like(mask)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            long_side = w if axis == "horizontal" else h
            short_side = h if axis == "horizontal" else w
            if long_side < min_long_side:
                continue
            if short_side > max_short_side:
                continue
            cv2.drawContours(filtered, [contour], -1, 255, thickness=-1)
        return filtered

    def _grid_mask_to_cells(self, grid_mask: np.ndarray) -> Optional[List[Tuple[int, int, int, int]]]:
        inv = cv2.bitwise_not(grid_mask)
        contours, hierarchy = cv2.findContours(inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours is None or hierarchy is None:
            return None

        h, w = grid_mask.shape[:2]
        cells: List[Tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, ww, hh = cv2.boundingRect(contour)
            if ww < self._cell_min_size or hh < self._cell_min_size:
                continue
            if ww > w * 0.95 and hh > h * 0.95:
                continue
            if ww * hh > (w * h * 0.35):
                continue
            cells.append((x, y, ww, hh))

        return cells if len(cells) >= 4 else None

    def _line_cells_to_grid(
        self, cells: Sequence[Tuple[int, int, int, int]], items: Sequence[OCRItem]
    ) -> Optional[List[List[str]]]:
        if not cells:
            return None

        sorted_cells = sorted(cells, key=lambda c: (c[1], c[0]))
        heights = sorted(c[3] for c in sorted_cells)
        med_h = heights[len(heights) // 2] if heights else 20
        row_tol = max(10.0, med_h * 0.6)

        row_centers: List[float] = []
        rows: List[List[Tuple[int, int, int, int]]] = []
        for cell in sorted_cells:
            x, y, ww, hh = cell
            yc = y + hh / 2.0
            placed = False
            for idx, center in enumerate(row_centers):
                if abs(yc - center) <= row_tol:
                    rows[idx].append(cell)
                    row_centers[idx] = (row_centers[idx] * 0.7) + (yc * 0.3)
                    placed = True
                    break
            if not placed:
                row_centers.append(yc)
                rows.append([cell])

        for row in rows:
            row.sort(key=lambda c: c[0])

        if len(rows) < 2 or max((len(row) for row in rows), default=0) < 2:
            return None

        grid: List[List[str]] = []
        for row in rows:
            line: List[str] = []
            for x, y, ww, hh in row:
                line.append(self._collect_text_in_rect(items, x, y, x + ww, y + hh))
            if any(cell.strip() for cell in line):
                grid.append(line)

        return grid if len(grid) >= 2 else None

    @staticmethod
    def _line_dominance(mask: np.ndarray, *, axis: str) -> float:
        if mask is None or mask.size == 0:
            return 0.0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        h, w = mask.shape[:2]
        target_span = h if axis == "vertical" else w
        spans: List[float] = []
        for contour in contours:
            x, y, ww, hh = cv2.boundingRect(contour)
            span = hh if axis == "vertical" else ww
            spans.append(span / max(1.0, float(target_span)))
        return float(max(spans)) if spans else 0.0

    def _collect_text_in_rect(
        self, items: Sequence[OCRItem], x1: int, y1: int, x2: int, y2: int
    ) -> str:
        matches: List[Tuple[float, str]] = []
        for item in items:
            bb = self._xyxy(item.box)
            if not bb:
                continue
            ix1, iy1, ix2, iy2 = bb
            cx = (ix1 + ix2) / 2.0
            cy = (iy1 + iy2) / 2.0
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                matches.append((cx, item.text))

        matches.sort(key=lambda t: t[0])
        return " ".join(text.strip() for _, text in matches if text.strip()).strip()

    def _estimate_title_crop_y(self, image_bgr: np.ndarray) -> Optional[int]:
        """
        尝试定位顶部标题行并给出裁切 y 坐标（裁掉标题）。
        """
        try:
            ocr = self._get_ocr()
            results = ocr.ocr(image_bgr, cls=True)
        except Exception:
            return None

        items: List[OCRItem] = []
        if not results:
            return None

        # 兼容 PaddleOCR 2.x/3.x
        if isinstance(results, list) and results and isinstance(results[0], dict):
            d = results[0]
            texts = d.get("rec_texts") or []
            scores = d.get("rec_scores") or []
            polys = d.get("rec_polys") or d.get("dt_polys") or []
            if hasattr(polys, "tolist"):
                polys = polys.tolist()
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            for i, text in enumerate(texts):
                t = str(text).strip()
                if not t:
                    continue
                box = self._coerce_box(polys[i] if i < len(polys) else None)
                items.append(OCRItem(text=t, confidence=float(scores[i] if i < len(scores) else 0), box=box))
        elif isinstance(results, list) and results and isinstance(results[0], list):
            for line in results[0] if isinstance(results, list) else results:
                if not line or len(line) < 2:
                    continue
                box, rec = line[0], line[1]
                if not rec or len(rec) < 2:
                    continue
                text = str(rec[0]).strip()
                if not text:
                    continue
                items.append(OCRItem(text=text, confidence=float(rec[1]), box=self._coerce_box(box)))
        else:
            # 兼容 predict 返回非标准结构时，放弃标题裁切
            return None

        # 根据行高排序，优先选择顶部大字号行
        candidates = []
        for it in items:
            bb = self._xyxy(it.box)
            if not bb:
                continue
            x1, y1, x2, y2 = bb
            w = x2 - x1
            h = y2 - y1
            if h <= 0:
                continue
            candidates.append((y1, y2, w, h, it.text))

        if not candidates:
            return None

        # 选择顶部若干行中，宽度占比较高、字符数较少的行作为标题
        candidates.sort(key=lambda x: x[0])
        h_img, w_img = image_bgr.shape[:2]
        top_rows = candidates[: self._title_max_rows]
        # 使用中位行高估计标题行的大字特征
        heights = [c[3] for c in top_rows]
        med_h = sorted(heights)[len(heights) // 2] if heights else 0
        for y1, y2, w, h, text in top_rows:
            if (w / w_img) >= self._title_min_width_ratio and len(text) <= self._title_max_chars:
                if med_h == 0 or h >= med_h * 1.1:
                    return int(y2 + 6)

        return None

    @staticmethod
    def _rotate_keep_size(image: np.ndarray, angle: float) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0 if len(image.shape) == 2 else 255),
        )

    @staticmethod
    def _rotate_with_padding(image: np.ndarray, angle: float, pad: int = 20) -> np.ndarray:
        rotated, _matrix = OCRService._rotate_with_padding_with_matrix(image, angle, pad)
        return rotated

    @staticmethod
    def _rotate_with_padding_with_matrix(
        image: np.ndarray, angle: float, pad: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        padded = cv2.copyMakeBorder(
            image,
            pad,
            pad,
            pad,
            pad,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        h, w = padded.shape[:2]
        center = (w / 2.0, h / 2.0)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            padded,
            m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(180, 180, 180),
        )
        pad_matrix = np.array([[1, 0, pad], [0, 1, pad], [0, 0, 1]], dtype=np.float32)
        rotate_matrix = np.vstack([m, [0, 0, 1]]).astype(np.float32)
        return rotated, rotate_matrix @ pad_matrix

    @staticmethod
    def _alpha_composite_white(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        将带 alpha 的图片合成到白色背景，避免透明网格干扰。
        """
        a = (alpha.astype(np.float32) / 255.0)[..., None]
        white = np.ones_like(bgr, dtype=np.float32) * 255.0
        out = (bgr.astype(np.float32) * a + white * (1.0 - a)).astype(np.uint8)
        return out

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

        # 1) 截图类场景优先走 OCR 聚类；只有表格线贯穿度足够高时才启用线结构恢复
        if ocr_items is None:
            ocr_items = self.recognize_text(image_bgr, preprocess=False)

        grid_from_items = self._items_to_grid(ocr_items)
        score_items = self._grid_score(grid_from_items)

        grid_from_lines = None
        score_lines = 0
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            bw = self._binarize_for_table(gray)
            horiz, vert, grid_mask = self._extract_grid_lines(bw)
            vertical_dominance = self._line_dominance(vert, axis="vertical")
            horizontal_dominance = self._line_dominance(horiz, axis="horizontal")
            line_confident = vertical_dominance >= 0.65 and horizontal_dominance >= 0.45
            cells = self._grid_mask_to_cells(grid_mask) if line_confident else None
            if cells and score_items < 40:
                grid_from_lines = self._line_cells_to_grid(cells, ocr_items)
                score_lines = self._grid_score(grid_from_lines)
                if score_lines > score_items and score_lines >= 20:
                    return grid_from_lines
        except Exception:
            grid_from_lines = None
            score_lines = 0

        # 2) 轻量路径：OCR 检测框聚类重建表格
        if score_items >= 20:
            return grid_from_items

        # 3) 重路径：PaddleX TableRecognition
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
                    if score_html > max(score_items, score_lines):
                        return grid_from_html
        except Exception:
            pass

        best_grid = grid_from_items if score_items >= score_lines else grid_from_lines
        return best_grid if best_grid and len(best_grid) >= 2 else None

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
    def best_match_box(query: str, items: Sequence[OCRItem]) -> Optional[List[List[float]]]:
        qn = OCRService.normalize_text(query)
        if not qn:
            return None

        exact_match: Optional[OCRItem] = None
        for it in items:
            if OCRService.normalize_text(it.text) == qn:
                if exact_match is None or it.confidence > exact_match.confidence:
                    exact_match = it
        if exact_match is not None:
            return exact_match.box

        qnum = OCRService._extract_number(qn)
        if qnum is None:
            return None

        numeric_match: Optional[OCRItem] = None
        for it in items:
            inum = OCRService._extract_number(OCRService.normalize_text(it.text))
            if inum is None:
                continue
            if abs(inum - qnum) < 1e-6:
                if numeric_match is None or it.confidence > numeric_match.confidence:
                    numeric_match = it
        return numeric_match.box if numeric_match is not None else None

    @staticmethod
    def _extract_number(s: str) -> Optional[float]:
        m = re.search(r"(\d+(?:\.\d+)?)", s or "")
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None


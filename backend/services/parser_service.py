from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import HTTPException
from openpyxl import load_workbook

from backend.services.ocr_service import OCRItem, OCRService


class FileParserService:
    """
    文件解析统一入口：
    - 支持班级成绩表解析（动态表头，多行数据）
    - PNG/JPG：PP-Structure 表格识别 + 全图 OCR 兜底
    - SVG：优先解析 <text> 文本节点；失败则转图片 OCR
    - XLSX：直接读取表格数据
    输出结构：{"headers": [...], "rows": [{"values": {...}, "confidences": {...}}, ...]}
    """

    def __init__(self, *, ocr_service: OCRService) -> None:
        self.ocr_service = ocr_service

    def parse(self, *, filename: str, content_type: str, data: bytes) -> Dict[str, Any]:
        ext = self._guess_ext(filename, content_type, data)
        if ext in {".png", ".jpg", ".jpeg"}:
            return self._parse_image(data)
        if ext == ".svg":
            return self._parse_svg(data)
        if ext == ".xlsx":
            return self._parse_xlsx(data)
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{ext}（仅支持 PNG/JPG/SVG/XLSX）")

    @staticmethod
    def _guess_ext(filename: str, content_type: str, data: bytes) -> str:
        name = (filename or "").lower()
        for ext in (".png", ".jpg", ".jpeg", ".svg", ".xlsx"):
            if name.endswith(ext):
                return ext
        ct = (content_type or "").lower()
        if "svg" in ct:
            return ".svg"
        if "excel" in ct or "spreadsheetml" in ct:
            return ".xlsx"
        if "png" in ct:
            return ".png"
        if "jpeg" in ct or "jpg" in ct:
            return ".jpg"
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if data[:3] == b"\xff\xd8\xff":
            return ".jpg"
        if data.startswith(b"PK\x03\x04"):
            return ".xlsx"
        head = data[:500].lstrip().lower()
        if head.startswith(b"<svg") or b"<svg" in head:
            return ".svg"
        return ""

    def _parse_image(self, data: bytes) -> Dict[str, Any]:
        image_bgr = self.ocr_service.decode_image(data)
        image_bgr = self.ocr_service.preprocess_image(image_bgr)

        # 先做一次文本 OCR（后续置信度匹配、表格重建都复用）
        ocr_items = self.ocr_service.recognize_text(image_bgr, preprocess=False)

        table = None
        try:
            table = self.ocr_service.recognize_table(image_bgr, ocr_items=ocr_items, preprocess=False)
        except Exception:
            table = None

        if table and len(table) >= 2:
            return self._extract_class_grid(table, ocr_items=ocr_items, default_conf=0.85)

        # 兜底：无法识别表格结构时，返回单列文本列表
        return self._extract_from_text_lines_fallback([it.text for it in ocr_items])

    def _parse_svg(self, data: bytes) -> Dict[str, Any]:
        # 1) 优先解析 SVG 文本节点
        try:
            texts = self._extract_svg_text_nodes(data)
        except Exception:
            texts = []

        if texts:
             # SVG 文本通常是散乱的，很难直接构建二维表格
             # 这里简单处理：转为单列数据
             return self._extract_from_text_lines_fallback(texts)

        # 2) 兜底：SVG 转图片再 OCR
        try:
            import cairosvg
            png_bytes = cairosvg.svg2png(bytestring=data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"SVG 解析失败且无法转图片：{e}") from e

        return self._parse_image(png_bytes)

    def _parse_xlsx(self, data: bytes) -> Dict[str, Any]:
        try:
            wb = load_workbook(io.BytesIO(data), data_only=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"XLSX 读取失败：{e}") from e

        ws = wb.active
        rows = []
        for r in ws.iter_rows(values_only=True):
            row = [("" if v is None else str(v).strip()) for v in r]
            # 只要这一行有内容，就保留（哪怕是空行中间的）
            if any(c for c in row):
                rows.append(row)
        
        return self._extract_class_grid(rows, ocr_items=[], default_conf=1.0)

    @staticmethod
    def _extract_svg_text_nodes(data: bytes) -> List[str]:
        try:
            root = ET.fromstring(data.decode("utf-8", errors="ignore"))
        except Exception:
            root = ET.fromstring(data)

        out: List[str] = []
        for el in root.iter():
            tag = el.tag.split("}")[-1].lower()
            if tag == "text":
                txt = " ".join(t.strip() for t in el.itertext() if t and t.strip()).strip()
                if txt:
                    out.append(txt)
        return out

    def _extract_class_grid(
        self,
        grid: List[List[str]],
        *,
        ocr_items: Sequence[OCRItem],
        default_conf: float,
    ) -> Dict[str, Any]:
        """
        通用表格解析逻辑：
        - 第一行非空行作为表头（Headers）
        - 后续行作为数据（Rows）
        - 自动转换数字
        - 匹配置信度
        """
        if not grid:
            return {"headers": [], "rows": []}

        # 1. 寻找表头行（假设第一行就是，或者根据关键词探测）
        header_row_idx = 0
        title_text: Optional[str] = None
        for i, row in enumerate(grid[:5]):
            joined = "".join(row)
            non_empty = sum(1 for c in row if (c or "").strip())
            # 标题行通常只有 1 个非空单元格，且包含“成绩表/成绩单”
            if non_empty <= 2 and any(k in joined for k in ("成绩表", "成绩单")):
                title_text = joined.strip()
                continue
            # 如果包含常见表头关键词，可能是表头
            if any(k in joined for k in ("姓名", "班级", "学号", "科目", "成绩", "分数", "等级")):
                header_row_idx = i
                break

        # 若第一行是标题（单格居中），而第二行更像表头，则跳过标题
        if len(grid) >= 2:
            first = grid[header_row_idx]
            second = grid[header_row_idx + 1] if header_row_idx + 1 < len(grid) else []
            first_non_empty = sum(1 for c in first if (c or "").strip())
            second_non_empty = sum(1 for c in second if (c or "").strip())
            first_joined = "".join(first)
            second_joined = "".join(second)
            if (
                first_non_empty <= 2
                and (any(k in first_joined for k in ("成绩表", "成绩单")) or first_non_empty == 1)
                and second_non_empty >= 2
                and any(k in second_joined for k in ("姓名", "班级", "学号", "科目", "成绩", "分数", "等级"))
            ):
                title_text = first_joined.strip() if first_joined.strip() else title_text
                header_row_idx = header_row_idx + 1
        
        headers_raw = grid[header_row_idx]
        # 清洗表头：去除空白，保留原始文本
        headers = [h.strip() for h in headers_raw]

        # 表头可能存在漏检/空白：为保证列对齐，补全为空表头列命名
        for i in range(len(headers)):
            if not headers[i]:
                headers[i] = f"列{i+1}"

        # 避免表头重复导致覆盖：追加序号确保唯一
        seen: Dict[str, int] = {}
        for i, h in enumerate(headers):
            if h not in seen:
                seen[h] = 1
                continue
            seen[h] += 1
            headers[i] = f"{h}_{seen[h]}"
        
        # 2. 提取数据行
        rows_out = []
        start_row = header_row_idx + 1
        
        for r_idx in range(start_row, len(grid)):
            row_raw = grid[r_idx]
            # 补齐或截断
            if len(row_raw) < len(headers):
                row_raw += [""] * (len(headers) - len(row_raw))
            else:
                row_raw = row_raw[:len(headers)]
            
            # 如果全空则跳过
            if not any(c.strip() for c in row_raw):
                continue
                
            values = {}
            confidences = {}
            
            for c_idx, cell_val in enumerate(row_raw):
                key = headers[c_idx]
                
                # 尝试转数字
                val_clean = cell_val.strip()
                val_num = self._parse_score(val_clean)
                final_val = val_num if val_num is not None else val_clean
                
                values[key] = final_val
                
                # 计算置信度
                conf = default_conf
                if ocr_items:
                    # 尝试匹配文本内容获取置信度
                    matched = OCRService.best_match_confidence(val_clean, ocr_items)
                    if matched is not None:
                        conf = matched
                
                confidences[key] = round(float(conf), 4)
            
            rows_out.append({"values": values, "confidences": confidences})
            
        # 3. 剔除“全空的补位列”（常见于合并单元格导致的空列）
        if rows_out:
            non_empty_counts: Dict[str, int] = {h: 0 for h in headers}
            for row in rows_out:
                vals = row.get("values", {})
                for h in headers:
                    v = vals.get(h, "")
                    if str(v).strip() != "":
                        non_empty_counts[h] += 1

            threshold = max(1, int(len(rows_out) * 0.05))
            drop_headers = {
                h for h in headers
                if re.match(r"^列\d+$", h) and non_empty_counts.get(h, 0) <= threshold
            }
            if drop_headers:
                headers = [h for h in headers if h not in drop_headers]
                for row in rows_out:
                    row["values"] = {k: v for k, v in row["values"].items() if k in headers}
                    row["confidences"] = {k: v for k, v in row["confidences"].items() if k in headers}

        result: Dict[str, Any] = {
            "headers": headers,
            "rows": rows_out,
        }
        if title_text:
            result["title"] = title_text
        return result

    def _extract_from_text_lines_fallback(self, lines: List[str]) -> Dict[str, Any]:
        """
        当无法识别表格结构时，返回一个简单的单列数据
        """
        rows = []
        for line in lines:
            if not line.strip():
                continue
            rows.append({
                "values": {"内容": line.strip()},
                "confidences": {"内容": 0.8} # 默认低置信度提示用户检查
            })
        return {
            "headers": ["内容"],
            "rows": rows
        }

    @staticmethod
    def _parse_score(v: Any) -> Optional[int | float]:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return int(v) if isinstance(v, int) or abs(float(v) - int(float(v))) < 1e-9 else float(v)

        s = str(v).strip()
        # 仅匹配纯数字，或者带小数点的数字
        if not re.match(r"^-?\d+(\.\d+)?$", s):
            return None
            
        try:
            num = float(s)
            return int(num) if abs(num - int(num)) < 1e-9 else round(num, 2)
        except Exception:
            return None

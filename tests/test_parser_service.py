import unittest

import cv2
import numpy as np

from backend.services.ocr_service import OCRService
from backend.services.ocr_service import OCRItem
from backend.services.parser_service import FileParserService


class DummyOCRService:
    pass


class ParserServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = FileParserService(ocr_service=DummyOCRService())
        self.ocr = OCRService()

    def test_normalize_header_name(self):
        self.assertEqual(self.parser._normalize_header_name("课程名称"), "课程")
        self.assertEqual(self.parser._normalize_header_name("总评成绩"), "成绩")

    def test_extract_class_grid_with_title_and_alias_headers(self):
        grid = [
            ["学生成绩单", "", ""],
            ["姓名", "课程名称", "总评成绩"],
            ["张三", "高等数学", "95"],
        ]

        result = self.parser._extract_class_grid(grid, ocr_items=[], default_conf=1.0, image_size=None)
        self.assertEqual(result["title"], "学生成绩单")
        self.assertEqual(result["headers"], ["姓名", "课程", "成绩"])
        self.assertEqual(result["rows"][0]["values"]["成绩"], 95)

    def test_extract_transcript_meta(self):
        class Item:
            def __init__(self, text: str) -> None:
                self.text = text

        meta = self.parser._extract_transcript_meta(
            [
                Item("学生成绩单"),
                Item("姓名：张三"),
                Item("学号：20240001"),
                Item("班级：空间1班"),
            ]
        )
        self.assertEqual(meta["标题"], "学生成绩单")
        self.assertEqual(meta["姓名"], "张三")
        self.assertEqual(meta["学号"], "20240001")

    def test_filter_short_vertical_line_component(self):
        mask = np.zeros((120, 240), dtype=np.uint8)
        cv2.line(mask, (20, 0), (20, 119), 255, 2)
        cv2.line(mask, (110, 25), (110, 55), 255, 2)

        filtered = self.ocr._filter_line_components(
            mask,
            axis="vertical",
            min_long_side=40,
            max_short_side=8,
        )

        self.assertEqual(int(filtered[40, 110]), 0)
        self.assertEqual(int(filtered[60, 20]), 255)

    def test_recognize_table_prefers_ocr_grid_for_screenshot_like_layout(self):
        items = [
            OCRItem("姓名", 0.99, [[10, 10], [60, 10], [60, 30], [10, 30]]),
            OCRItem("班级", 0.99, [[80, 10], [140, 10], [140, 30], [80, 30]]),
            OCRItem("平时1", 0.99, [[160, 10], [230, 10], [230, 30], [160, 30]]),
            OCRItem("张三", 0.99, [[10, 40], [60, 40], [60, 60], [10, 60]]),
            OCRItem("995392", 0.99, [[80, 40], [150, 40], [150, 60], [80, 60]]),
            OCRItem("61", 0.99, [[180, 40], [210, 40], [210, 60], [180, 60]]),
        ]

        grid = self.ocr._items_to_grid(items)
        self.assertGreaterEqual(self.ocr._grid_score(grid), 20)


if __name__ == "__main__":
    unittest.main()

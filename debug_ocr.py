import sys
from unittest.mock import MagicMock

# 1. Pre-Mock ALL missing modules
sys.modules["cv2"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["paddleocr"] = MagicMock()
sys.modules["bs4"] = MagicMock()
sys.modules["fastapi"] = MagicMock()
sys.modules["openpyxl"] = MagicMock()
sys.modules["openpyxl.load_workbook"] = MagicMock()

# Now imports will work
from backend.services.ocr_service import OCRItem, OCRService
from backend.services.parser_service import FileParserService
import json

# 2. Mock OCRService methods
class MockOCRService(OCRService):
    def __init__(self):
        # Skip super init which might try to load things
        self.use_gpu = False
        self.lang = "ch"
        self._ocr = None
        self._table_engine = None

    def preprocess_image(self, image_bgr):
        return image_bgr

    def decode_image(self, data: bytes):
        return MagicMock()  # Return a mock image object

    def recognize_table(self, image_bgr, *, ocr_items=None, preprocess=True):
        # Return the grid structure as seen in the image
        return [
            ["姓名", "班级", "平时1", "平时2"],
            ["班俊悟", "995392", "61", "99"],
            ["边晓", "717835", "62", "98"],
            ["陈屏", "901489", "63", "97"],
            ["充纳", "513021", "64", "96"],
            ["董虹", "794027", "65", "95"],
        ]

    def recognize_text(self, image_bgr, *, preprocess=True):
        lines = [
            "姓名 班级 平时1 平时2",
            "班俊悟 995392 61 99",
            "边晓 717835 62 98",
            "陈屏 901489 63 97",
            "充纳 513021 64 96",
            "董虹 794027 65 95",
        ]
        return [OCRItem(text=line, confidence=0.99, box=[]) for line in lines]

    @staticmethod
    def best_match_confidence(query, items):
        return 0.99


# 3. Patch Parser Service to use our Mock OCR
def run_test():
    # Patch FileParserService dependencies if needed, but constructor injection is enough
    mock_ocr = MockOCRService()
    parser = FileParserService(ocr_service=mock_ocr)

    print("--- Testing with mocked image (Table Structure) ---")
    try:
        # We simulate a PNG upload
        result = parser.parse(
            filename="test.png",
            content_type="image/png",
            data=b"\x89PNG\r\n\x1a\n_dummy_data",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()

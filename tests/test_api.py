import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app, excel_service


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self._tmpdir = tempfile.TemporaryDirectory()
        excel_service.export_dir = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_export_validation(self):
        response = self.client.post("/export", json={"headers": "bad", "rows": []})
        self.assertEqual(response.status_code, 400)

    def test_export_success(self):
        response = self.client.post(
            "/export",
            json={
                "title": "学生成绩单",
                "headers": ["姓名", "成绩"],
                "rows": [{"values": {"姓名": "张三", "成绩": "95"}, "confidences": {}}],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("download_url", response.json())


if __name__ == "__main__":
    unittest.main()

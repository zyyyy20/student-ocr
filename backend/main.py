from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.services.excel_service import ExcelService
from backend.services.ocr_service import OCRService
from backend.services.parser_service import FileParserService


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
EXPORT_DIR = BASE_DIR / "static" / "exports"


def _ensure_dirs() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


_ensure_dirs()

app = FastAPI(title="学生成绩单自动识别系统", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/downloads", StaticFiles(directory=str(EXPORT_DIR)), name="downloads")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

ocr_service = OCRService()
excel_service = ExcelService(export_dir=EXPORT_DIR)
parser_service = FileParserService(ocr_service=ocr_service)


@app.get("/")
def index() -> FileResponse:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="前端资源不存在：请确认 frontend/index.html 已创建")
    return FileResponse(str(index_file))


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="上传文件为空")

        result = parser_service.parse(
            filename=file.filename or f"upload-{uuid.uuid4().hex}",
            content_type=file.content_type or "",
            data=raw,
        )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败：{e}") from e


@app.post("/export")
def export(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    try:
        filename = excel_service.export_transcript(payload)
        return JSONResponse(content={"download_url": f"/downloads/{filename}"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出失败：{e}") from e


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


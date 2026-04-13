# -*- coding: utf-8 -*-
"""
线上主链路 preprocess_with_context 的六步演示（与 preprocess_stage_snapshots 对齐）。

生成内容（相对仓库根目录）：
- testdemo/output/fig_4_1.png … fig_4_6.png — 各步「处理前 | 处理后」对照（供论文插入）
- testdemo/stages/*.png — 各步结束后的单帧快照（01_normalized … 06_enhanced）

运行（在仓库根目录）：
  python testdemo/run_pipeline_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.ocr_service import OCRService
from tools.build_thesis_case_figures import (
    _pick_primary_image,
    _stages,
    write_chapter4_pair_figures,
)

OUTPUT_DIR = ROOT / "testdemo" / "output"
STAGES_DIR = ROOT / "testdemo" / "stages"

_STAGE_KEYS = (
    "01_normalized",
    "02_paper_crop",
    "03_perspective",
    "04_deskew",
    "05_table_roi",
    "06_enhanced",
)


def main() -> int:
    ocr = OCRService()
    picked = _pick_primary_image(ocr)
    if picked is None:
        print("无可用样例图像，请放置 Snipaste_2026-04-13_10-26-13.png（或回退样例）或调整 tools/build_thesis_case_figures.py 中的路径。")
        return 1

    im, name = picked
    STAGES_DIR.mkdir(parents=True, exist_ok=True)
    st = _stages(ocr, im)
    for key in _STAGE_KEYS:
        arr = st.get(key)
        if arr is not None:
            cv2.imwrite(str(STAGES_DIR / f"{key}.png"), arr)
    print("已写入阶段快照:", STAGES_DIR, "主案例:", name)

    n = write_chapter4_pair_figures(ocr, OUTPUT_DIR)
    if n:
        print("已写入论文用对照图:", OUTPUT_DIR)
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())

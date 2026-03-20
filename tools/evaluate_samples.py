from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.ocr_service import OCRService
from backend.services.parser_service import FileParserService


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".xlsx"}


def load_truth(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_value(value: Any) -> str:
    return str(value if value is not None else "").strip()


def compare_rows(pred_rows: List[Dict[str, Any]], truth_rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    correct = 0
    total = 0
    row_count = min(len(pred_rows), len(truth_rows))
    for idx in range(row_count):
        pred_values = pred_rows[idx].get("values", {})
        truth_values = truth_rows[idx].get("values", {})
        for key, truth_value in truth_values.items():
            total += 1
            if normalize_value(pred_values.get(key)) == normalize_value(truth_value):
                correct += 1
    return correct, total


def save_debug_images(ocr_service: OCRService, file_path: Path, out_dir: Path) -> None:
    if file_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return
    image = ocr_service.decode_image(file_path.read_bytes())
    stages = ocr_service.debug_preprocess(image)
    target_dir = out_dir / file_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in stages.items():
        out_path = target_dir / f"{name}.png"
        cv2.imencode(".png", frame)[1].tofile(str(out_path))


def evaluate(samples_dir: Path, debug_dir: Path | None = None) -> Dict[str, Any]:
    ocr_service = OCRService()
    parser = FileParserService(ocr_service=ocr_service)

    files = sorted(p for p in samples_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
    report: Dict[str, Any] = {
        "sample_count": 0,
        "field_correct": 0,
        "field_total": 0,
        "full_match_count": 0,
        "total_time_ms": 0.0,
        "samples": [],
    }

    for file_path in files:
        truth = load_truth(file_path.with_suffix(".json"))
        raw = file_path.read_bytes()
        start = time.perf_counter()
        parsed = parser.parse(filename=file_path.name, content_type="", data=raw)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        truth_rows = truth.get("rows", [])
        pred_rows = parsed.get("rows", [])
        correct, total = compare_rows(pred_rows, truth_rows)
        full_match = bool(total) and correct == total and len(pred_rows) == len(truth_rows)

        sample_result = {
            "file": file_path.name,
            "elapsed_ms": round(elapsed_ms, 2),
            "field_accuracy": round((correct / total) if total else 0.0, 4),
            "full_match": full_match,
            "predicted_rows": len(pred_rows),
            "truth_rows": len(truth_rows),
        }
        report["samples"].append(sample_result)
        report["sample_count"] += 1
        report["field_correct"] += correct
        report["field_total"] += total
        report["total_time_ms"] += elapsed_ms
        if full_match:
            report["full_match_count"] += 1

        if debug_dir is not None:
            save_debug_images(ocr_service, file_path, debug_dir)

    report["avg_time_ms"] = round(
        report["total_time_ms"] / report["sample_count"], 2
    ) if report["sample_count"] else 0.0
    report["field_accuracy"] = round(
        report["field_correct"] / report["field_total"], 4
    ) if report["field_total"] else 0.0
    report["full_match_rate"] = round(
        report["full_match_count"] / report["sample_count"], 4
    ) if report["sample_count"] else 0.0
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate transcript OCR samples.")
    parser.add_argument("samples_dir", type=Path, help="Directory containing sample files and truth JSON files.")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Optional directory for preprocessing debug images.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save the JSON report.")
    args = parser.parse_args()

    report = evaluate(args.samples_dir, args.debug_dir)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

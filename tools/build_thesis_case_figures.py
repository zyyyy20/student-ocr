# -*- coding: utf-8 -*-
"""
论文插图生成：
- 第 4 章六张对照图：论文生成脚本优先使用 testdemo/output/fig_4_*.png（由 testdemo/run_pipeline_demo.py 写入）；
  本脚本仍写入 out/test1/preprocess_by_section/ 同名文件以便与 out/test1 阶段图并存；逻辑为 write_chapter4_pair_figures（与 preprocess_stage_snapshots 一致）
- out/test1/、out/test2/：同一主案例的阶段导出与整线对比（test2 为 preprocess_with_context 端到端）

主案例（优先）：仓库根目录 Snipaste_2026-04-13_10-26-13.png（班级平时成绩表截图）。
若不存在则依次回退到 qq_pic 合并图、IMG 拍照样例、DEFAULT_CASE、白边截图、旋转样例路径。

运行：在仓库根目录 python tools/build_thesis_case_figures.py

多案例（4.1—4.6 各节专用合成图与对照）：python tools/build_chapter4_eg_figures.py → 写入 eg/case_4_*.png 与 eg/fig_4_*.png。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "out"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 论文与插图统一主案例（与 generate_thesis_2252733.py 正文描述一致）
PRIMARY_CASE = ROOT / "Snipaste_2026-04-13_10-26-13.png"
CASE_FALLBACK_QQ = ROOT / "qq_pic_merged_1776046640446.jpg"
CASE_FALLBACK_IMG = ROOT / "IMG_20260412_225722.jpg"
DEFAULT_CASE = Path(r"C:\Users\zy\Desktop\{F96D111D-6FF3-46e7-AA5B-74A534B582D8}.png")
CASE_ROTATED = Path(r"C:\Users\zy\Desktop\IMG_20260412_212902.png")
CASE_WHITEBORDER = Path(r"C:\Users\zy\Desktop\Snipaste_2026-04-12_21-31-16.png")


def _to_bgr3(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return img
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    c = img.shape[2]
    if c == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if c == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    return cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)


def _hstack_with_gap(left: np.ndarray, right: np.ndarray, gap_px: int = 20) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    a = _resize_to_height(left, h)
    b = _resize_to_height(right, h)
    gap = np.ones((h, gap_px, 3), dtype=np.uint8) * 255
    return np.hstack([a, gap, b])


def _label_bar(img: np.ndarray, text: str) -> np.ndarray:
    bar_h = 34
    w = img.shape[1]
    bar = np.ones((bar_h, w, 3), dtype=np.uint8) * 255
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _font_cn(size: int = 20):
    from PIL import ImageFont

    for path in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def render_pair_cn(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    out_path: Path,
    *,
    main_title: str,
    left_caption: str,
    right_caption: str,
    side_gap: int = 20,
    top_bar: int = 52,
    bottom_bar: int = 46,
    max_pair_h: int = 520,
) -> None:
    """左右两图 + 顶栏中文总标题 + 底栏左右中文小标题。"""
    from PIL import Image, ImageDraw

    left_bgr = _to_bgr3(left_bgr)
    right_bgr = _to_bgr3(right_bgr)
    mh = max(left_bgr.shape[0], right_bgr.shape[0])
    if mh > max_pair_h:
        s = max_pair_h / float(mh)
        left_bgr = cv2.resize(
            left_bgr,
            (int(left_bgr.shape[1] * s), int(left_bgr.shape[0] * s)),
            interpolation=cv2.INTER_AREA,
        )
        right_bgr = cv2.resize(
            right_bgr,
            (int(right_bgr.shape[1] * s), int(right_bgr.shape[0] * s)),
            interpolation=cv2.INTER_AREA,
        )
    core = _hstack_with_gap(left_bgr, right_bgr, gap_px=side_gap)
    pil_core = Image.fromarray(_bgr_to_rgb(core))
    w, h = pil_core.size
    canvas = Image.new("RGB", (w, h + top_bar + bottom_bar), (255, 255, 255))
    canvas.paste(pil_core, (0, top_bar))
    draw = ImageDraw.Draw(canvas)
    font = _font_cn(20)
    font_sm = _font_cn(16)
    bx = draw.textbbox((0, 0), main_title, font=font)
    tw, th = bx[2] - bx[0], bx[3] - bx[1]
    draw.text(((w - tw) // 2, max(4, (top_bar - th) // 2)), main_title, fill=(0, 0, 0), font=font)
    draw.text((10, top_bar + h + 8), left_caption, fill=(40, 40, 40), font=font_sm)
    rbx = draw.textbbox((0, 0), right_caption, font=font_sm)
    rw = rbx[2] - rbx[0]
    draw.text((w - rw - 10, top_bar + h + 8), right_caption, fill=(40, 40, 40), font=font_sm)
    out_bgr = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out_bgr)


def _write_default_legacy(ocr, image_bgr: np.ndarray, test1: Path, test2: Path) -> None:
    test1.mkdir(parents=True, exist_ok=True)
    test2.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(test1 / "00_original.png"), image_bgr)
    stages = ocr.preprocess_stage_snapshots(image_bgr)
    for key in (
        "01_normalized",
        "02_paper_crop",
        "03_perspective",
        "04_deskew",
        "05_table_roi",
        "06_enhanced",
    ):
        arr = stages.get(key)
        if arr is not None:
            cv2.imwrite(str(test1 / f"{key}.png"), arr)
    enhanced = stages.get("06_enhanced")
    if enhanced is None:
        enhanced = stages.get("05_table_roi", image_bgr)
    enhanced = _to_bgr3(enhanced)
    cmp1 = _label_bar(
        _hstack_with_gap(image_bgr, enhanced),
        "legacy: original | tail enhanced",
    )
    cv2.imwrite(str(test1 / "compare_before_after_enhanced.png"), cmp1)
    thumbs = []
    labels = [
        "00_original",
        "01_normalized",
        "02_paper_crop",
        "03_perspective",
        "04_deskew",
        "05_table_roi",
        "06_enhanced",
    ]
    for lab in labels:
        p = test1 / f"{lab}.png"
        if not p.exists() and lab == "00_original":
            arr = image_bgr
        elif p.exists():
            arr = _to_bgr3(cv2.imread(str(p), cv2.IMREAD_UNCHANGED))
        else:
            continue
        t = cv2.resize(arr, (320, int(320 * arr.shape[0] / max(arr.shape[1], 1))))
        thumbs.append(_label_bar(t, lab))
    if thumbs:
        max_h = max(t.shape[0] for t in thumbs)
        padded = []
        for t in thumbs:
            if t.shape[0] < max_h:
                pad = np.ones((max_h - t.shape[0], t.shape[1], 3), dtype=np.uint8) * 255
                t = np.vstack([t, pad])
            padded.append(t)
        gap = np.ones((max_h, 16, 3), dtype=np.uint8) * 255
        parts: list[np.ndarray] = []
        for i, t in enumerate(padded):
            if i:
                parts.append(gap)
            parts.append(t)
        mont = np.hstack(parts)
        mh, mw = mont.shape[:2]
        if mw > 5200:
            sc = 5200 / float(mw)
            mont = cv2.resize(mont, (int(mw * sc), int(mh * sc)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(test1 / "montage_all_stages.png"), mont)
    pr = ocr.preprocess_with_context(image_bgr)
    proc = _to_bgr3(pr["image"])
    cv2.imwrite(str(test2 / "00_original.png"), image_bgr)
    cv2.imwrite(str(test2 / "01_full_preprocess_pipeline.png"), proc)
    cv2.imwrite(
        str(test2 / "02_original_vs_pipeline.png"),
        _label_bar(_hstack_with_gap(image_bgr, proc), "original | preprocess_with_context"),
    )


def _load_bgr(ocr, path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return _to_bgr3(ocr.decode_image(path.read_bytes()))
    except Exception as e:
        print("解码失败:", path, e)
        return None


def _stages(ocr, img: np.ndarray) -> dict:
    return ocr.preprocess_stage_snapshots(img)


def _pick_primary_image(ocr) -> tuple[np.ndarray, str] | None:
    """返回 (BGR 图, 用于日志的源路径说明)。"""
    for path in (PRIMARY_CASE, CASE_FALLBACK_QQ, CASE_FALLBACK_IMG, DEFAULT_CASE, CASE_WHITEBORDER, CASE_ROTATED):
        if not path.exists():
            continue
        img = _load_bgr(ocr, path)
        if img is not None:
            return img, path.name
    return None


def write_chapter4_pair_figures(ocr, out_dir: Path) -> int:
    """
    按线上 preprocess_with_context 顺序生成图 4-1—4-6 左右对照 PNG（与 preprocess_stage_snapshots 一致）。
    out_dir 例如：out/test1/preprocess_by_section 或 testdemo/output。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    picked = _pick_primary_image(ocr)
    if picked is None:
        print("无可用样例图像，跳过第4章对照图:", out_dir)
        return 0
    im, src_name = picked
    st = _stages(ocr, im)
    print("第4章分节插图主案例:", src_name, "→", out_dir)

    def g(key: str, fb: np.ndarray) -> np.ndarray:
        x = st.get(key)
        return _to_bgr3(x) if x is not None else _to_bgr3(fb)

    s01 = g("01_normalized", im)
    render_pair_cn(
        im,
        s01,
        out_dir / "fig_4_1.png",
        main_title="4.1 解码与预处理入口（同一案例，与线上一致）",
        left_caption="处理前：解码后（本案例：" + src_name + "）",
        right_caption="处理后：归一化（灰度转三通道 / 透明合成白底）",
    )

    render_pair_cn(
        im,
        s01,
        out_dir / "fig_4_2.png",
        main_title="4.2 输入归一化与透明通道（同一案例）",
        left_caption="处理前：解码结果（JPG 多为三通道；截图类可有白边或透明）",
        right_caption="处理后：_normalize_input_image 之后（与 preprocess_with_context 首步一致）",
    )

    n1 = g("01_normalized", im)
    n2 = g("02_paper_crop", n1)
    render_pair_cn(
        n1,
        n2,
        out_dir / "fig_4_3.png",
        main_title="4.3 白纸区域预裁切（同一案例，与线上一致）",
        left_caption="处理前：归一化后整页",
        right_caption="处理后：纸张掩膜 bbox 预裁切（无检出或未触发时与左侧近似）",
    )

    p3 = g("03_perspective", n2)
    render_pair_cn(
        n2,
        p3,
        out_dir / "fig_4_4.png",
        main_title="4.4 透视校正（同一案例，与线上一致）",
        left_caption="处理前：预裁切后（透明图跳过透视时与右侧近似）",
        right_caption="处理后：透视拉直（无四边形或未执行时与左侧近似）",
    )

    d4 = g("04_deskew", p3)
    render_pair_cn(
        p3,
        d4,
        out_dir / "fig_4_5.png",
        main_title="4.5 旋转纠偏（同一案例，与线上一致）",
        left_caption="处理前：透视后",
        right_caption="处理后：_deskew_rotate_with_matrix 之后",
    )

    r5 = g("05_table_roi", d4)
    en = g("06_enhanced", r5)
    render_pair_cn(
        r5,
        en,
        out_dir / "fig_4_6.png",
        main_title="4.6 表格 ROI 与送 OCR 前增强（同一案例，与线上一致）",
        left_caption="处理前：表格/标题 ROI（_crop_to_table_or_title_with_matrix 之后）",
        right_caption="处理后：小图白边 padding + 放大（_enhance_for_ocr_with_matrix）",
    )

    print("已写入分节对照图:", out_dir)
    return 1


def build_section_figures(ocr) -> int:
    return write_chapter4_pair_figures(ocr, OUT_ROOT / "test1" / "preprocess_by_section")


def main() -> int:
    from backend.services.ocr_service import OCRService

    ocr = OCRService()
    n = build_section_figures(ocr)
    picked = _pick_primary_image(ocr)
    if picked is not None:
        img, name = picked
        _write_default_legacy(ocr, img, OUT_ROOT / "test1", OUT_ROOT / "test2")
        print("已写入 out/test1/ 与 out/test2/，主案例:", name)
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())

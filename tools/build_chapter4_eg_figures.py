# -*- coding: utf-8 -*-
"""
基于用户提供的 Snipaste 成绩表截图，在 eg/ 下合成 4.1—4.5 各节专用案例图；case_4_6.png 由 case_4_5.png
经完整预处理链得到「纠偏后」（04_deskew）再缩小生成，使 4.6 与 4.5 案例衔接。
再经与线上一致的 preprocess_stage_snapshots 生成左右对照图 fig_4_1.png … fig_4_6.png。

运行（仓库根目录）：
  python tools/build_chapter4_eg_figures.py

命名约定：
  eg/case_4_1—4_5.png — 程序合成；case_4_6.png — 4.5 案例流水线纠偏后再缩小
  eg/fig_4_k.png  — 第 4.k 节「处理前 | 处理后」对照（与 build_thesis_case_figures 同一套 Pillow 标注逻辑）
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EG_DIR = ROOT / "eg"
BASE_IMAGE = ROOT / "Snipaste_2026-04-13_10-26-13.png"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.ocr_service import OCRService  # noqa: E402
from tools.build_thesis_case_figures import render_pair_cn, _to_bgr3  # noqa: E402


def _read_base_bgr() -> np.ndarray:
    if not BASE_IMAGE.exists():
        raise FileNotFoundError(f"缺少基准图：{BASE_IMAGE}")
    bgr = cv2.imread(str(BASE_IMAGE), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise ValueError(f"无法读取：{BASE_IMAGE}")
    return bgr


def synthesize_case_4_1_jpeg_decode(bgr: np.ndarray) -> np.ndarray:
    """4.1：模拟 JPG 有损解码后再读入（解码入口 + 三通道常见形态）。"""
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return bgr.copy()
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def synthesize_case_4_2_bgra(bgr: np.ndarray) -> np.ndarray:
    """4.2：BGRA，边缘半透明，模拟截图透明通道与白底合成需求。"""
    h, w = bgr.shape[:2]
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    alpha = np.ones((h, w), dtype=np.float32) * 255.0
    border = max(18, min(h, w) // 14)
    for t in range(border):
        fade = t / max(border, 1)
        alpha[t, :] = np.minimum(alpha[t, :], fade * 255)
        alpha[h - 1 - t, :] = np.minimum(alpha[h - 1 - t, :], fade * 255)
        alpha[:, t] = np.minimum(alpha[:, t], fade * 255)
        alpha[:, w - 1 - t] = np.minimum(alpha[:, w - 1 - t], fade * 255)
    bgra[:, :, 3] = alpha.astype(np.uint8)
    return bgra


def synthesize_case_4_3_desk_background(bgr: np.ndarray) -> np.ndarray:
    """4.3：成绩表贴在较大桌面区域上，便于白纸掩膜预裁切。"""
    h, w = bgr.shape[:2]
    desk_h = int(h * 1.42)
    desk_w = int(w * 1.38)
    rng = np.random.default_rng(42)
    desk = np.zeros((desk_h, desk_w, 3), dtype=np.uint8)
    for c in range(3):
        band = np.linspace(90 + c * 8, 150 + c * 12, desk_w, dtype=np.float32)
        desk[:, :, c] = (band + rng.normal(0, 4, (desk_h, desk_w))).astype(np.int32).clip(40, 200).astype(np.uint8)
    desk = cv2.GaussianBlur(desk, (31, 31), 0)
    y0 = (desk_h - h) // 2
    x0 = (desk_w - w) // 2
    roi = desk[y0 : y0 + h, x0 : x0 + w].copy()
    blended = cv2.addWeighted(roi, 0.22, bgr, 0.78, 0)
    desk[y0 : y0 + h, x0 : x0 + w] = blended
    return desk


def synthesize_case_4_4_perspective(bgr: np.ndarray) -> np.ndarray:
    """4.4：明显透视梯形，便于透视校正前后对比。"""
    h, w = bgr.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    skew_x = w * 0.11
    skew_y = h * 0.07
    dst = np.float32(
        [
            [skew_x, skew_y * 0.3],
            [w - 1 - skew_x * 0.4, skew_y],
            [w - 1 - skew_x * 0.2, h - 1 - skew_y * 0.2],
            [skew_x * 0.5, h - 1 - skew_y * 0.35],
        ]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(bgr, M, (w, h), borderValue=(245, 245, 245))


def synthesize_case_4_5_deskew(bgr: np.ndarray) -> np.ndarray:
    """4.5：整页小角度旋转，便于纠偏步骤展示。"""
    h, w = bgr.shape[:2]
    angle = -7.2
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += (nw - w) / 2.0
    M[1, 2] += (nh - h) / 2.0
    return cv2.warpAffine(bgr, M, (nw, nh), borderValue=(255, 255, 255))


def synthesize_case_4_6_small_enhance(bgr: np.ndarray) -> np.ndarray:
    """4.6：极小分辨率，触发送 OCR 前白边与放大。"""
    h, w = bgr.shape[:2]
    max_side = 340
    scale = min(max_side / max(h, w), 1.0)
    if scale >= 1.0:
        scale = 0.28
    tw = max(32, int(w * scale))
    th = max(32, int(h * scale))
    return cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)


def write_all_synthetic_cases() -> None:
    """写入 case_4_1—4_5；case_4_6 依赖 OCR，见 write_case_4_6_from_post_4_5。"""
    EG_DIR.mkdir(parents=True, exist_ok=True)
    bgr = _read_base_bgr()
    writers = [
        ("case_4_1.png", synthesize_case_4_1_jpeg_decode(bgr)),
        ("case_4_2.png", synthesize_case_4_2_bgra(bgr)),
        ("case_4_3.png", synthesize_case_4_3_desk_background(bgr)),
        ("case_4_4.png", synthesize_case_4_4_perspective(bgr)),
        ("case_4_5.png", synthesize_case_4_5_deskew(bgr)),
    ]
    for name, img in writers:
        out = EG_DIR / name
        if img.ndim == 3 and img.shape[2] == 4:
            cv2.imwrite(str(out), img)
        else:
            cv2.imwrite(str(out), _to_bgr3(img))
        print("已写入案例图:", out)


def write_case_4_6_from_post_4_5(ocr: OCRService) -> None:
    """
    4.6 输入 = 对 case_4_5 跑 preprocess_stage_snapshots 得到的 04_deskew（与 4.5 节「处理后」一致），
    再缩小以触发送 OCR 前增强；保证与 4.5 案例同一成像链衔接。
    """
    p5 = EG_DIR / "case_4_5.png"
    if not p5.exists():
        print("跳过 case_4_6：缺少", p5)
        return
    im5 = ocr.decode_image(p5.read_bytes())
    st = ocr.preprocess_stage_snapshots(im5)
    d4 = st.get("04_deskew")
    if d4 is None or d4.size == 0:
        d4 = im5
    d4_bgr = _to_bgr3(d4)
    small = synthesize_case_4_6_small_enhance(d4_bgr)
    out = EG_DIR / "case_4_6.png"
    cv2.imwrite(str(out), small)
    print("已写入案例图（4.5 纠偏后缩小）:", out)


def _stages(ocr: OCRService, im: np.ndarray) -> dict:
    return ocr.preprocess_stage_snapshots(im)


def write_pair_figures_for_eg_cases(ocr: OCRService) -> None:
    """对 eg/case_4_k.png 分别跑六步快照，写出 eg/fig_4_k.png。"""
    meta = [
        (
            1,
            "4.1 解码与预处理入口（案例：case_4_1.png，模拟 JPG解码链）",
            "处理前：解码后（case_4_1.png）",
            "处理后：归一化（灰度转三通道 / 透明合成白底）",
        ),
        (
            2,
            "4.2 输入归一化与透明通道（案例：case_4_2.png，BGRA 边缘半透明）",
            "处理前：解码结果（含 alpha，跳过部分几何分支）",
            "处理后：_normalize_input_image 之后",
        ),
        (
            3,
            "4.3 白纸区域预裁切（案例：case_4_3.png，桌面背景）",
            "处理前：归一化后整页",
            "处理后：纸张掩膜 bbox 预裁切",
        ),
        (
            4,
            "4.4 透视校正（案例：case_4_4.png，强透视）",
            "处理前：预裁切后",
            "处理后：透视拉直",
        ),
        (
            5,
            "4.5 旋转纠偏（案例：case_4_5.png，整页倾斜）",
            "处理前：透视后",
            "处理后：_deskew_rotate_with_matrix 之后",
        ),
        (
            6,
            "4.6 表格 ROI 与送 OCR 前增强（案例：case_4_6.png，由 case_4_5 纠偏后缩小）",
            "处理前：表格/标题 ROI",
            "处理后：白边 padding + 放大",
        ),
    ]

    for k, main_title, cap_l_base, cap_r_base in meta:
        path = EG_DIR / f"case_4_{k}.png"
        if not path.exists():
            print("跳过（无文件）:", path)
            continue
        im = ocr.decode_image(path.read_bytes())
        st = _stages(ocr, im)
        src_name = path.name

        def g(key: str, fb: np.ndarray) -> np.ndarray:
            x = st.get(key)
            return _to_bgr3(x) if x is not None else _to_bgr3(fb)

        s01 = g("01_normalized", im)
        n1 = s01
        n2 = g("02_paper_crop", n1)
        p3 = g("03_perspective", n2)
        d4 = g("04_deskew", p3)
        r5 = g("05_table_roi", d4)
        en = g("06_enhanced", r5)

        if k == 1:
            render_pair_cn(im, s01, EG_DIR / "fig_4_1.png", main_title=main_title, left_caption=cap_l_base, right_caption=cap_r_base)
        elif k == 2:
            render_pair_cn(im, s01, EG_DIR / "fig_4_2.png", main_title=main_title, left_caption=cap_l_base, right_caption=cap_r_base)
        elif k == 3:
            render_pair_cn(n1, n2, EG_DIR / "fig_4_3.png", main_title=main_title, left_caption=cap_l_base, right_caption=cap_r_base)
        elif k == 4:
            render_pair_cn(n2, p3, EG_DIR / "fig_4_4.png", main_title=main_title, left_caption=cap_l_base, right_caption=cap_r_base)
        elif k == 5:
            render_pair_cn(p3, d4, EG_DIR / "fig_4_5.png", main_title=main_title, left_caption=cap_l_base, right_caption=cap_r_base)
        else:
            render_pair_cn(r5, en, EG_DIR / "fig_4_6.png", main_title=main_title, left_caption=cap_l_base, right_caption=cap_r_base)

        print("已写入对照图:", EG_DIR / f"fig_4_{k}.png", "←", src_name)


def main() -> int:
    try:
        write_all_synthetic_cases()
    except Exception as e:
        print("合成案例图失败:", e)
        return 1
    ocr = OCRService()
    write_case_4_6_from_post_4_5(ocr)
    write_pair_figures_for_eg_cases(ocr)
    print("完成。案例与对照图目录:", EG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

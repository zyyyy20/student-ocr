# -*- coding: utf-8 -*-
"""
生成毕业论文 Word（2252733-林子榆），默认输出到仓库 out/ 目录。
运行：在仓库根目录执行  python tools/generate_thesis_2252733.py

生成前会自动依次运行 testdemo/run_pipeline_demo.py、tools/build_thesis_case_figures.py、
tools/build_chapter4_eg_figures.py；第 4 章插图优先使用 eg/fig_4_*.png（分节多案例），缺失时回退 testdemo/output/。

正文引用：在传给 add_body_paragraph 的字符串中用「句内占位{{文献序号}}」，例如「……有效特征{{10}}。」（当前参考文献表为 [1]—[11]）
生成 Word 时：正文「{{n}}」插入为 **REF 域**（指向参考文献中书签 litref_n），显示为右上角上标 [n]，与参考文献编号联动；用 Word 打开后可选中全文按 **F9** 更新域。
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

# 路径（论文与插图副本写入仓库 out/）
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "out"
# 使用 Unicode 转义，避免 Windows 控制台默认编码导致文件名乱码
OUT_DOCX = OUTPUT_ROOT / "2252733-\u6797\u5b50\u6986-\u8bba\u6587.docx"
FIG_SRC = Path(r"C:\Users\zy\Desktop\student-ocr\Snipaste_2026-02-24_19-13-43.png")
PROJECT = Path(__file__).resolve().parent.parent
THESIS_CASE_SCRIPT = PROJECT / "tools" / "build_thesis_case_figures.py"
THESIS_DEMO_SCRIPT = PROJECT / "testdemo" / "run_pipeline_demo.py"
THESIS_CH4_EG_SCRIPT = PROJECT / "tools" / "build_chapter4_eg_figures.py"
FIG4_EG_DIR = PROJECT / "eg"
FIG4_FALLBACK_DIR = PROJECT / "testdemo" / "output"
FIG6_4_A = OUTPUT_ROOT / "test2" / "02_original_vs_pipeline.png"
FIG6_4_B = OUTPUT_ROOT / "test2" / "01_full_preprocess_pipeline.png"


def resolve_fig4(index: int) -> Path:
    """第 4 章插图：优先 eg/fig_4_{index}.png（分节多案例），否则 testdemo/output/。"""
    name = f"fig_4_{index}.png"
    eg = FIG4_EG_DIR / name
    if eg.exists():
        return eg
    return FIG4_FALLBACK_DIR / name

# 正文内引用占位：在字符串中写 {{1}}、{{2}} 等；生成 Word 时为 REF 交叉引用（见 add_literature_ref_field）
_CITE_MARKER_RE = re.compile(r"\{\{(\d+)\}\}")

# 参考文献书签 ID（与 w:bookmarkStart / w:bookmarkEnd 的 w:id 对应，全文唯一）
_DOC_BOOKMARK_ID = 0


def _reset_doc_bookmark_ids() -> None:
    global _DOC_BOOKMARK_ID
    _DOC_BOOKMARK_ID = 0


def _next_bookmark_id() -> int:
    global _DOC_BOOKMARK_ID
    _DOC_BOOKMARK_ID += 1
    return _DOC_BOOKMARK_ID


def add_literature_ref_field(paragraph, ref_num: int) -> None:
    """插入 REF 域，引用参考文献中的书签 litref_{ref_num}，显示为上标 [ref_num]。"""
    bookmark = f"litref_{ref_num}"
    display = f"[{ref_num}]"
    r1 = paragraph.add_run()
    bc = OxmlElement("w:fldChar")
    bc.set(qn("w:fldCharType"), "begin")
    r1._r.append(bc)

    r2 = paragraph.add_run()
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" REF {bookmark} \\h \\* MERGEFORMAT "
    r2._r.append(instr)

    r3 = paragraph.add_run()
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    r3._r.append(sep)

    r4 = paragraph.add_run(display)
    set_run_font(r4, east_asia="宋体", ascii_font="Times New Roman", size=Pt(12))
    r4.font.superscript = True

    r5 = paragraph.add_run()
    ec = OxmlElement("w:fldChar")
    ec.set(qn("w:fldCharType"), "end")
    r5._r.append(ec)


def add_reference_paragraph(doc: Document, ref_line: str) -> None:
    """参考文献一行：为前置 [n] 加书签 litref_n，供正文 REF 域交叉引用。"""
    rp = doc.add_paragraph()
    rp.paragraph_format.first_line_indent = Pt(0)
    rp.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    m = re.match(r"^\[(\d+)\]\s*(.*)$", ref_line, re.DOTALL)
    if not m:
        rr = rp.add_run(ref_line)
        set_run_font(rr, east_asia="宋体", ascii_font="Times New Roman", size=Pt(12))
        return
    num = int(m.group(1))
    rest = m.group(2)
    bid = _next_bookmark_id()
    bm_s = OxmlElement("w:bookmarkStart")
    bm_s.set(qn("w:id"), str(bid))
    bm_s.set(qn("w:name"), f"litref_{num}")
    rp._p.append(bm_s)
    r_tag = rp.add_run(f"[{num}]")
    set_run_font(r_tag, east_asia="宋体", ascii_font="Times New Roman", size=Pt(12))
    bm_e = OxmlElement("w:bookmarkEnd")
    bm_e.set(qn("w:id"), str(bid))
    r_tag._r.addnext(bm_e)
    prefix = " " if rest else ""
    r_rest = rp.add_run(f"{prefix}{rest}")
    set_run_font(r_rest, east_asia="宋体", ascii_font="Times New Roman", size=Pt(12))

ABSTRACT_CN_MARKED = (
    "高校教学管理中存在大量以图片、拍照或扫描件形式流转的学生成绩单与班级成绩表，人工录入效率低、易出错。"
    "本文设计并实现了一套基于PaddleOCR{{1}}与PaddlePaddle{{7}}生态、并结合OpenCV{{6}}的学生成绩单自动识别系统，完成从图像输入、预处理、"
    "文字检测到表格结构恢复、字段结构化、浏览器端校对编辑与 Excel 导出的完整链路。"
    "工程上将检测识别结果统一为包含文本、置信度与四点框的 OCRItem 列表，作为表格恢复与业务解析的直接输入。"
    "系统采用FastAPI{{4}}提供 REST 接口，浏览器端以原生 HTML/CSS/JavaScript 实现交互。"
    "在表格恢复方面，综合采用基于 OCR 检测框的几何聚类、表格线网格提取以及PaddleX{{2}}表格识别流水线兜底的多路径策略；"
    "并引入二次聚焦 OCR 与单元格局部补识别，以提升短数字与小数点的召回率。"
    "实验与运行结果表明，相较传统“整图单次 OCR + 按阅读顺序拼接”或纯人工录入方式，本系统在结构化准确率、"
    "可解释性（单元格置信度与低分标红提示）以及端到端处理效率方面具有明显优势。"
)


def set_run_font(run, *, east_asia: str, ascii_font: str = "Times New Roman", size: Pt | None = None, bold: bool | None = None) -> None:
    """设置中英文字体；通过 get_or_add_rFonts 避免 rFonts 为空。"""
    r_pr = run._element.get_or_add_rPr()
    r_fonts = r_pr.get_or_add_rFonts()
    r_fonts.set(qn("w:ascii"), ascii_font)
    r_fonts.set(qn("w:hAnsi"), ascii_font)
    r_fonts.set(qn("w:eastAsia"), east_asia)
    r_fonts.set(qn("w:cs"), east_asia)
    if size is not None:
        run.font.size = size
    if bold is not None:
        run.font.bold = bold


def set_cell_font(cell, cn: str, en: str, size: Pt, bold: bool = False) -> None:
    for p in cell.paragraphs:
        for r in p.runs:
            set_run_font(r, east_asia=cn, ascii_font=en, size=size, bold=bold)


def add_body_paragraph(doc: Document, text: str, first_line_indent: bool = True) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    p.paragraph_format.space_after = Pt(6)
    if first_line_indent:
        p.paragraph_format.first_line_indent = Pt(24)
    if "{{" in text:
        pos = 0
        for m in _CITE_MARKER_RE.finditer(text):
            if m.start() > pos:
                run = p.add_run(text[pos : m.start()])
                set_run_font(run, east_asia="宋体", ascii_font="Times New Roman", size=Pt(12))
            add_literature_ref_field(p, int(m.group(1)))
            pos = m.end()
        if pos < len(text):
            run = p.add_run(text[pos:])
            set_run_font(run, east_asia="宋体", ascii_font="Times New Roman", size=Pt(12))
        return
    run = p.add_run(text)
    set_run_font(run, east_asia="宋体", ascii_font="Times New Roman", size=Pt(12))


def add_heading(doc: Document, text: str, level: int) -> None:
    # level 1: 章, 2: 节, 3: 小节
    sizes = {1: Pt(16), 2: Pt(14), 3: Pt(12)}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    set_run_font(run, east_asia="黑体", ascii_font="Times New Roman", size=sizes.get(level, Pt(12)), bold=True)


def add_code_block(doc: Document, code: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Pt(12)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(code)
    set_run_font(run, east_asia="宋体", ascii_font="Consolas", size=Pt(9.5))


def add_figure_placeholder(doc: Document, caption: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(caption)
    set_run_font(run, east_asia="宋体", ascii_font="Times New Roman", size=Pt(10.5))
    run.italic = True


def run_case_figure_script() -> None:
    """运行插图流水线：testdemo → build_thesis_case_figures → build_chapter4_eg_figures（eg/ 多案例第4章图）；失败不阻断论文生成。"""
    scripts: list[Path] = []
    if THESIS_DEMO_SCRIPT.exists():
        scripts.append(THESIS_DEMO_SCRIPT)
    if THESIS_CASE_SCRIPT.exists():
        scripts.append(THESIS_CASE_SCRIPT)
    if THESIS_CH4_EG_SCRIPT.exists():
        scripts.append(THESIS_CH4_EG_SCRIPT)
    if not scripts:
        return
    for script in scripts:
        try:
            cp = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(PROJECT),
                capture_output=True,
                text=True,
                timeout=600,
            )
            if cp.returncode != 0:
                tail = (cp.stderr or cp.stdout or "").strip() or f"exit {cp.returncode}"
                print(f"插图脚本未完成 {script.name}（论文仍生成）：", tail[:800])
        except Exception as e:
            print(f"插图脚本调用异常 {script.name}（论文仍生成）：", e)


def add_thesis_figure(doc: Document, image_path: Path, caption: str, *, width=Inches(5.2)) -> None:
    if not image_path.exists():
        add_figure_placeholder(
            doc,
            f"【{caption}】未找到 {image_path.name}。请执行 python tools/build_chapter4_eg_figures.py（生成 eg/fig_4_*.png），"
            "或 python testdemo/run_pipeline_demo.py / python tools/build_thesis_case_figures.py（testdemo/output/）。",
        )
        return
    doc.add_paragraph()
    pic_p = doc.add_paragraph()
    pic_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic_p.add_run().add_picture(str(image_path), width=width)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cr = cap.add_run(caption)
    set_run_font(cr, east_asia="宋体", ascii_font="Times New Roman", size=Pt(10.5))


def build_docx() -> None:
    doc = Document()
    _reset_doc_bookmark_ids()
    run_case_figure_script()
    sec = doc.sections[0]
    sec.top_margin = Pt(72)
    sec.bottom_margin = Pt(72)
    sec.left_margin = Pt(90)
    sec.right_margin = Pt(72)

    # —— 封面 ——
    for _ in range(2):
        doc.add_paragraph()
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("上海海洋大学\n本科毕业论文（设计）")
    set_run_font(r, east_asia="黑体", ascii_font="Times New Roman", size=Pt(22), bold=True)

    doc.add_paragraph()
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr = sub.add_run("题目：基于深度学习的学生成绩单\n自动识别系统设计与实现")
    set_run_font(rr, east_asia="黑体", ascii_font="Times New Roman", size=Pt(16), bold=True)

    for _ in range(4):
        doc.add_paragraph()

    info = doc.add_table(rows=6, cols=2)
    info.autofit = True
    rows_data = [
        ("学　　院", "信息学院"),
        ("专　　业", "计算机科学与技术"),
        ("学生姓名", "林子榆"),
        ("学　　号", "2252733"),
        ("指导教师", "（请填写指导教师姓名）"),
        ("完成日期", f"{date.today().year}年 5 月"),
    ]
    for i, (k, v) in enumerate(rows_data):
        info.rows[i].cells[0].text = k
        info.rows[i].cells[1].text = v
        for j in range(2):
            for p in info.rows[i].cells[j].paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    set_run_font(run, east_asia="宋体", ascii_font="Times New Roman", size=Pt(14))

    doc.add_page_break()

    # 声明页（简化）
    doc.add_paragraph()
    q = doc.add_paragraph()
    q.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rq = q.add_run("上海海洋大学\n本科毕业论文（设计）学术诚信声明")
    set_run_font(rq, east_asia="黑体", ascii_font="Times New Roman", size=Pt(16), bold=True)
    add_body_paragraph(
        doc,
        "本人郑重声明：所呈交的毕业论文（设计），是在导师的指导下独立进行研究所取得的成果。"
        "除文中已经注明引用的内容外，本论文不包含任何其他个人或集体已经发表或撰写过的作品成果。"
        "对本文的研究做出重要贡献的个人和集体，均已在文中以明确方式标明。本人完全意识到本声明的法律结果由本人承担。",
        first_line_indent=True,
    )
    doc.add_paragraph()
    add_body_paragraph(doc, "学位论文作者签名：　　　　　　　　日期：　　　　年　　月　　日", first_line_indent=False)
    doc.add_page_break()

    # 中文摘要
    h = doc.add_paragraph()
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    hr = h.add_run("摘　要")
    set_run_font(hr, east_asia="黑体", ascii_font="Times New Roman", size=Pt(16), bold=True)

    add_body_paragraph(doc, ABSTRACT_CN_MARKED)

    kw = doc.add_paragraph()
    kwr = kw.add_run("关键词：成绩单识别；PaddleOCR；表格结构恢复；FastAPI；OpenCV")
    set_run_font(kwr, east_asia="宋体", ascii_font="Times New Roman", size=Pt(12), bold=True)

    doc.add_page_break()

    # 英文摘要
    he = doc.add_paragraph()
    he.alignment = WD_ALIGN_PARAGRAPH.CENTER
    her = he.add_run("ABSTRACT")
    her.bold = True
    her.font.size = Pt(16)

    abstract_en = (
        "Student transcripts and class grade sheets often circulate as images or scans in university administration, "
        "where manual digitization is slow and error-prone. This thesis presents an automatic recognition system "
        "based on PaddleOCR and OpenCV, covering image preprocessing, text detection and recognition, table structure "
        "recovery, structured field extraction, browser-based proofreading, and Excel export. "
        "The backend is built with FastAPI; the frontend uses plain HTML/CSS/JavaScript. "
        "For table recovery, a multi-path strategy combines geometry clustering on OCR boxes, grid-line based "
        "reconstruction, and a PaddleX table-recognition fallback. Secondary focused OCR and local cell re-recognition "
        "improve recall for short numeric cells. Compared with traditional single-pass OCR or purely manual entry, "
        "the proposed system offers better structural fidelity, explainability via per-cell confidences and low-score "
        "highlighting in the editable grid, and higher end-to-end efficiency."
    )
    pe = doc.add_paragraph()
    pe.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    re = pe.add_run(abstract_en)
    re.font.name = "Times New Roman"
    re.font.size = Pt(12)

    kwe = doc.add_paragraph()
    kwer = kwe.add_run(
        "Keywords: transcript recognition; PaddleOCR; table structure recovery; FastAPI; OpenCV"
    )
    kwer.font.name = "Times New Roman"
    kwer.font.size = Pt(12)
    kwer.bold = True

    doc.add_page_break()

    # 目录
    toc = doc.add_paragraph()
    toc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tocr = toc.add_run("目　录")
    set_run_font(tocr, east_asia="黑体", ascii_font="Times New Roman", size=Pt(16), bold=True)

    toc_lines = [
        "第1章　绪论………………………………………………………………1",
        "　1.1　研究背景与意义………………………………………………1",
        "　1.2　国内外研究现状………………………………………………2",
        "　1.3　研究内容与方法………………………………………………3",
        "　1.4　论文组织结构……………………………………………………4",
        "第2章　相关技术与理论基础…………………………………………5",
        "　2.1　文档图像与 OCR 概述…………………………………………5",
        "　2.2　深度学习文字检测与识别……………………………………7",
        "　2.3　PaddleOCR 与 PaddleX 表格理解……………………………9",
        "　2.4　OpenCV 与经典文档图像处理………………………………11",
        "　2.5　Web 服务、序列化与 Excel 导出……………………………13",
        "　2.6　评测指标与实验设计原则……………………………………14",
        "第3章　系统需求分析与总体设计……………………………………16",
        "　3.1　角色、场景与业务流程……………………………………16",
        "　3.2　功能需求详述…………………………………………………18",
        "　3.3　非功能需求与约束……………………………………………20",
        "　3.4　总体架构与模块划分…………………………………………21",
        "　3.5　接口、数据模型与错误处理…………………………………23",
        "　3.6　数据流、部署与安全边界……………………………………24",
        "第4章　图像预处理与 OCR 模块设计………………………………26",
        "　4.1　图像表示、解码与调试输出…………………………………26",
        "　4.2　输入归一化与透明通道处理…………………………………28",
        "　4.3　白纸区域检测与主体裁切……………………………………29",
        "　4.4　透视校正与几何变换矩阵……………………………………31",
        "　4.5　旋转纠偏与文档朝向…………………………………………32",
        "　4.6　表格与标题区域聚焦及小图增强……………………………34",
        "　4.7　OCR 调用封装与版本兼容……………………………………35",
        "　4.8　HTTP 服务层设计………………………………………………37",
        "　4.9　OCR 输出数据结构与中间表示………………………………38",
        "第5章　表格结构恢复与业务解析……………………………………39",
        "　5.1　多路径表格恢复与路径决策……………………………………39",
        "　5.2　解析管线与业务网格抽取………………………………………42",
        "　5.3　交互、导出与接口数据契约……………………………………45",
        "第6章　系统实现、测试与实验分析…………………………………54",
        "　6.1　实现环境与部署运行……………………………………………54",
        "　6.2　端到端功能与流程说明…………………………………………56",
        "　6.3　自动化测试与回归验证…………………………………………57",
        "　6.4　系统界面与实验插图……………………………………………59",
        "　6.5　消融讨论、性能与改进方向……………………………………61",
        "第7章　总结与展望……………………………………………………62",
        "参考文献…………………………………………………………………66",
        "致谢………………………………………………………………………68",
    ]
    for line in toc_lines:
        tp = doc.add_paragraph()
        tr = tp.add_run(line)
        set_run_font(tr, east_asia="宋体", ascii_font="Times New Roman", size=Pt(12))

    doc.add_page_break()

    # ========== 正文 ==========
    add_heading(doc, "第1章　绪论", 1)
    add_heading(doc, "1.1　研究背景与意义", 2)
    add_body_paragraph(
        doc,
        "成绩管理是高校教学运行的基础环节。除教务系统导出的电子表格外，教师与辅导员手中还大量存在 "
        "截图、手机拍照、扫描 PDF 转图等“非结构化”成绩单。将其转为可计算、可统计的电子表格，传统做法依赖人工对照录入，"
        "耗时长且易在姓名、学号、小数成绩等字段上产生笔误。光学字符识别（OCR）可将像素转为文本{{1}}，但成绩单类图像往往伴随透视变形、"
        "背景杂乱、表格线断裂、字号偏小等问题；若仅调用通用 OCR 引擎而不做版面理解与表格重建，输出仍难以直接用于业务系统。",
    )
    add_body_paragraph(
        doc,
        "因此，面向成绩单场景构建“预处理—检测识别—表格恢复—结构化—人机协同校对—导出”的一体化系统，"
        "对提升教学管理信息化水平、减轻基层教务负担具有现实意义；同时在技术层面，也可作为文档智能与多模态感知应用的典型实践。",
    )

    add_heading(doc, "1.2　国内外研究现状", 2)
    add_body_paragraph(
        doc,
        "文字识别经历了由传统模板匹配、手工特征到深度卷积网络与序列建模的演进。近年来，开源方案如 Tesseract、"
        "PaddleOCR、EasyOCR 等在中文场景取得了较高可用性{{1}}。表格理解方面，除基于规则的几何分析外，端到端表格检测与结构识别网络、"
        "以及PaddleX等工具链中的表格识别流水线亦被广泛使用{{2}}{{5}}。工程上，单一模型往往难以覆盖“规则截图、纸质拍照、弱线表”等混合分布，"
        "多路径融合与轻量回退策略成为提高鲁棒性的重要手段。",
    )

    add_heading(doc, "1.3　研究内容与方法", 2)
    add_body_paragraph(
        doc,
        "本文以开源项目“学生成绩单自动识别系统”为工程载体，主要工作包括：（1）基于OpenCV{{6}}的输入归一化、白纸区域提取、"
        "透视校正、旋转纠偏与表格区域裁切；（2）封装PaddleOCR{{1}}完成文本检测与识别，并记录检测框与置信度；（3）设计 OCR 框聚类、"
        "表格线网格与PaddleX{{2}}兜底相结合的多路径表格恢复；（4）在解析层实现表头识别、碎行合并、单元格区域估计与数字补识别；"
        "（5）基于FastAPI{{4}}与静态前端实现上传、双图预览、可编辑校对与 Excel 导出。研究方法以工程实现与定性分析为主；"
        "第6章给出 unittest 回归方案、系统运行说明、界面与预处理插图位，以及消融、性能与可维护性讨论。",
    )

    add_heading(doc, "1.4　论文组织结构", 2)
    add_body_paragraph(
        doc,
        "全文共分七章，整体框架遵循“理论基础—需求与总体设计—核心算法与模块—系统实现与实验—总结”的工科论文常见写法。"
        "第2章从文档图像与 OCR 基本概念出发，综述深度学习检测识别、飞桨生态、OpenCV 文档处理、Web 与序列化技术及评测原则；"
        "第3章分析角色场景、功能与非功能需求，给出分层架构、模块职责、接口与数据模型；"
        "第4章围绕预处理流水线、几何变换、OCR 封装、HTTP 服务以及 OCR 统一输出格式（OCRItem 列表）展开；"
        "第5章在该中间表示之上讨论多路径表格恢复、业务解析纠错与对外 JSON 数据契约；"
        "第6章从实现部署、自动化测试与回归验证出发，以有序插图佐证界面与预处理效果，并讨论消融、性能与改进；"
        "第7章总结全文并展望未来工作。",
    )

    add_heading(doc, "第2章　相关技术与理论基础", 1)
    add_heading(doc, "2.1　文档图像与 OCR 概述", 2)
    add_body_paragraph(
        doc,
        "文档图像与自然场景图像在统计特性上存在显著差异：前者往往具有规则网格、重复出现的笔画结构、较强的行列对齐先验，"
        "但也常见扫描噪声、JPEG 压缩块效应、复印底纹与拍照引入的透视与阴影。光学字符识别（OCR）的目标是将像素阵列映射为机器可读符号序列，"
        "在工程上通常拆为“在哪里读”（检测）与“读成什么”（识别）两个子问题，必要时还引入版面分析、表格结构恢复与语义后处理{{5}}。",
    )
    add_body_paragraph(
        doc,
        "传统 OCR 依赖二值化、连通域分析、投影分割与模板匹配，对版式固定、噪声可控的印刷体效果尚可，但对复杂背景、断笔、粘连与非常规字号的鲁棒性有限{{3}}。"
        "深度学习方法通过卷积或注意力机制自动学习层次特征，使检测与识别在开放场景下获得数量级提升；但神经网络对输入分辨率、对比度与几何对齐仍敏感，"
        "因此在成绩单这类“半结构化文档”上，往往仍需经典图像算法做归一化与几何校正，以放大深度模型的有效感受野并降低误检率{{6}}。",
    )
    add_heading(doc, "2.2　深度学习文字检测与识别", 2)
    add_body_paragraph(
        doc,
        "文本检测可视为一般目标检测的特例：早期工作将文字作为多方向目标，采用锚框回归或分割掩膜定位文本行；"
        "识别阶段则常将裁剪条带编码为序列，用 CTC、Attention 或 Transformer 解码器输出字符序列{{8}}。"
        "端到端模型试图联合优化检测与识别，但在表格单元格尺度差异大、密集小数字多的场景，解耦式“检测框 + 识别头”仍便于调试与替换{{5}}；"
        "其中检测分支广泛采用可微二值化等快速文本检测思想{{9}}。",
    )
    add_body_paragraph(
        doc,
        "置信度是工程落地的重要信号：低置信度不一定意味着错误，但可作为“需要人工复核”的触发条件；前端据此对单元格标红提示，与可编辑表格共同支撑人工校对。"
        "本系统在 OCRItem 中统一保存 text、confidence 与 box，为后续表格聚类、低分标红与局部补识别提供一致的数据契约{{1}}。",
    )
    add_heading(doc, "2.3　PaddleOCR 与 PaddleX 表格理解", 2)
    add_body_paragraph(
        doc,
        "PaddleOCR{{1}}基于PaddlePaddle{{7}}提供面向中文优化的预训练权重与统一推理 API，并可选 use_angle_cls 启用角度分类，以缓解拍照倒置或 90° 旋转带来的识别崩溃。"
        "其返回结构在 2.x 与 3.x 之间存在差异：旧版以 list 嵌套 [box, (text, conf)] 为主，新版可能返回字典字段 rec_texts、rec_scores、rec_polys 等，"
        "工程上需做兼容封装以避免升级依赖后线上服务静默失败。",
    )
    add_body_paragraph(
        doc,
        "PaddleX{{2}}的 table_recognition 流水线将表格理解抽象为更高层的推理任务，可输出 HTML 表格或结构化单元格。"
        "相较轻量几何重建，其计算开销更大、对环境依赖更多，但在弱线表、非常规合并单元格或检测框严重缺失时，可作为兜底路径。"
        "本系统将其置于“得分不足再触发”的位置，体现“先快后稳”的资源权衡思想。",
    )
    add_heading(doc, "2.4　OpenCV 与经典文档图像处理", 2)
    add_body_paragraph(
        doc,
        "OpenCV{{6}}在颜色空间转换（BGR、HSV、LAB）、阈值与自适应阈值、形态学开闭运算、轮廓近似、Canny/Hough、仿射与透视变换等方面提供成熟实现。"
        "文档预处理中常见流程包括：去噪、对比度增强、倾斜校正、透视拉直与感兴趣区域（ROI）裁剪。"
        "本系统在白纸检测上组合 HSV 与 LAB 阈值以提取高亮纸张区域，并结合面积与宽高比约束抑制桌面背景；在表格线分析上采用二值化后分别腐蚀提取横竖线再融合，"
        "属于可解释、可调参的经典管线，便于针对本校成绩单样式做微调。",
    )
    add_heading(doc, "2.5　Web 服务、序列化与 Excel 导出", 2)
    add_body_paragraph(
        doc,
        "FastAPI 与 Uvicorn 构成常用的异步 Web 部署组合{{4}}：前者基于类型注解自动生成 OpenAPI 文档并原生支持 async/await，适合 I/O 密集的上传解析场景；后者作为 ASGI 服务器提供高性能事件循环。"
        "前后端以 JSON 作为交换格式，可嵌套 headers、rows、meta、每格 confidences 及预处理预览等字段，兼顾机器处理与人工校对。",
    )
    add_body_paragraph(
        doc,
        "Excel 导出选用 openpyxl{{10}}：相较 CSV，xlsx 支持列宽、冻结窗格、自动筛选与简单表样式，更贴近教务人员使用习惯。"
        "导出接口与识别接口解耦，使前端可在人工修正表格后再提交，避免“识别即落库”的刚性流程。",
    )
    add_heading(doc, "2.6　评测指标与实验设计原则", 2)
    add_body_paragraph(
        doc,
        "除字符级错误率（CER）外，表格类任务更关心单元格级准确率、表头对齐率、整表可导入率以及端到端耗时。"
        "由于真实成绩单涉及隐私，公开数据集稀缺，论文实验可采用“自建脱敏样本 + 合成扰动（旋转、噪声、压缩）”的方式构造对照。"
        "同时应报告失败样例类型（强反光、折痕、手写覆盖等），避免只给出乐观均值而缺乏可复现细节。",
    )

    add_heading(doc, "第3章　系统需求分析与总体设计", 1)
    add_heading(doc, "3.1　角色、场景与业务流程", 2)
    add_body_paragraph(
        doc,
        "主要用户可抽象为教务管理员、辅导员与任课教师：其共同需求是将微信群、邮箱或纸质材料中的成绩单图片尽快转为可统计的电子表。"
        "典型业务流程为：选择文件上传 → 等待服务端解析 → 在网页上对照原图与预处理预览、核对表头与成绩并可直接编辑 → 确认后导出 Excel → 在教务或统计软件中二次处理。",
    )
    add_body_paragraph(
        doc,
        "输入形态包括：教务系统截图（列对齐好、噪声小）、手机斜拍纸质表（透视明显）、扫描件转 PNG（可能存在轻微倾斜与黑边）、以及已数字化的 XLSX（应直通读取避免重复 OCR）。"
        "系统需在入口根据扩展名与魔数区分路由，并对不支持的格式给出明确错误提示，减少无效等待。",
    )
    add_heading(doc, "3.2　功能需求详述", 2)
    add_body_paragraph(
        doc,
        "（1）图像类输入：支持 PNG/JPG 解码，完成预处理、OCR、表格恢复与结构化输出；（2）矢量与电子表：SVG 优先解析 text 节点，失败则栅格化后走图像链路；"
        "XLSX 直接读取单元格矩阵并包装为与 OCR 结果一致的 JSON，便于前端复用表格组件。（3）元信息：自动抽取标题、姓名、学号、班级、学院、学期等键值对，允许缺失。"
        "（4）交互：原图与预处理结果预览（可点击放大查看）；右侧表格可编辑，低置信度单元格标红。（5）导出：将当前表格写入 xlsx 并返回下载链接。",
    )
    add_body_paragraph(
        doc,
        "功能优先级上，识别准确率与结构稳定性优先于极致速度；但在 CPU 部署约束下，仍需避免无条件调用重型表格模型，故采用多路径与早停策略。",
    )
    add_heading(doc, "3.3　非功能需求与约束", 2)
    add_body_paragraph(
        doc,
        "性能方面：在主流笔记本 CPU 上，单张常见分辨率成绩单的处理时间应控制在可接受范围（具体阈值可由学校机房环境标定），且不应出现无限阻塞。"
        "可靠性方面：单张失败不得拖垮进程；异常应记录为 HTTP 500 并携带可读 detail。可维护性方面：核心服务类具备单元测试；依赖版本写入 requirements.txt。"
        "兼容性方面：支持PaddleOCR{{1}}多版本参数差异；支持 Windows 10/11 与 Chrome/Edge 浏览器。",
    )
    add_heading(doc, "3.4　总体架构与模块划分", 2)
    add_body_paragraph(
        doc,
        "系统采用浏览器/服务器（B/S）结构：表现层为静态 HTML/CSS/JS；应用层为 FastAPI 路由与中间件（含 CORS）；领域层由 OCRService、FileParserService、ExcelService 组成；"
        "基础设施层包括PaddleOCR{{1}}/PaddleX{{2}}推理运行时、OpenCV{{6}}与文件系统。静态资源通过 mount 暴露，导出目录单独挂载为 /downloads，实现读解析与写导出的路径隔离。",
    )
    add_body_paragraph(
        doc,
        "模块边界上，OCRService 专注“图像进、OCRItem 列表与二维表格网格出”；FileParserService 负责类型分派、在网格与 OCR 观测之上做业务字段解析与 JSON 组装；ExcelService 专注工作簿样式与写入。"
        "这种划分有利于单元测试按模块 stub 依赖，也便于将来替换 OCR 引擎而少改上层逻辑。",
    )
    add_heading(doc, "3.5　接口、数据模型与错误处理", 2)
    add_body_paragraph(
        doc,
        "POST /upload 接受 multipart/form-data，字段名为 file；成功时返回 application/json，包含 headers、rows、meta、可选 title、processed_preview 及每格 confidences 等。"
        "POST /export 接受 JSON body，校验 headers 为字符串数组、rows 为对象数组，否则返回 400。GET /health 返回 {\"status\":\"ok\"} 供运维探活。",
    )
    add_body_paragraph(
        doc,
        "错误处理遵循：输入为空 → 400；类型不支持 → 400 并列出允许扩展名；解析内部异常 → 500 且 detail 含异常信息（生产环境可再收敛日志粒度）。"
        "该策略与FastAPI{{4}}的 HTTPException 机制一致，前端可用统一 toast 呈现。",
    )
    add_heading(doc, "3.6　数据流、部署与安全边界", 2)
    add_body_paragraph(
        doc,
        "数据流上，上传字节仅在内存中解码与推理，不落盘原始隐私图像；processed_preview 以 Base64 内嵌响应，应注意体积对网关超时时间的影响，必要时可改为对象存储 URL。"
        "导出文件写入服务器 exports 目录，应配置定期清理或容量阈值，防止磁盘占满。安全上，本课程设计默认内网部署；若公网暴露，需增加鉴权、限流与 HTTPS。",
    )

    add_heading(doc, "第4章　图像预处理与 OCR 模块设计", 1)
    add_body_paragraph(
        doc,
        "本章 4.1—4.6 节均围绕「送入 OCR 模型之前」的图像处理展开；各节配图是对应环节的处理前后对照，用于帮助理解算法在像素层面的作用，"
        "属于说明性、示例性材料。当前论文脚本所采用的少数几张脱敏或自建案例图，并不能覆盖成绩单类图像在真实环境中的全部情况（例如不同光照、不同教务系统皮肤、"
        "不同手机/扫描仪分辨率、JPEG 压缩与摩尔纹、手写批注、严重褶皱或反光等）。因此插图应与正文公式化表述一并阅读：图示展示的是「在该类输入下流水线大致如何改变图像」，"
        "而非对任意样本都能达到与图完全一致的视觉效果。若答辩或论文修订希望增强说服力，建议按成像类型分别补充案例，并在图注中注明分辨率、设备与是否脱敏。",
    )
    add_body_paragraph(
        doc,
        "从实验材料角度，较理想的案例图可按下列方向准备（与 4.1—4.6 的侧重点对应，可多张图分工，不必强求一张图包办）："
        "（1）4.1：常规 PNG/JPG 即可；若有含透明通道的截图类 PNG，更能体现解码与后续分支差异。（2）4.2：带明显白边的小尺寸截图、或 BGRA 需与白底合成的素材，"
        "以及灰度单通道源图，用于展示归一化与通道统一。（3）4.3：主体纸张置于桌面/木纹等复杂背景上、非成绩表区域占比较大的拍照，便于观察「归一化后→白纸掩膜预裁切」与线上一致分支。"
        "（4）4.4：侧向或俯视斜拍导致纸面呈明显梯形的照片，四边轮廓相对完整，便于透视拉直前后对比。（5）4.5：整体存在可感知倾斜（约数度至十余度）的图像，"
        "便于突出旋转纠偏。（6）4.6：表格 ROI 已较集中、且分辨率偏低或边长较短的小图，便于观察白边 padding 与放大等送 OCR 前增强效果。"
        "上述每一类各准备若干张经脱敏处理的样本并记录元数据，即可与第 6 章数据集划分（A—D 类）相互印证。",
    )
    add_body_paragraph(
        doc,
        "本章后半部分（4.7—4.9）转入识别与服务层：说明 PaddleOCR 调用封装、版本兼容、HTTP 接入方式，并在 4.9 给出统一的 OCRItem 中间表示，"
        "作为第5章表格结构恢复的直接输入。",
    )
    add_heading(doc, "4.1　图像表示、解码与调试输出", 2)
    add_body_paragraph(
        doc,
        "系统内部统一使用OpenCV{{6}}的 BGR uint8 ndarray 表示彩色图。decode_image 将上传字节经 np.frombuffer 与 cv2.imdecode 解码，"
        "对无法解码的文件抛出明确 ValueError，由上层转换为 HTTP 错误信息。带 alpha 的 PNG 在后续归一化中单独保留透明通道，"
        "以便在“截图类透明背景”场景下跳过纸张预裁切与透视校正分支（与 preprocess_with_context 一致）。",
    )
    add_body_paragraph(
        doc,
        "从直观角度理解，一张图像是按二维网格排列的彩色像素点阵；OpenCV 提供读图、缩放、色彩与几何变换、阈值与形态学等算子，"
        "不改变“文字语义由深度学习模型推断”的总体分工，但负责成像链路中可解释、低成本的一段：例如用透视变换把斜拍纸张拉成近似矩形、"
        "用阈值与形态学突出表格线、用旋转抵消小幅倾斜，使笔画在像素层面更分离、对比更强，后续检测与识别更容易对齐行向与笔画边缘。",
    )
    add_body_paragraph(
        doc,
        "OCRService.preprocess_stage_snapshots 与线上 preprocess_with_context 逐步对齐，将归一化、白纸掩膜预裁切（非透明分支）、透视、纠偏、表格/标题 ROI 裁切与送 OCR 前增强等各步之后的图像分别导出，"
        "便于对照观察“哪一步开始劣化”。第 4.1—4.6 节各配一张「处理前 | 处理后」对照图：默认插入 eg/fig_4_1.png—fig_4_6.png，"
        "由 tools/build_chapter4_eg_figures.py 基于「Snipaste_2026-04-13_10-26-13.png」合成 case_4_1—case_4_5；case_4_6 为 case_4_5 经流水线纠偏后再缩小，再对各案例走同一预处理链生成对照图；"
        "图内中文标题与左右说明与 tools/build_thesis_case_figures.py 共用 Pillow 绘制逻辑。"
        "若 eg/ 下尚无插图，生成论文脚本会回退使用 testdemo/output/fig_4_*.png（单主案例）。更换基准图或合成规则可编辑 build_chapter4_eg_figures.py。",
    )
    add_thesis_figure(
        doc,
        resolve_fig4(1),
        "图4-1　4.1 节：解码与预处理入口前后对照（案例：eg/case_4_1.png，由基准 Snipaste 图合成；图中为中文标注）",
        width=Inches(5.2),
    )

    add_heading(doc, "4.2　输入归一化与透明通道处理", 2)
    add_body_paragraph(
        doc,
        "_normalize_input_image 负责灰度转三通道、以及将 BGRA 合成白底，避免透明区域在二值化中被误判为黑块。"
        "对截图类素材，合成白底可显著提升后续阈值分割与 OCR 对比度稳定性。",
    )
    add_body_paragraph(
        doc,
        "小图增强策略通过最小边长、白边 padding 与缩放系数控制：小分辨率截图中笔画覆盖像素过少，检测框易漂移，适度放大等价于提高输入 DPI，"
        "是成本极低、收益常显著的工程技巧（与 4.6 节及图4-6 对应）。",
    )
    add_thesis_figure(
        doc,
        resolve_fig4(2),
        "图4-2　4.2 节：归一化与透明通道前后对照（案例：eg/case_4_2.png，BGRA边缘半透明）",
        width=Inches(5.2),
    )

    add_heading(doc, "4.3　白纸区域检测与主体裁切", 2)
    add_body_paragraph(
        doc,
        "当输入为办公桌木纹背景上的拍照成绩单时，整图直接 OCR 会引入大量无关检测框。系统在归一化之后、透视校正之前（preprocess_with_context 中非透明图分支），"
        "先用 HSV 与 LAB 组合阈值提取偏亮区域，辅以形态学开闭运算去噪，再在轮廓集合中选择面积与矩形度符合“纸张”假设的外接矩形，得到 crop_bbox 并做整页级预裁切。"
        "裁切后更新齐次矩阵 forward_matrix，使服务端解析管线内在多步几何变换下保持坐标一致，并为二次聚焦等裁剪步骤提供可复合的变换上下文。"
        "原理上，阈值分割把像素按「是否像纸」分为两类，形态学开闭用于填小孔、断小梗，轮廓分析则在离散边界上选取最符合面积与长宽比约束的候选。",
    )
    add_thesis_figure(
        doc,
        resolve_fig4(3),
        "图4-3　4.3 节：归一化后与白纸掩膜预裁切前后对照（案例：eg/case_4_3.png，桌面背景）",
        width=Inches(5.2),
    )

    add_heading(doc, "4.4　透视校正与几何变换矩阵", 2)
    add_body_paragraph(
        doc,
        "_detect_document_quad 在（可能已做纸张预裁切的）前景上寻找文档四边形，若成功则执行透视变换将梯形拉成近似矩形。"
        "对透明背景截图，算法上跳过纸张预裁切与透视步骤，以避免缺少真实纸缘时的虚假顶点估计。每一步几何操作均左乘到 3×3 变换矩阵，"
        "保证可逆；最终在上下文中返回 original_size 与 processed_size，供解析与聚焦等后续步骤在统一度量下使用。"
        "原理上，先在灰度图上经高斯平滑削弱噪声，再用 Canny 得到边缘，经膨胀与闭运算连接断缘，在候选轮廓上用多边形近似筛出四角点；"
        "透视变换由单应矩阵描述平面到平面的映射，warpPerspective 将源四边形像素重采样到目标矩形网格。",
    )
    add_thesis_figure(
        doc,
        resolve_fig4(4),
        "图4-4　4.4 节：透视校正前后对照（案例：eg/case_4_4.png，强透视；未检出四边形时左右可能相近）",
        width=Inches(5.2),
    )

    add_heading(doc, "4.5　旋转纠偏与文档朝向", 2)
    add_body_paragraph(
        doc,
        "_deskew_rotate_with_matrix 在限定最大角度范围内估计倾斜角：优先利用纸面主方向；失败时退化到霍夫直线与投影方差最小化等经典策略。"
        "纠偏对表格线提取与 OCR 行顺序稳定性至关重要；但过大角度旋转会引入插值模糊，因此需与后续“表格 ROI 裁切”协同权衡。"
        "原理简述：纸面掩膜上非零像素的最小外接矩形给出整体倾斜；霍夫分支统计近似水平线段倾角的中位数；投影分支则在若干小角度上旋转二值图，使行间投影方差最大以对齐行向。",
    )
    add_thesis_figure(
        doc,
        resolve_fig4(5),
        "图4-5　4.5 节：旋转纠偏前后对照（案例：eg/case_4_5.png，整页倾斜；左为透视后，右为纠偏后）",
        width=Inches(5.2),
    )

    add_heading(doc, "4.6　表格与标题区域聚焦及小图增强", 2)
    add_body_paragraph(
        doc,
        "_crop_to_table_or_title_with_matrix 在纠偏后的整页图像中抑制大面积空白与页脚说明文字，使识别资源集中于成绩表主体（与 preprocess_with_context 中透视、纠偏之后的表格/标题 ROI 一步一致）。"
        "该步骤结合表格线候选与文字连通域分布估计内容区域，属于面向业务的启发式裁剪：对灰度图做自适应阈值得到反色二值图后，用横向与纵向长条结构元素做形态学开运算分离横线与竖线，"
        "再融合为网格掩膜并取最大连通区域估计表格边界；若无可靠表格框则退化为按标题行高度裁掉页顶冗余。"
        "随后 _enhance_for_ocr_with_matrix 在最短边小于设定阈值时对图像做白边 padding 与固定倍率放大（双三次插值），等价于提高输入分辨率以利于检测框稳定；图4-6给出 ROI 与增强前后对照。",
    )
    add_thesis_figure(
        doc,
        resolve_fig4(6),
        "图4-6　4.6 节：表格/标题 ROI 与小图增强（送 OCR 前）前后对照（案例：eg/case_4_6.png，由 case_4_5 纠偏后缩小）",
        width=Inches(5.2),
    )

    add_heading(doc, "4.7　OCR 调用封装、版本兼容与 recognize_text", 2)
    add_body_paragraph(
        doc,
        "recognize_text 在可选预处理后将图像送入PaddleOCR{{1}}，cls=True 启用角度分类；若版本不支持 cls 参数则捕获异常回退到无 cls 调用。"
        "对 3.x 字典式返回与 2.x 列表式返回分别解析，统一写入 OCRItem。",
    )
    add_code_block(
        doc,
        """def recognize_text(self, image_bgr: np.ndarray, *, preprocess: bool = True) -> List[OCRItem]:
    if preprocess:
        image_bgr = self.preprocess_image(image_bgr)
    ocr = self._get_ocr()
    results = ocr.ocr(image_bgr, cls=True)
    ...
    items.append(OCRItem(text=text, confidence=conf, box=box_pts))
    return items""",
    )
    add_body_paragraph(
        doc,
        "_create_with_compat 通过正则解析 Unknown argument 并迭代剔除无效 kwargs，解决 PaddleOCR 升级后参数名变化带来的部署摩擦，"
        "对实验可复现性与课程设计评分中的“可运行性”尤为关键。",
    )
    add_code_block(
        doc,
        """def _create_with_compat(factory, kwargs: Dict[str, Any]):
    k = dict(kwargs)
    for _ in range(20):
        try:
            return factory(**k)
        except Exception as e:
            m = re.search(r"Unknown argument:\s*([A-Za-z_][A-Za-z0-9_]*)", str(e))
            if m and m.group(1) in k:
                k.pop(m.group(1), None)
                continue
            raise""",
    )

    add_heading(doc, "4.7.1　PaddleOCR 模型角色与工作原理（简释）", 3)
    add_body_paragraph(
        doc,
        "工程上可将 PaddleOCR{{1}} 视作“检测 + 识别 +（可选）方向分类”的组合流水线：检测模型在整图上预测文字区域的位置与轮廓，相当于回答“字在哪儿”；"
        "识别模型再对每个文字条带做序列推断，相当于回答“这些像素排成什么字符串”。"
        "检测侧常见思路包括基于分割图的可微二值化（Differentiable Binarization，DB）等，以兼顾速度与轮廓质量；识别侧多采用卷积特征提取配合序列解码（如 CTC），"
        "把二维文字条带转为一维特征序列再输出字符或子词。",
    )
    add_body_paragraph(
        doc,
        "本系统在 recognize_text 中启用 cls=True 时，由角度分类器判断行是否倒置，可减少整行 180° 颠倒造成的误读。"
        "代码层对 PaddleOCR 2.x 列表式返回与 3.x 字典式返回做了兼容解析，统一为 OCRItem（文本、置信度、四点框），供表格聚类、线网格落格与局部补识别复用。",
    )

    add_heading(doc, "4.8　HTTP 服务层设计", 2)
    add_body_paragraph(
        doc,
        "main.py 将 FastAPI 实例与静态目录、CORS、全局服务单例组装在一起。上传路由仅负责读取字节与调用 parser_service.parse，"
        "不在路由层展开业务分支，符合单一职责。导出路由校验 payload 结构，避免恶意超大 JSON 直接进入写文件逻辑。",
    )
    add_code_block(
        doc,
        """@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    result = parser_service.parse(
        filename=file.filename or f"upload-{uuid.uuid4().hex}",
        content_type=file.content_type or "",
        data=raw,
    )
    return JSONResponse(content=result)""",
    )
    add_body_paragraph(
        doc,
        "第4章至此说明如何将请求接入服务并调用解析入口；下一小节给出 OCR 侧统一的数据结构，使“识别到了什么”在代码中有明确、可序列化的形态，"
        "并作为第5章表格恢复与业务解析的直接输入。",
    )

    add_heading(doc, "4.9　OCR 输出数据结构与中间表示", 2)
    add_body_paragraph(
        doc,
        "recognize_text 将 PaddleOCR 各版本返回格式解析后，统一为不可变数据类 OCRItem：text 为识别字符串，confidence 为模型给出的置信度，"
        "box 为四个顶点构成的平面四边形，顶点坐标为浮点数列表 [[x0,y0],…,[x3,y3]]，顺序与检测器输出一致。"
        "一次整图（或 ROI）识别得到 List[OCRItem]，即若干条相互独立的文字条带观测；该列表本身不携带“第几行第几列”的表格语义，"
        "也不保证阅读顺序，这正是第5章需要解决的“从几何观测到网格”的鸿沟。",
    )
    add_body_paragraph(
        doc,
        "在本项目中，该中间表示被多处复用：估计表格/标题包围盒与二次聚焦 ROI、从全图文本中正则抽取成绩单元信息、"
        "将文本落入聚类或线网格单元格、以及对空数值格做局部补识别时的空间对齐。下列代码与示意与 backend/services/ocr_service.py 一致。",
    )
    add_code_block(
        doc,
        """@dataclass(frozen=True)
class OCRItem:
    text: str
    confidence: float
    box: List[List[float]]""",
    )
    add_code_block(
        doc,
        """# 示意：两张学生姓名附近的局部观测（坐标随图像分辨率变化）
[
  {"text": "姓名", "confidence": 0.98, "box": [[120.0, 40.0], [165.0, 38.0], [166.0, 62.0], [121.0, 64.0]]},
  {"text": "张三", "confidence": 0.95, "box": [[118.0, 72.0], [158.0, 71.0], [159.0, 94.0], [119.0, 95.0]]}
]""",
    )
    add_body_paragraph(
        doc,
        "表格结构恢复模块 recognize_table 在工程上的核心输出是二维字符串网格 List[List[str]]（行优先，单元格内文本可为拼接后的字符串），"
        "再经 FileParserService._extract_class_grid 等步骤映射为对外 API 中的 headers 与 rows。"
        "因此数据流可概括为：图像 → List[OCRItem] →（多路径）网格 → 业务字段；第5章仅分三节概述核心步骤，并附关键代码片段。",
    )

    add_heading(doc, "第5章　表格结构恢复与业务解析", 1)
    add_body_paragraph(
        doc,
        "第4章末给出的 List[OCRItem] 仅描述“条带文字与框”，不含行列索引；本章将其收敛为 List[List[str]] 再映射为班级表业务字段。"
        "下面依三条主线展开：多路径表格恢复（几何聚类、可选线网格、PaddleX{{2}} HTML 兜底及 _grid_score 仲裁）、解析管线与 _extract_class_grid（元信息、聚焦、表头与碎行、补识别等）、前端展示与 REST 契约。",
    )

    add_heading(doc, "5.1　多路径表格恢复与路径决策", 2)
    add_body_paragraph(
        doc,
        "主路径为 _items_to_grid：将每个 OCRItem 转为轴对齐框与中心点，按 y 聚类为行、按 x 聚类为列，得到 grid_from_items，适合列对齐明显的截图类成绩单。"
        "线结构分支对灰度图二值化后用形态学抽出长横线与长竖线，仅当竖线/横线主导度均较高且当前聚类分尚不理想时，将 mask 分解为单元格并调用 _line_cells_to_grid 将 OCR 文本落入格内。"
        "若 _grid_score 仍不满意，则进入 PaddleX table_recognition：对 predict 结果中的 html 调用 _html_table_to_grid；该路径利于复杂表，但依赖重、冷启动慢。",
    )
    add_body_paragraph(
        doc,
        "_grid_score 用表头关键词命中、行列规模、首行空单元惩罚等加权为各候选网格打分；recognize_table 在分支间比较分数与阈值（如聚类分已较高则早停返回），体现先轻后重、择优回退。下列代码摘录与 backend/services/ocr_service.py 一致，为版面略有省略。",
    )
    add_code_block(
        doc,
        """def recognize_table(self, image_bgr, *, ocr_items=None, preprocess=True):
    if preprocess:
        image_bgr = self.preprocess_image(image_bgr)
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
        pass

    if score_items >= 20:
        return grid_from_items

    try:
        engine = self._get_table_engine()
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
    return best_grid if best_grid and len(best_grid) >= 2 else None""",
    )

    add_heading(doc, "5.2　解析管线与业务网格抽取", 2)
    add_body_paragraph(
        doc,
        "FileParserService._parse_image 将第4章预处理、OCR 与上一节 recognize_table 串成一条可执行链：先后 recognize_text、_extract_transcript_meta、可选 _focus_table_region 与二次 recognize_text；"
        "再将 ocr_items 与当前图传入 recognize_table；若得到合法二维表则进入 _extract_class_grid，在其中完成动态表头、别名归一、空列剔除、碎行合并、_estimate_cell_regions、_recover_numeric_cells 与 _normalize_transcript_rows；"
        "其中 _extract_class_grid 返回前会调用 _strip_highlight_fields，去掉仅服务内部的单元格多边形等字段。失败则走 _extract_from_text_lines_fallback。下列为核心骨架（摘自 backend/services/parser_service.py）。",
    )
    add_code_block(
        doc,
        """def _parse_image(self, data: bytes) -> Dict[str, Any]:
    image_bgr = self.ocr_service.decode_image(data)
    preprocess_result = self.ocr_service.preprocess_with_context(image_bgr)
    image_bgr = preprocess_result["image"]
    transform_context = preprocess_result["context"]
    image_size = transform_context["original_size"] if transform_context else None

    ocr_items = self.ocr_service.recognize_text(image_bgr, preprocess=False)
    meta = self._extract_transcript_meta(ocr_items)
    image_bgr, transform_context, focused = self._focus_table_region(
        image_bgr, ocr_items, transform_context
    )
    if focused:
        ocr_items = self.ocr_service.recognize_text(image_bgr, preprocess=False)
        meta = self._extract_transcript_meta(ocr_items)

    try:
        table = self.ocr_service.recognize_table(
            image_bgr, ocr_items=ocr_items, preprocess=False
        )
    except Exception:
        table = None

    if table and len(table) >= 2:
        result = self._extract_class_grid(
            table,
            image_bgr=image_bgr,
            ocr_items=ocr_items,
            default_conf=0.85,
            image_size=image_size,
            transform_context=transform_context,
        )
        result["processed_preview"] = self._encode_preview_image(image_bgr)
        if meta:
            result["meta"] = meta
            if not result.get("title") and meta.get("标题"):
                result["title"] = str(meta["标题"]).strip()
        return result

    result = self._extract_from_text_lines_fallback(
        [it.text for it in ocr_items],
        meta=meta,
        image_size=image_size,
        ocr_items=ocr_items,
    )
    result["processed_preview"] = self._encode_preview_image(image_bgr)
    return result""",
    )

    add_heading(doc, "5.3　交互、导出与接口数据契约", 2)
    add_body_paragraph(
        doc,
        "前端静态页提供上传、原图与预处理预览（可放大）、可编辑表格及按 confidences 的低分标红；用户确认后 POST /export，由 openpyxl{{10}} 写入样式化的 xlsx。"
        "相对 4.9 的逐条 OCRItem，POST /upload 成功体收敛为 headers、rows（含 values 与 confidences）、可选 title、meta、processed_preview；单元格多边形等不再下发。",
    )
    add_code_block(
        doc,
        """// 典型成功响应（字段视样本略有增减；不含单元格多边形）
{
  "headers": ["姓名", "班级", "平时成绩", "期末成绩"],
  "rows": [
    {
      "values": {"姓名": "张三", "班级": "计科1班", "平时成绩": 88, "期末成绩": 90},
      "confidences": {"姓名": 0.99, "班级": 0.97, "平时成绩": 0.95, "期末成绩": 0.96}
    }
  ],
  "title": "…",
  "meta": {"姓名": "…", "学号": "…"},
  "processed_preview": "data:image/png;base64,..."
}""",
    )

    add_heading(doc, "第6章　系统实现、测试与实验分析", 1)
    add_body_paragraph(
        doc,
        "本章按“如何运行系统 → 如何用 unittest 做回归验证 → 如何用插图佐证界面与预处理效果 → 如何归纳模块分工与部署注意点”的顺序组织，"
        "与第4章、第5章的算法叙述形成“可运行、可验证、可展示”的收尾。",
    )

    add_heading(doc, "6.1　实现环境与部署运行", 2)
    add_body_paragraph(
        doc,
        "建议开发与实验环境为 Windows 10/11 64 位、16GB 及以上内存、支持 AVX 的 x64 CPU，Python 3.9 及以上虚拟环境。"
        "关键依赖与仓库 requirements 一致：fastapi、uvicorn[standard]、paddleocr、paddlepaddle、opencv-python、openpyxl{{10}}、beautifulsoup4{{11}}、lxml 等；"
        "若需 SVG 栅格化可补充 cairosvg。可按官方说明设置 PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK 以减轻表格流水线冷启动时的模型源检查开销。",
    )
    add_body_paragraph(
        doc,
        "代码布局：backend 为 FastAPI 应用与领域服务（OCRService、FileParserService、ExcelService），frontend 为静态资源，tests 为回归用例，"
        "tools 含论文插图流水线等辅助脚本。启动命令为 "
        "`python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000`，浏览器访问 http://localhost:8000/ 进入主界面。",
    )

    add_heading(doc, "6.2　端到端功能与流程说明", 2)
    add_body_paragraph(
        doc,
        "从用户视角，一次完整使用闭环为：选择 PNG/JPG/SVG/XLSX → POST /upload → 查看原图与预处理预览并对照右侧可编辑表格 → 修正低置信度单元格 → POST /export 下载 xlsx。"
        "与第3章需求一致：图像类输入走 decode、preprocess_with_context、recognize_text 得到 List[OCRItem]，再经聚焦（可选）、recognize_table 与 _extract_class_grid 得到 headers/rows；"
        "JSON 中不包含服务端内部用过的单元格多边形，仅保留业务字段与 confidences，见第5章。",
    )
    add_body_paragraph(
        doc,
        "异常路径包括空文件、无法解码、扩展名不在支持列表、OCR 或表格引擎内部异常等；路由层将错误映射为 4xx/5xx，前端宜统一提示并建议用户缩小体积或更换素材。",
    )

    add_heading(doc, "6.3　自动化测试与回归验证", 2)
    add_body_paragraph(
        doc,
        "tests/test_api.py 覆盖 /upload、/export、/health 的契约与典型错误码；tests/test_parser_service.py 对解析结果结构做回归；"
        "tests/test_excel_service.py 检查导出工作簿的冻结窗格、筛选与基本样式。仓库根目录执行 `python -m unittest discover -s tests -v` 应全部通过，"
        "适合纳入持续集成，作为依赖升级后的第一道闸门。答辩材料中可附录命令行全绿截图，与正文相互印证。",
    )

    add_heading(doc, "6.4　系统界面与实验插图", 2)
    add_body_paragraph(
        doc,
        "插图按阅读顺序编排：图6-1 为整页主界面；图6-2 为识别结果表格特写；图6-3 为固定样本原图与 preprocess_with_context 输出对照（与 /upload 输入一致）；"
        "图6-4 为同一样本完整预处理后的单幅结果（可选，版芯紧张时可省略）。印刷前可将 PNG 换为高分辨率截图；图6-3、图6-4 可与第4章预处理各节叙述对照阅读。",
    )

    add_figure_placeholder(
        doc,
        "【图6-1 插入位置】系统 Web 主界面全屏截图：上传区、原图与预处理预览、右侧可编辑表格同框。",
    )

    if FIG_SRC.exists():
        doc.add_paragraph()
        pic_p = doc.add_paragraph()
        pic_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pic_p.add_run().add_picture(str(FIG_SRC), width=Inches(5.2))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cr = cap.add_run("图6-2　识别结果示例：班级成绩片段表格展示（可替换为自有截图）")
        set_run_font(cr, east_asia="宋体", ascii_font="Times New Roman", size=Pt(10.5))
    else:
        add_figure_placeholder(
            doc,
            "【图6-2】请将识别结果表格截图置于 generate_thesis_2252733.py 中 FIG_SRC 路径，或生成后手工插入。",
        )

    add_figure_placeholder(
        doc,
        "【可选配图】低置信度标红单元格与编辑态特写；若不单独占版面，建议与图6-1 合并为一张截图，仍统称图6-1。",
    )

    add_thesis_figure(
        doc,
        FIG6_4_A,
        "图6-3　实验用样本：原图与 preprocess_with_context 输出对照（与线上 /upload 一致；第4章分节案例见 eg/fig_4_*.png）",
        width=Inches(5.4),
    )
    add_thesis_figure(
        doc,
        FIG6_4_B,
        "图6-4　同一样本经完整预处理后的单幅结果（可选；默认来自 out/test2/01_full_preprocess_pipeline.png）",
        width=Inches(4.2),
    )

    add_heading(doc, "6.5　消融讨论、性能与改进方向", 2)
    add_body_paragraph(
        doc,
        "与第5章模块的对应关系如下：检测与识别由 PaddleOCR 承担，cls 改善倒置行；几何与表格线由 OpenCV 经典算子承担；结构歧义由多路径评分与 PaddleX{{2}} 兜底。"
        "若需做消融，可在实现中分别旁路：白纸裁切、透视校正、二次聚焦、线结构分支、PaddleX 兜底、数字补识别，在典型样本（如斜拍、弱线表）上对比解析结果与耗时。",
    )
    add_body_paragraph(
        doc,
        "性能与部署方面：CPU 推理受线程数、模型缓存与输入分辨率影响；PaddleX 首次 predict 往往显著慢于热路径，宜区分冷启动与稳态耗时表述。"
        "响应体中的 processed_preview 为 Base64 内嵌图，会放大 JSON；弱网环境可改为对象存储 URL 或缩略图策略。",
    )
    add_body_paragraph(
        doc,
        "可维护性方面：以 unittest 全量 discover 作为接口与核心逻辑的回归抓手；依赖或 Paddle 版本升级后应先跑测试再验收界面。"
        "答辩材料中测试通过截图建议与第6.3节相互引用。",
    )

    add_heading(doc, "第7章　总结与展望", 1)
    add_body_paragraph(
        doc,
        "本文完成了面向学生成绩单场景的端到端识别系统，涵盖预处理、OCR、多路径表格恢复、解析纠错与浏览器端校对导出。"
        "不足之处在于：极端反光、严重卷曲、手写批注等样本仍具挑战；PaddleX{{2}}兜底路径计算开销较大。"
        "后续可探索轻量表格 Transformer、主动学习筛选难例，以及与教务 API 的直连同步{{5}}。",
    )

    doc.add_page_break()
    add_heading(doc, "参考文献", 1)
    refs = [
        "[1] 百度飞桨. PaddleOCR 文档与模型库[EB/OL]. https://github.com/PaddlePaddle/PaddleOCR （项目默认 OCR 推理与模型说明）",
        "[2] 百度飞桨. PaddleX 文档与表格识别流水线[EB/OL]. https://github.com/PaddlePaddle/PaddleX （table_recognition 等能力说明）",
        "[3] Bradski G, Kaehler A. Learning OpenCV: Computer Vision with the OpenCV Library[M]. Sebastopol: O'Reilly Media, 2008.",
        "[4] Ramírez S. FastAPI documentation[EB/OL]. https://fastapi.tiangolo.com/ ; "
        "Encode OSS Ltd. Uvicorn documentation[EB/OL]. https://www.uvicorn.org/ （与 requirements 中 fastapi、uvicorn 对应）",
        "[5] Zhong X, ShafieiBavani E, Jimeno-Yepes A. Image-based table recognition: data, model, and evaluation[C]//Computer Vision—ECCV 2020. "
        "Lecture Notes in Computer Science, Vol. 12366. Cham: Springer, 2020: 564-580. （表格图像→结构化 HTML 等问题的经典工作，与 PaddleX 表格场景相关）",
        "[6] OpenCV Team. OpenCV documentation[EB/OL]. https://docs.opencv.org/4.x/ （与 opencv-python 文档一致）",
        "[7] 百度飞桨. PaddlePaddle 官方文档[EB/OL]. https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html （与 paddlepaddle 依赖对应）",
        "[8] Shi B, Bai X, Yao C. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition[J]. "
        "IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017, 39(11): 2298-2304.",
        "[9] Liao M, Wan Z, Yao C, et al. Real-time scene text detection with differentiable binarization[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2020, 34(07): 11474-11481.",
        "[10] openpyxl development team. openpyxl documentation[EB/OL]. https://openpyxl.readthedocs.io/en/stable/ （与 requirements 中 openpyxl 对应）",
        "[11] Richardson L. Beautiful Soup documentation[EB/OL]. https://www.crummy.com/software/BeautifulSoup/bs4/doc/ ; "
        "lxml.de. lxml - XML and HTML with Python[EB/OL]. https://lxml.de/ （与 beautifulsoup4、lxml 依赖对应）",
    ]
    for r in refs:
        add_reference_paragraph(doc, r)

    doc.add_paragraph()
    add_heading(doc, "致　谢", 1)
    add_body_paragraph(
        doc,
        "感谢上海海洋大学提供的培养平台，感谢指导老师在选题、研究与论文撰写过程中给予的悉心指导，"
        "感谢同学与开源社区在项目依赖与资料查阅方面提供的帮助。",
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        doc.save(str(OUT_DOCX))
        print("Saved:", OUT_DOCX)
    except PermissionError:
        alt = OUTPUT_ROOT / "2252733-\u6797\u5b50\u6986-\u8bba\u6587-\u5907\u4efd.docx"
        doc.save(str(alt))
        print("目标论文文件可能被 Word 占用，已改为保存到:", alt)


def main() -> None:
    build_docx()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    copies: list[tuple[Path, Path]] = [
        (FIG_SRC, OUTPUT_ROOT / "2252733-\u8bba\u6587-\u56fe6-1.png"),
        (resolve_fig4(1), OUTPUT_ROOT / "2252733-\u8bba\u6587-\u56fe4-1.png"),
        (resolve_fig4(2), OUTPUT_ROOT / "2252733-\u8bba\u6587-\u56fe4-2.png"),
        (resolve_fig4(3), OUTPUT_ROOT / "2252733-\u8bba\u6587-\u56fe4-3.png"),
        (resolve_fig4(4), OUTPUT_ROOT / "2252733-\u8bba\u6587-\u56fe4-4.png"),
        (resolve_fig4(5), OUTPUT_ROOT / "2252733-\u8bba\u6587-\u56fe4-5.png"),
        (resolve_fig4(6), OUTPUT_ROOT / "2252733-\u8bba\u6587-\u56fe4-6.png"),
        (FIG6_4_A, OUTPUT_ROOT / "2252733-\u8bba\u6587-\u56fe6-4a.png"),
        (FIG6_4_B, OUTPUT_ROOT / "2252733-\u8bba\u6587-\u56fe6-4b.png"),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)


if __name__ == "__main__":
    main()

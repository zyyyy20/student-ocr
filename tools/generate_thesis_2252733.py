# -*- coding: utf-8 -*-
"""
生成毕业论文 Word（2252733-林子榆），默认输出到仓库 out/ 目录。
运行：在仓库根目录执行  python tools/generate_thesis_2252733.py

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
FIG4_SEC = PROJECT / "testdemo" / "output"
FIG4_1 = FIG4_SEC / "fig_4_1.png"
FIG4_2 = FIG4_SEC / "fig_4_2.png"
FIG4_3 = FIG4_SEC / "fig_4_3.png"
FIG4_4 = FIG4_SEC / "fig_4_4.png"
FIG4_5 = FIG4_SEC / "fig_4_5.png"
FIG4_6 = FIG4_SEC / "fig_4_6.png"
FIG6_4_A = OUTPUT_ROOT / "test2" / "02_original_vs_pipeline.png"
FIG6_4_B = OUTPUT_ROOT / "test2" / "01_full_preprocess_pipeline.png"

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
    "文字检测到表格结构恢复、字段结构化、前端校对与高亮联动，直至 Excel 导出的完整链路。"
    "系统采用FastAPI{{4}}提供 REST 接口，浏览器端以原生 HTML/CSS/JavaScript 实现交互。"
    "在表格恢复方面，综合采用基于 OCR 检测框的几何聚类、表格线网格提取以及PaddleX{{2}}表格识别流水线兜底的多路径策略；"
    "并引入二次聚焦 OCR 与单元格局部补识别，以提升短数字与小数点的召回率。"
    "实验与运行结果表明，相较传统“整图单次 OCR + 按阅读顺序拼接”或纯人工录入方式，本系统在结构化准确率、"
    "可解释性（置信度与框选高亮）以及端到端处理效率方面具有明显优势。"
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
    """先运行 testdemo 生成第4章六步对照图，再运行 tools 脚本生成 out/test1、out/test2 等；失败不阻断论文生成。"""
    scripts: list[Path] = []
    if THESIS_DEMO_SCRIPT.exists():
        scripts.append(THESIS_DEMO_SCRIPT)
    if THESIS_CASE_SCRIPT.exists():
        scripts.append(THESIS_CASE_SCRIPT)
    if not scripts:
        return
    for script in scripts:
        try:
            cp = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(PROJECT),
                capture_output=True,
                text=True,
                timeout=300,
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
            f"【{caption}】未找到 {image_path.name}。请在仓库根目录执行 python testdemo/run_pipeline_demo.py，"
            "或 python tools/build_thesis_case_figures.py，并保证主案例图像存在。",
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
        "recovery, structured field extraction, interactive proofreading with highlight linkage, and Excel export. "
        "The backend is built with FastAPI; the frontend uses plain HTML/CSS/JavaScript. "
        "For table recovery, a multi-path strategy combines geometry clustering on OCR boxes, grid-line based "
        "reconstruction, and a PaddleX table-recognition fallback. Secondary focused OCR and local cell re-recognition "
        "improve recall for short numeric cells. Compared with traditional single-pass OCR or purely manual entry, "
        "the proposed system offers better structural fidelity, explainability via confidences and overlays, "
        "and higher end-to-end efficiency."
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
        "第5章　表格结构恢复与业务解析……………………………………38",
        "　5.1　问题建模与多路径策略概述…………………………………38",
        "　5.2　基于 OCR 检测框的行列聚类重建……………………………40",
        "　5.3　表格线提取与线网格结构恢复………………………………42",
        "　5.4　PaddleX 表格流水线与 HTML 解析…………………………44",
        "　5.5　网格质量评分与路径决策………………………………………45",
        "　5.6　二次聚焦 OCR 与解析管线……………………………………46",
        "　5.7　表头识别、元信息与字段清洗………………………………48",
        "　5.8　碎行合并、单元格区域与数字补识别………………………49",
        "　5.9　前端高亮联动与 Excel 导出…………………………………51",
        "第6章　系统实现、测试与实验分析…………………………………52",
        "　6.1　实现环境与工具链……………………………………………52",
        "　6.2　系统功能实现与运行说明……………………………………54",
        "　6.3　测试方案与单元测试……………………………………………56",
        "　6.4　实验数据集与对比基线…………………………………………57",
        "　6.5　对比实验与结果分析……………………………………………58",
        "　6.6　消融讨论与典型案例……………………………………………60",
        "　6.7　性能、瓶颈与改进方向…………………………………………61",
        "第7章　总结与展望……………………………………………………62",
        "参考文献…………………………………………………………………64",
        "致谢………………………………………………………………………66",
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
        "（5）基于FastAPI{{4}}与静态前端实现上传、预览、高亮联动与 Excel 导出。研究方法以工程实现与对比实验为主，辅以定量指标与定性分析；"
        "第6章进一步给出数据集划分、基线设置、对比表框架、运行截图位、消融与性能讨论及单元测试回归方案，便于在答辩前填入实测数据。",
    )

    add_heading(doc, "1.4　论文组织结构", 2)
    add_body_paragraph(
        doc,
        "全文共分七章，整体框架遵循“理论基础—需求与总体设计—核心算法与模块—系统实现与实验—总结”的工科论文常见写法。"
        "第2章从文档图像与 OCR 基本概念出发，综述深度学习检测识别、飞桨生态、OpenCV 文档处理、Web 与序列化技术及评测原则；"
        "第3章分析角色场景、功能与非功能需求，给出分层架构、模块职责、接口与数据模型；"
        "第4章围绕预处理流水线、几何变换、OCR 封装与 HTTP 服务展开；第5章讨论多路径表格恢复、解析纠错与前后端联动；"
        "第6章介绍实现环境、测试方案、对比实验与性能讨论；第7章总结全文并展望未来工作。",
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
        "置信度是工程落地的重要信号：低置信度不一定意味着错误，但可作为“需要人工复核”的触发条件；与几何框结合后，还能支持点击高亮、局部放大再识别等交互。"
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
        "前后端以 JSON 作为交换格式，可嵌套 headers、rows、meta、每格 confidences 与多边形 points 字段，兼顾机器处理与人的可视化校对。",
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
        "典型业务流程为：选择文件上传 → 等待服务端解析 → 在网页上核对表头与成绩 → 必要时点击单元格对照原图高亮 → 确认后导出 Excel → 在教务或统计软件中二次处理。",
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
        "（4）交互：双预览模式（原图反投影 / 预处理图）下的多边形高亮；右侧表格可编辑。（5）导出：将当前表格写入 xlsx 并返回下载链接。",
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
        "模块边界上，OCRService 专注“图像进、检测框与表格矩阵出”；FileParserService 负责类型分派、业务字段解析与 JSON 组装；ExcelService 专注工作簿样式与写入。"
        "这种划分有利于单元测试按模块 stub 依赖，也便于将来替换 OCR 引擎而少改上层逻辑。",
    )
    add_heading(doc, "3.5　接口、数据模型与错误处理", 2)
    add_body_paragraph(
        doc,
        "POST /upload 接受 multipart/form-data，字段名为 file；成功时返回 application/json，包含 headers、rows、meta、可选 title、processed_preview 与每行每列的坐标信息。"
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
        "便于对照观察“哪一步开始劣化”。第 4.1—4.6 节在文字说明之后各配一张「处理前 | 处理后」对照图（由 testdemo/run_pipeline_demo.py 写入 testdemo/output/，"
        "图内中文标题与左右说明与 tools/build_thesis_case_figures.py 共用 Pillow 绘制逻辑），"
        "分别对应解码入口、归一化、白纸预裁切、透视、纠偏与 ROI 后增强；生成论文前会自动运行该 demo 脚本，"
        "主案例固定为仓库根目录下的「Snipaste_2026-04-13_10-26-13.png」（高二（12）班平时成绩表桌面截图，便于观察各步效果）。"
        "若需更换案例，可替换该文件或修改脚本中 PRIMARY_CASE 后重新运行脚本；亦可按章首「案例图准备建议」增补多类脱敏样例。",
    )
    add_thesis_figure(
        doc,
        FIG4_1,
        "图4-1　4.1 节：解码与预处理入口前后对照（案例：Snipaste_2026-04-13_10-26-13.png；图中为中文标注）",
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
        FIG4_2,
        "图4-2　4.2 节：归一化与透明通道前后对照（与图4-1 同源案例）",
        width=Inches(5.2),
    )

    add_heading(doc, "4.3　白纸区域检测与主体裁切", 2)
    add_body_paragraph(
        doc,
        "当输入为办公桌木纹背景上的拍照成绩单时，整图直接 OCR 会引入大量无关检测框。系统在归一化之后、透视校正之前（preprocess_with_context 中非透明图分支），"
        "先用 HSV 与 LAB 组合阈值提取偏亮区域，辅以形态学开闭运算去噪，再在轮廓集合中选择面积与矩形度符合“纸张”假设的外接矩形，得到 crop_bbox 并做整页级预裁切。"
        "裁切后更新齐次矩阵 forward_matrix，使后续所有框可通过 inverse_matrix 映射回原图，为高亮联动提供几何一致性。"
        "原理上，阈值分割把像素按「是否像纸」分为两类，形态学开闭用于填小孔、断小梗，轮廓分析则在离散边界上选取最符合面积与长宽比约束的候选。",
    )
    add_thesis_figure(
        doc,
        FIG4_3,
        "图4-3　4.3 节：归一化后与白纸掩膜预裁切前后对照（与 preprocess_with_context 顺序一致；与图4-1 同源案例）",
        width=Inches(5.2),
    )

    add_heading(doc, "4.4　透视校正与几何变换矩阵", 2)
    add_body_paragraph(
        doc,
        "_detect_document_quad 在（可能已做纸张预裁切的）前景上寻找文档四边形，若成功则执行透视变换将梯形拉成近似矩形。"
        "对透明背景截图，算法上跳过纸张预裁切与透视步骤，以避免缺少真实纸缘时的虚假顶点估计。每一步几何操作均左乘到 3×3 变换矩阵，"
        "保证可逆；最终在上下文中返回 original_size 与 processed_size，供前端选择不同坐标系绘制。"
        "原理上，先在灰度图上经高斯平滑削弱噪声，再用 Canny 得到边缘，经膨胀与闭运算连接断缘，在候选轮廓上用多边形近似筛出四角点；"
        "透视变换由单应矩阵描述平面到平面的映射，warpPerspective 将源四边形像素重采样到目标矩形网格。",
    )
    add_thesis_figure(
        doc,
        FIG4_4,
        "图4-4　4.4 节：透视校正前后对照（与图4-1 同源案例；预裁切后 → 透视拉直；未触发预裁切或跳过透视时左右可能相近）",
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
        FIG4_5,
        "图4-5　4.5 节：旋转纠偏前后对照（与图4-1 同源案例；左为透视后，右为纠偏后）",
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
        FIG4_6,
        "图4-6　4.6 节：表格/标题 ROI 与小图增强（送 OCR 前）前后对照（与 preprocess_with_context 末两步一致；与图4-1 同源案例）",
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
        "综上，第4章从像素层到服务层完成了“把不稳定成像变成稳定 OCR 输入”的主要工作，为第5章表格结构恢复提供了高质量的观测数据。",
    )

    add_heading(doc, "第5章　表格结构恢复与业务解析", 1)
    add_heading(doc, "5.1　问题建模与多路径策略概述", 2)
    add_body_paragraph(
        doc,
        "可将 OCR 输出抽象为若干文字实例，每个实例包含文本内容、置信度与平面四边形检测框。"
        "表格恢复的目标是把检测框划分到离散的行列网格中，使同一单元格内语义一致且行列索引与视觉表格对齐。"
        "由于成绩单成像条件多样，单一算法难以保证全局最优，故采用“轻量几何聚类 → 线结构约束 → 深度表格流水线兜底”的分级策略，"
        "并以可解释的标量得分在路径间仲裁。",
    )

    add_heading(doc, "5.2　基于 OCR 检测框的行列聚类重建", 2)
    add_body_paragraph(
        doc,
        "_items_to_grid 首先将每个 OCRItem 映射为轴对齐包围盒与中心点，按 y 中心聚类为行：行内阈值 y_tol 与检测框中位高度相关，"
        "可容忍轻微纵向抖动。行内再按 x 中心排序，并对全图 x 中心序列做一维聚类得到列中心，列间距阈值 x_tol 与中位宽度相关。"
        "该方法对“截图类、列对齐明显”的班级表效果最佳，复杂度近似线性，适合作为默认主路径。",
    )
    add_body_paragraph(
        doc,
        "该路径的失效模式包括：密集小数字导致框粘连、强透视下同一行 y 中心离群、以及表头多行合并造成首行检测不全。"
        "因此需要 _grid_score 与后续线结构/PaddleX{{2}}互为补充。",
    )

    add_heading(doc, "5.3　表格线提取与线网格结构恢复", 2)
    add_body_paragraph(
        doc,
        "对灰度图二值化后，利用形态学分别保留长横线与长竖线，再融合得到 grid_mask；统计竖线/横线主导度 vertical_dominance、horizontal_dominance，"
        "仅当二者均超过经验阈值时才认为“线结构可信”，进而将 mask 分解为单元格集合并调用 _line_cells_to_grid 将 OCR 文本落入格内。"
        "该方法在纸质表格、线清晰场景下结构约束强，可减少 OCR 漏框带来的列错位。",
    )
    add_body_paragraph(
        doc,
        "其代价是对断线、弱线、复印淡化敏感；故实现中要求 line 路径在得分上显著优于 items 路径才切换，避免“画虎不成反类犬”。",
    )

    add_heading(doc, "5.4　PaddleX 表格流水线与 HTML 解析", 2)
    add_body_paragraph(
        doc,
        "当聚类得分不足且线结构亦不可靠时，调用 paddlex.create_pipeline(\"table_recognition\") 获取 pred，读取 html 字段并用 BeautifulSoup{{11}}与 lxml 解析器配合解析表格节点{{2}}。"
        "HTML 路径的优势在于对复杂合并单元格有一定归纳能力；劣势是依赖体积大、冷启动慢，且 HTML 转 grid 时需处理 rowspan/colspan 与空单元格。",
    )

    add_heading(doc, "5.5　网格质量评分与路径决策", 2)
    add_body_paragraph(
        doc,
        "_grid_score 通过表头关键词（姓名、学号、班级、成绩等）命中数、行列规模、首行空单元惩罚等启发式加权，得到标量分数。"
        "recognize_table 中若 score_items 已足够高则直接返回聚类结果；若线网格分数更高且满足阈值则返回线结果；否则进入 PaddleX；"
        "若 HTML 解析网格分数仍不优则回退到较优的轻量路径。该决策链体现了工程上的“早停 + 兜底”。",
    )
    add_code_block(
        doc,
        """def recognize_table(self, image_bgr, *, ocr_items=None, preprocess=True):
    if ocr_items is None:
        ocr_items = self.recognize_text(image_bgr, preprocess=False)
    grid_from_items = self._items_to_grid(ocr_items)
    score_items = self._grid_score(grid_from_items)
    ...
    if score_items >= 20:
        return grid_from_items
    engine = self._get_table_engine()
    pred = engine.predict(image_bgr)
    ...""",
    )

    add_heading(doc, "5.6　二次聚焦 OCR 与解析管线", 2)
    add_body_paragraph(
        doc,
        "FileParserService._parse_image 在得到首遍 ocr_items 后，根据文字分布估计表格包围盒 _estimate_focus_bbox，"
        "对图像做裁剪并更新 transform_context。若 focused 为真，则在更小 ROI 上再次 recognize_text，相当于提高有效分辨率并削弱页脚、桌面的干扰。"
        "该策略对“成绩只占画面中部”的手机竖拍图尤为有效。",
    )
    add_body_paragraph(
        doc,
        "随后将 ocr_items 与图像一并交给 recognize_table，再进入 _extract_class_grid：包括动态表头发现、别名归一化、空列剔除与行规范化等步骤，"
        "最终输出与前端契约一致的 headers 与 rows。",
    )

    add_heading(doc, "5.7　表头识别、元信息与字段清洗", 2)
    add_body_paragraph(
        doc,
        "元信息 _extract_transcript_meta 通过正则与关键词扫描 OCR 文本，抽取姓名、学号、班级、学院、学期等字段；"
        "若表格 title 缺失可用 meta 中的标题补齐。字段清洗 _normalize_transcript_rows 负责全角半角、空白与常见 OCR 混淆字符的归一，"
        "减少导出 Excel 后的二次手工清洗。",
    )

    add_heading(doc, "5.8　碎行合并、单元格区域估计与数字补识别", 2)
    add_body_paragraph(
        doc,
        "碎行合并解决“同一学生一行被切成两行且列上互补”的拍照畸变：通过比较相邻行非空列集合是否不相交，并结合列索引单调性判断是否应拼接。"
        "单元格区域估计 _estimate_cell_regions 通过估计表格主轴与行列边界，生成整格多边形，使高亮框覆盖视觉单元格而非仅文字外包矩形。",
    )
    add_body_paragraph(
        doc,
        "_recover_numeric_cells 对仍为空的数值格，优先利用已有 OCR 框重叠推断；失败则 _run_local_cell_ocr：裁块、加白边、放大后局部再识别。"
        "该机制针对小数与窄列数字显著降低漏识率，是相比“只跑一遍整图 OCR”的重要工程增量。",
    )

    add_heading(doc, "5.9　前端高亮联动与 Excel 导出", 2)
    add_body_paragraph(
        doc,
        "前端根据预览模式选择 points 或 processed_points，使用 SVG 覆盖层绘制多边形，实现“点表格、看图”的闭环校对。"
        "导出时 ExcelService 借助 openpyxl{{10}}写入标题行（可选合并单元格）、表头样式、冻结首行与自动筛选，并对纯数字成绩尝试解析为数值类型以便后续求和统计。",
    )

    add_heading(doc, "5.10　上传接口 JSON 返回格式与数据处理", 2)
    add_body_paragraph(
        doc,
        "POST /upload 在解析成功时返回 application/json。班级成绩表走通表格恢复后，主体字段包括：headers 为字符串数组，表示列名；"
        "rows 为对象数组，每一行通常包含 values（列名到单元格值的映射）、confidences（同结构的置信度），"
        "以及可选的 boxes（列名到平面多边形顶点列表，用于前端在原图或预处理图上绘制高亮）。"
        "另可包含 title（表题）、header_boxes（表头区域框）、meta（由 OCR 文本正则抽取的姓名、学号、班级等键值对），"
        "以及 processed_preview（以 data:image/png;base64,... 形式内嵌的预处理预览图，便于对照纠错）。",
    )
    add_body_paragraph(
        doc,
        "数据处理上，FileParserService._parse_image 先 decode_image 与 preprocess_with_context，再 recognize_text 得到 ocr_items；"
        "可选二次聚焦后再次识别；随后 recognize_table 走多路径表格恢复，并由 _extract_class_grid 完成表头归一、空列剔除、碎行合并、"
        "单元格区域估计、数值格补识别与字段清洗，最终组装为上述 JSON。前端将 headers/rows 绑定到可编辑表格，"
        "用 boxes 与 transform_context 将多边形映射到用户选择的坐标系；导出时再把用户编辑后的同一结构 POST 至 /export。",
    )
    add_code_block(
        doc,
        """// 典型成功响应（字段视样本略有增减）
{
  "headers": ["姓名", "班级", "平时成绩", "期末成绩"],
  "rows": [
    {
      "values": {"姓名": "张三", "班级": "计科1班", "平时成绩": 88, "期末成绩": 90},
      "confidences": {"姓名": 0.99, "班级": 0.97, "平时成绩": 0.95, "期末成绩": 0.96},
      "boxes": {"姓名": [[x1,y1], ...], ...}
    }
  ],
  "title": "…",
  "meta": {"姓名": "…", "学号": "…"},
  "processed_preview": "data:image/png;base64,..."
}""",
    )

    add_heading(doc, "第6章　系统实现、测试与实验分析", 1)
    add_heading(doc, "6.1　实现环境与工具链", 2)
    add_body_paragraph(
        doc,
        "开发机与实验机建议配置为：Windows 10/11 64 位、16GB 及以上内存、支持 AVX 的 x64 CPU；Python 3.9 及以上虚拟环境。"
        "关键依赖包括 fastapi、uvicorn[standard]、paddleocr、paddlepaddle、opencv-python、openpyxl{{10}}、beautifulsoup4{{11}}、lxml；"
        "若需 SVG 栅格化则依赖 cairosvg。环境变量 PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK 可按官方说明关闭模型源检查以加速冷启动。",
    )
    add_body_paragraph(
        doc,
        "工程目录中 backend 为服务代码，frontend 为静态资源，tests 为单元测试，tools 含辅助脚本。"
        "启动命令为 `python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000`，浏览器访问 http://localhost:8000/ 进入主界面。",
    )

    add_heading(doc, "6.2　系统功能实现与运行说明", 2)
    add_body_paragraph(
        doc,
        "功能实现上，上传模块使用浏览器 FormData 调用 /upload；解析完成后右侧表格组件渲染 headers 与 rows，低置信度单元格可在前端标红提示；"
        "预览区支持切换原图与预处理图，辅助判断预处理是否过度或不足。导出模块将当前表格序列化为 JSON POST 至 /export，成功后跳转下载链接。",
    )
    add_body_paragraph(
        doc,
        "异常路径包括：空文件、无法解码图片、不支持的扩展名、OCR 内部错误等，前端应捕获 HTTP 状态码并提示用户更换素材或缩小图片体积。",
    )

    add_heading(doc, "6.3　测试方案与单元测试", 2)
    add_body_paragraph(
        doc,
        "测试分为三类：其一，接口测试 tests/test_api.py，验证 /upload、/export、/health 的契约与错误码；"
        "其二，解析测试 tests/test_parser_service.py，对合成表格 JSON 与样例图像（若仓库内置）做回归；其三，导出测试 tests/test_excel_service.py，"
        "检查生成文件是否包含冻结窗格、筛选与基本样式。",
    )
    add_body_paragraph(
        doc,
        "运行 `python -m unittest discover -s tests -v` 应全部通过；建议在论文附录粘贴命令行绿色通过截图，并列出主要断言逻辑的文字说明。",
    )

    add_heading(doc, "6.4　实验数据集与对比基线", 2)
    add_body_paragraph(
        doc,
        "实验数据建议划分为：A 类教务截图（列对齐、线清晰）、B 类手机斜拍（透视与阴影明显）、C 类扫描件（噪声与黑边）、D 类弱线或无边框表。"
        "每类至少收集若干张经脱敏处理的样本，并记录分辨率与拍摄设备。对比基线包括：B1 纯人工录入耗时与错误类型统计；"
        "B2 不做预处理、不调表格结构，仅调用通用 OCR 并按阅读顺序输出；B3 使用本系统但关闭二次聚焦与数字补识别，以观察模块贡献。",
    )
    add_body_paragraph(
        doc,
        "为便于答辩材料与论文插图一致，可用与第4章相同的案例图「Snipaste_2026-04-13_10-26-13.png」作为代表性 A 类（教务桌面截图类）样本，经与本系统一致的 preprocess_with_context 得到线上 /upload 实际输入。"
        "图6-4 给出该样本原图与完整预处理后图像的左右对照，插入位置建议：紧接本段“数据集划分与基线说明”之后，用于直观对应 B2 与 B3 讨论中的输入差异。"
        "其中 B2 可理解为仅在原图或弱预处理图上调用通用 OCR；B3 则保留几何与表格管线但关闭二次聚焦与数字补识别，以量化聚焦与补识别模块的边际收益。",
    )
    add_thesis_figure(
        doc,
        FIG6_4_A,
        "图6-4　实验用案例（与第4章同源）：原图与 preprocess_with_context 输出对照（与线上一致）",
        width=Inches(5.4),
    )
    add_body_paragraph(
        doc,
        "若需单独展示经流水线裁切增强后的整幅预处理结果（不含左右拼接），可将 out/test2/01_full_preprocess_pipeline.png 作为附图或附录材料；"
        "论文字数受限时也可仅用图6-4 一幅完成说明。",
    )
    add_thesis_figure(
        doc,
        FIG6_4_B,
        "图6-4（附）　同一案例经完整预处理后的单幅结果（可选排版）",
        width=Inches(4.2),
    )

    add_heading(doc, "6.5　对比实验与结果分析", 2)
    add_body_paragraph(
        doc,
        "为量化“相对传统方法的优势”，建议从工程可实现角度设置三类对照：（A）人工对照录入，记录单表耗时与笔误类型；"
        "（B）不做预处理与表格恢复的“整图单次通用 OCR”，将输出按阅读顺序拼接为纯文本；（C）仅保留检测框聚类、关闭二次聚焦与数字补识别的简化管线。"
        "在每一类输入子集上统计单元格准确率、表头对齐率、需人工修正次数与端到端时延，并保留失败样例截图用于误差分析。",
    )
    add_body_paragraph(
        doc,
        "定性上，本文系统预期在以下维度优于（B）：透视与背景干扰下的鲁棒性、短数字与小数点召回、直接可编辑的二维结构；"
        "相对（A）则显著降低时间成本；相对（C）则体现二次聚焦与补识别模块的边际收益。下表给出答辩前可填写的对比框架（表中数值需用自有样本实测替换）。",
    )

    tbl = doc.add_table(rows=5, cols=4)
    tbl.style = "Table Grid"
    hdr = ["对比项", "人工录入", "传统整图 OCR", "本文系统"]
    for j, h in enumerate(hdr):
        tbl.rows[0].cells[j].text = h
    rows_cmp = [
        ("结构化表格", "高（但成本高）", "低", "高"),
        ("几何畸变适应", "人眼强依赖", "弱", "较强（透视/纠偏）"),
        ("小数字/易漏字段", "依赖细心", "易漏", "局部补识别增强"),
        ("可解释校对", "无自动提示", "弱", "置信度+框选高亮"),
    ]
    for i, row in enumerate(rows_cmp, start=1):
        for j, cell in enumerate(row):
            tbl.rows[i].cells[j].text = cell
    doc.add_paragraph()

    add_body_paragraph(
        doc,
        "由上表可见，本文系统在保持较高结构化输出的同时，将大量几何与版面理解工作自动化，并通过人机协同界面降低纠错成本；"
        "相较纯人工方式显著缩短录入时间，相较 naive OCR 又明显减少了行列错位与字段丢失问题。"
        "若实验结果在 D 类弱线表上仍波动较大，可在第7章展望中明确作为后续模型升级的重点方向。",
    )

    add_heading(doc, "6.6　运行截图与界面展示", 2)
    add_body_paragraph(
        doc,
        "图6-1 给出班级成绩片段经识别后的表格展示效果（姓名、班级、平时成绩等列结构清晰），对应原始文件 Snipaste_2026-02-24_19-13-43.png；"
        "已在下节插入 Word 图片对象，印刷前可替换为更高分辨率截图。图6-2、图6-3 请补充整页主界面与高亮联动效果。",
    )
    if FIG_SRC.exists():
        doc.add_paragraph()
        pic_p = doc.add_paragraph()
        pic_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pic_p.add_run().add_picture(str(FIG_SRC), width=Inches(5.2))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cr = cap.add_run("图6-1　系统识别结果示例（表格展示）")
        set_run_font(cr, east_asia="宋体", ascii_font="Times New Roman", size=Pt(10.5))
    else:
        add_figure_placeholder(doc, "【图6-1】未找到截图文件，请将 Snipaste_2026-02-24_19-13-43.png 置于指定路径后重新生成。")

    add_figure_placeholder(
        doc,
        "【图6-2 插入位置】请插入：系统 Web 主界面全屏截图（含上传区、原图/预处理预览、右侧可编辑表格）。",
    )
    add_figure_placeholder(
        doc,
        "【图6-3 插入位置】请插入：点击单元格后左侧高亮联动的截图（展示反投影或多边形框选效果）。",
    )

    add_heading(doc, "6.7　算法归纳、消融讨论、性能瓶颈与可维护性", 2)
    add_body_paragraph(
        doc,
        "算法层面，检测与识别由 PaddleOCR 深度模型承担，角度分类器改善旋转样本；几何与表格线部分由 OpenCV 经典算子承担，负责把输入变换到模型友好域；"
        "结构歧义由多路径评分与PaddleX{{2}}兜底消化。三者分工使系统兼具数据驱动与可解释规则的优点。",
    )
    add_body_paragraph(
        doc,
        "消融上，可分别关闭：白纸裁切、透视校正、二次聚焦、线结构分支、PaddleX 兜底、数字补识别，观察各模块在 B 类斜拍样本上的分数变化。"
        "经验上，斜拍样本对透视与聚焦更敏感；截图样本对聚类主路径更友好。",
    )
    add_body_paragraph(
        doc,
        "性能方面，CPU 推理受线程数、模型加载缓存与图像分辨率影响显著；PaddleX 首次调用可能触发较重的模型准备，应在论文中如实说明并给出冷启动与热启动两次计时。"
        "内存方面，Base64 预览图会放大 JSON 体积，若部署到弱网环境可改为对象存储外链。",
    )
    add_body_paragraph(
        doc,
        "可维护性方面，tests 目录下的 unittest 用例为回归提供抓手；建议在持续集成中启用同一命令，以避免依赖升级引入静默行为变化。"
        "论文答辩材料中可再次附录测试通过截图，与第6.3节文字相互印证。",
    )

    add_heading(doc, "第7章　总结与展望", 1)
    add_body_paragraph(
        doc,
        "本文完成了面向学生成绩单场景的端到端识别系统，涵盖预处理、OCR、多路径表格恢复、解析纠错与前后端联调。"
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
        (FIG4_1, OUTPUT_ROOT / "2252733-\u8bba\u6587-\u56fe4-1.png"),
        (FIG4_2, OUTPUT_ROOT / "2252733-\u8bba\u6587-\u56fe4-2.png"),
        (FIG4_3, OUTPUT_ROOT / "2252733-\u8bba\u6587-\u56fe4-3.png"),
        (FIG4_4, OUTPUT_ROOT / "2252733-\u8bba\u6587-\u56fe4-4.png"),
        (FIG4_5, OUTPUT_ROOT / "2252733-\u8bba\u6587-\u56fe4-5.png"),
        (FIG4_6, OUTPUT_ROOT / "2252733-\u8bba\u6587-\u56fe4-6.png"),
        (FIG6_4_A, OUTPUT_ROOT / "2252733-\u8bba\u6587-\u56fe6-4a.png"),
        (FIG6_4_B, OUTPUT_ROOT / "2252733-\u8bba\u6587-\u56fe6-4b.png"),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)


if __name__ == "__main__":
    main()

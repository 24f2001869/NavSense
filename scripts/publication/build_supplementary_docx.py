"""
build_supplementary_docx.py
---------------------------
Builds a clean, formal, reader-facing Supplementary Material DOCX and PDF
for The Journal of Navigation (Cambridge University Press).

Applies:
- 12 pt Times New Roman
- 2.54 cm (1.0 inch) margins
- Single line spacing (1.0)
- Paragraph spacing before = 0, after = 0
- Left-aligned headings
- Native OMML equation rendering for all display ($$...$$) and inline ($...$) mathematics
- Clean booktabs styling for Tables S1, S2, S3 with repeating header rows (w:tblHeader)
- Exports to Supplementary_Material.docx and Supplementary_Material.pdf
"""

import os
import re
import csv
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
import latex2mathml.converter
from lxml import etree

XSL_PATH = r'C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL'
XSLT_TRANSFORM = etree.XSLT(etree.parse(XSL_PATH))

def latex_to_omml_element(latex_str):
    """Converts a LaTeX math string into a native Word OMML oxml element."""
    mml = latex2mathml.converter.convert(latex_str)
    mml_tree = etree.fromstring(mml)
    omml_tree = XSLT_TRANSFORM(mml_tree)
    omml_str = etree.tostring(omml_tree).decode('utf-8')
    return parse_xml(omml_str)

def set_cell_margins(cell, top=20, bottom=20, left=40, right=40):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_border(cell, **kwargs):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for edge in ('top', 'left', 'bottom', 'right'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = f'w:{edge}'
            element = OxmlElement(tag)
            element.set(qn('w:val'), edge_data.get('val', 'single'))
            element.set(qn('w:sz'), str(edge_data.get('sz', 4)))
            element.set(qn('w:space'), '0')
            element.set(qn('w:color'), edge_data.get('color', 'auto'))
            tcBorders.append(element)
    tcPr.append(tcBorders)

def add_runs_with_inline_math(paragraph, text, font_size=12, italic_default=False):
    tokens = re.split(r'(\$[^$\n]+\$)', text)
    for token in tokens:
        if not token:
            continue
        if token.startswith('$') and token.endswith('$') and len(token) > 2:
            raw_math = token[1:-1].strip()
            try:
                omml_el = latex_to_omml_element(raw_math)
                paragraph._p.append(omml_el)
            except Exception:
                clean = raw_math.replace('\\text{', '').replace('}', '')
                r = paragraph.add_run(clean)
                r.font.name = 'Times New Roman'
                r.font.size = Pt(font_size)
                r.font.italic = True
        else:
            sub_tokens = re.split(r'(\*\*.*?\*\*|\*.*?\*)', token)
            for st in sub_tokens:
                if not st:
                    continue
                if st.startswith('**') and st.endswith('**'):
                    r = paragraph.add_run(st[2:-2])
                    r.font.bold = True
                elif st.startswith('*') and st.endswith('*'):
                    r = paragraph.add_run(st[1:-1])
                    r.font.italic = True
                else:
                    r = paragraph.add_run(st)
                    if italic_default:
                        r.font.italic = True
                r.font.name = 'Times New Roman'
                r.font.size = Pt(font_size)
                r.font.color.rgb = RGBColor(0, 0, 0)

def build_supplementary():
    doc = Document()

    # Set page margins to 2.54 cm (1 inch)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.page_width = Inches(8.5)
        section.page_height = Inches(11.0)

    # Base style
    normal = doc.styles['Normal']
    normal.font.name = 'Times New Roman'
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.0
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Read markdown source
    md_path = 'publication/journal_of_navigation_submission/supplementary/Supplementary_Material.md'
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Horizontal rule
        if line == '---':
            i += 1
            continue

        # Title / Headings
        if line.startswith('# '):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.keep_with_next = True
            run = p.add_run(line[2:])
            run.font.name = 'Times New Roman'
            run.font.size = Pt(14)
            run.font.bold = False
            i += 1
            continue

        if line.startswith('## '):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.keep_with_next = True
            run = p.add_run(line[3:])
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12)
            run.font.bold = False
            i += 1
            continue

        if line.startswith('### '):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.keep_with_next = True
            run = p.add_run(line[4:])
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12)
            run.font.italic = True
            i += 1
            continue

        # Display equation ($$...$$)
        if line.startswith('$$') and line.endswith('$$') and len(line) > 4:
            raw_eq = line[2:-2].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.0
            try:
                omml_el = latex_to_omml_element(raw_eq)
                p._p.append(omml_el)
            except Exception as e:
                print(f"Warning: Supplementary display equation OMML failed: {e}")
                r = p.add_run(raw_eq)
                r.font.name = 'Times New Roman'
                r.font.size = Pt(12)
                r.font.italic = True
            i += 1
            continue

        # Table caption
        if line.startswith('Table S'):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.keep_with_next = True
            run = p.add_run(line)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(10.5)
            run.font.bold = False
            i += 1
            continue

        # Table block (Markdown table starting with |)
        if line.startswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1

            rows = []
            for tl in table_lines:
                cells = [c.strip() for c in tl.split('|')[1:-1]]
                if cells and all(set(c).issubset({'-', ':', ' '}) for c in cells):
                    continue # Skip separator line
                rows.append(cells)

            if rows:
                num_rows = len(rows)
                num_cols = max(len(r) for r in rows)
                table = doc.add_table(rows=num_rows, cols=num_cols)
                table.alignment = WD_TABLE_ALIGNMENT.CENTER

                top_b = {'sz': 12, 'val': 'single', 'color': '000000'}
                hdr_b = {'sz': 6, 'val': 'single', 'color': '000000'}
                bot_b = {'sz': 12, 'val': 'single', 'color': '000000'}
                none_b = {'val': 'none'}

                for r_idx, row in enumerate(rows):
                    trPr = table.rows[r_idx]._tr.get_or_add_trPr()
                    trPr.append(OxmlElement('w:cantSplit'))
                    if r_idx == 0:
                        trPr.append(OxmlElement('w:tblHeader'))

                    for c_idx, val in enumerate(row):
                        if c_idx >= num_cols:
                            continue
                        cell = table.cell(r_idx, c_idx)
                        cell.text = ""
                        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                        set_cell_margins(cell, top=15, bottom=15, left=30, right=30)

                        p_cell = cell.paragraphs[0]
                        p_cell.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
                        p_cell.paragraph_format.line_spacing = 1.0
                        p_cell.paragraph_format.space_before = Pt(0)
                        p_cell.paragraph_format.space_after = Pt(0)
                        add_runs_with_inline_math(p_cell, val, font_size=8.5)
                        if r_idx == 0 and len(p_cell.runs) > 0:
                            p_cell.runs[0].font.bold = True

                        if r_idx == 0:
                            set_cell_border(cell, top=top_b, bottom=hdr_b, left=none_b, right=none_b)
                        elif r_idx == num_rows - 1:
                            set_cell_border(cell, bottom=bot_b, top=none_b, left=none_b, right=none_b)
                        else:
                            set_cell_border(cell, top=none_b, bottom=none_b, left=none_b, right=none_b)
            continue

        # Regular paragraph
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.first_line_indent = Pt(0)
        add_runs_with_inline_math(p, line, font_size=12)
        i += 1

    # Save Word document
    out_docx = 'publication/journal_of_navigation_submission/supplementary/Supplementary_Material.docx'
    doc.save(out_docx)
    print(f"Created: {out_docx}")

    # Export to PDF via Word COM Automation
    try:
        import win32com.client
        import pythoncom
        pythoncom.CoInitialize()
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        abs_docx = os.path.abspath(out_docx)
        abs_pdf = os.path.abspath('publication/journal_of_navigation_submission/supplementary/Supplementary_Material.pdf')

        doc_obj = word.Documents.Open(abs_docx)
        doc_obj.ExportAsFixedFormat(abs_pdf, 17)
        pages = doc_obj.ComputeStatistics(2)
        words = doc_obj.ComputeStatistics(0)
        doc_obj.Close(False)
        word.Quit()
        print(f"Created: {abs_pdf} (Pages: {pages}, Words: {words})")
    except Exception as e:
        print(f"PDF export warning: {e}")

if __name__ == '__main__':
    build_supplementary()

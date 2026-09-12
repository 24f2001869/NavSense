"""
Script to build the publication-ready Word document (.docx) and PDF (.pdf)
for The Journal of Navigation (Cambridge University Press).

Cambridge Requirements Enforced:
- 12 pt Times New Roman
- Single line spacing = 1.0
- Paragraph spacing before = 0, after = 0
- 2.54 cm (1 inch) margins on all sides
- Single column
- Fully justified body text
- Left-aligned section and subsection headings
- No paragraph indents
- Unbolded headings per Cambridge JoN instructions
- Sequential equation numbering from (1) in a 2-cell borderless table structure
- Native editable Word equations (OMML) for display and inline mathematics
- Table captions above tables; repeating header rows (w:tblHeader) on continuation pages
- Figure captions beneath figures
- Complete Harvard author-date references (alphabetical, unnumbered, hanging indent)
- Target length <= 20 pages
"""

import os
import csv
import re
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls
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

def set_cell_border(cell, **kwargs):
    """Set cell borders: top, bottom, left, right."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.first_child_found_in("w:tcBorders")
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = f'w:{edge}'
            element = tcBorders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcBorders.append(element)
            for key, val in edge_data.items():
                element.set(qn(f'w:{key}'), str(val))

def set_cell_margins(cell, top=20, bottom=20, left=50, right=50):
    """Set cell margins in dxa (1 pt = 20 dxa)."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_runs_with_inline_math(paragraph, text, font_size=12, italic_default=False):
    """
    Parses a text string containing inline $math$ expressions and adds
    native text runs and inline OMML math elements to the paragraph.
    """
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
                # Fallback clean text if converter fails
                clean = raw_math.replace('\\text{', '').replace('}', '')
                r = paragraph.add_run(clean)
                r.font.name = 'Times New Roman'
                r.font.size = Pt(font_size)
                r.font.italic = True
        else:
            # Check for markdown bold or italic in regular text
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

def build_manuscript():
    doc = docx.Document()

    # 1. Page Setup: 1 inch (2.54 cm) margins, Portrait
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.page_width = Inches(8.5)
        section.page_height = Inches(11.0)

    # 2. Base Normal Style: 12pt Times New Roman, Single Spacing (1.0), 0 before, 0 after
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.0
    normal_style.paragraph_format.space_before = Pt(0)
    normal_style.paragraph_format.space_after = Pt(0)
    normal_style.paragraph_format.first_line_indent = Pt(0)
    normal_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # 3. Read Source Markdown Text
    with open('publication/journal_of_navigation_submission/01_JON_manuscript.md', 'r', encoding='utf-8') as f:
        content = f.read()

    # 4. Title block
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(0)
    p_title.paragraph_format.line_spacing = 1.0
    run_title = p_title.add_run("Failure-Aware Smartphone Inertial Dead Reckoning Under GNSS Outages: An Empirical Study of Learned Velocity, Sensor Fusion, Map Constraints, and Distribution Shift")
    run_title.font.name = 'Times New Roman'
    run_title.font.size = Pt(12)
    run_title.font.bold = False

    # Running title
    p_run_title = doc.add_paragraph()
    p_run_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_run_title.paragraph_format.space_before = Pt(0)
    p_run_title.paragraph_format.space_after = Pt(0)
    p_run_title.paragraph_format.line_spacing = 1.0
    run_rt = p_run_title.add_run("Short running title: Failure-Aware Smartphone Dead Reckoning")
    run_rt.font.name = 'Times New Roman'
    run_rt.font.size = Pt(12)
    run_rt.font.italic = True

    # Author
    p_author = doc.add_paragraph()
    p_author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_author.paragraph_format.space_before = Pt(0)
    p_author.paragraph_format.space_after = Pt(0)
    p_author.paragraph_format.line_spacing = 1.0
    run_author = p_author.add_run("Rahul Kumar")
    run_author.font.name = 'Times New Roman'
    run_author.font.size = Pt(12)
    run_author.font.bold = False

    # Affiliation
    p_affil = doc.add_paragraph()
    p_affil.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_affil.paragraph_format.space_before = Pt(0)
    p_affil.paragraph_format.space_after = Pt(0)
    p_affil.paragraph_format.line_spacing = 1.0
    run_affil = p_affil.add_run("Integrated M.Tech. (Materials Engineering), School of Engineering Sciences & Technology, University of Hyderabad, Hyderabad, India")
    run_affil.font.name = 'Times New Roman'
    run_affil.font.size = Pt(12)

    # Corresponding Author Email
    p_email = doc.add_paragraph()
    p_email.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_email.paragraph_format.space_before = Pt(0)
    p_email.paragraph_format.space_after = Pt(0)
    p_email.paragraph_format.line_spacing = 1.0
    run_email = p_email.add_run("Corresponding author email: 24f2001869@ds.study.iitm.ac.in")
    run_email.font.name = 'Times New Roman'
    run_email.font.size = Pt(12)

    # 5. Abstract Heading (Left-aligned)
    p_abs_h = doc.add_paragraph()
    p_abs_h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_abs_h.paragraph_format.space_before = Pt(0)
    p_abs_h.paragraph_format.space_after = Pt(0)
    p_abs_h.paragraph_format.line_spacing = 1.0
    run_abs_h = p_abs_h.add_run("Abstract")
    run_abs_h.font.name = 'Times New Roman'
    run_abs_h.font.size = Pt(12)
    run_abs_h.font.bold = False

    abs_text = (
        "During Global Navigation Satellite System (GNSS) outages, smartphone inertial dead reckoning "
        "suffers from rapid nonlinear open-loop position drift due to low-cost sensor errors. We present an "
        "empirical study evaluating learned forward velocity estimation, sensor fusion, kinematic "
        "constraints, map feedback, and distribution shifts using the public IO-VNBD benchmark dataset "
        "(64 passenger-car trips, 16.27 hours). On 19 trip-disjoint test routes, scaling a causal temporal "
        "convolutional network from 6 to 39 training trips reduces velocity error by 55.27% (6.17 to 2.76 m/s) "
        "and 60-second drift by 64.72% (263.6 to 93.0 m). However, unconditional lateral kinematic constraints "
        "degrade 60-second drift by 45.47%, and closed-loop map heading feedback degrades drift by 125.5%. "
        "Across 13 usable 60-second blackout trajectories, adaptive fusion achieves 96.65 m mean absolute drift "
        "and a macro-average normalized drift of 16.76% across the 12 dynamic routes (stationary control Vw15 "
        "excluded per Eq. 13), satisfying a sub-10% drift benchmark on 3 of 13 routes (23.08%). "
        "Physical road validation and multi-driver generalisation remain open challenges."
    )
    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_abs.paragraph_format.space_before = Pt(0)
    p_abs.paragraph_format.space_after = Pt(0)
    p_abs.paragraph_format.line_spacing = 1.0
    run_abs = p_abs.add_run(abs_text)
    run_abs.font.name = 'Times New Roman'
    run_abs.font.size = Pt(12)

    # 6. Parse Tables from CSV
    tables_data = {}
    for i in range(1, 8):
        csv_path = f"publication/journal_of_navigation_submission/tables/Table_{i:02d}.csv"
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as tf:
                reader = csv.reader(tf)
                tables_data[f"Table {i}"] = [row for row in reader]

    # Map of figures to insert
    figures_data = {
        1: ("publication/journal_of_navigation_submission/figures/Figure_01.png",
            "Figure 1. NavSense end-to-end modular navigation architecture showing raw sensor ingestion, temporal preprocessing, causal TCN-kin velocity estimation (4 causal residual blocks, 61-sample receptive field reach corresponding to 6.0 s causal look-back), 15-state Error-State Kalman Filter with decoupled velocity damping, and downstream open-loop map matching."),
        2: ("publication/journal_of_navigation_submission/figures/Figure_02.png",
            "Figure 2. Open-loop strapdown inertial dead reckoning divergence during a 60-second GNSS outage on test route Vta20, illustrating rapid nonlinear position error growth exceeding 300 metres in the absence of external velocity aiding. Panel (b) reports the route-specific drift trajectory for Vta20; multi-route mean values across all eligible held-out routes are reported separately in Table 3."),
        3: ("publication/journal_of_navigation_submission/figures/Figure_03.png",
            "Figure 3. Controlled scaling of causal TCN-kin training across 19 held-out test routes (115,420 prediction epochs, 240.6 km): comparison between 6-trip (1.1 h) and 39-trip (12.5 h) models, demonstrating a 55.27% reduction in velocity MAE and a 64.72% reduction in 60-second position drift."),
        4: ("publication/journal_of_navigation_submission/figures/Figure_04.png",
            "Figure 4. Kinematic constraint instability and decoupled velocity damping recovery: comparison of dead reckoning trajectories under standard coupled lateral NHC versus NavSense decoupled lateral velocity damping ($K_y[6:15] = 0$), alongside $NIS_x$ innovation surge."),
        5: ("publication/journal_of_navigation_submission/figures/Figure_05.png",
            "Figure 5. Breakdown of closed-loop road network map matching during a 60-second satellite blackout: candidate road link ambiguity and heading injection cause unrecoverable trajectory divergence (+125.47% degradation)."),
        6: ("publication/journal_of_navigation_submission/figures/Figure_06.png",
            "Figure 6. Statistical feature separability and regression bias during steady-state motorway cruising ($v \\ge 25\\text{ m/s}$): near-zero specific force leaves 80 km/h vs 110 km/h cruising weakly separable (ROC-AUC = 0.625), producing an average negative prediction bias of -3.5 m/s ($r = -0.4866$)."),
        7: ("publication/journal_of_navigation_submission/figures/Figure_07.png",
            "Figure 7. Spectral vibration speed audit across 64 passenger-car trips (>450,000 one-second windows): dominant inertial vibration frequency (~2.2–2.5 Hz) remains invariant to vehicle speed ($r = -0.032, \\rho = -0.028$), consistent with low-frequency vehicle-body/chassis dynamics. Panel (a) shows a motorway-specific spectral example with a peak near 2.9 Hz; the corpus-wide dominant-frequency census in panel (b) centres near 2.2–2.5 Hz across all 64 trips."),
        8: ("publication/journal_of_navigation_submission/figures/Figure_08.png",
            "Figure 8. Pedestrian out-of-distribution dynamic response (71.66 m/s spike induced by +26.08$\\sigma$ arm-swing yaw rates) alongside Android on-device execution benchmarks on Google Pixel 7a (mean latency 9.15 ms, 99th percentile 13.89 ms).")
    }

    # Split markdown by double newlines into blocks
    raw_blocks = content.split('\n\n')
    skip = True
    in_references = False
    equation_counter = 0
    pending_declaration = None

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        
        # Start processing at "1. Introduction"
        if block.startswith("1. Introduction"):
            skip = False
        
        if skip:
            continue

        # Check for References section
        if block == "References":
            in_references = True
            p_ref_h = doc.add_paragraph()
            p_ref_h.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p_ref_h.paragraph_format.space_before = Pt(0)
            p_ref_h.paragraph_format.space_after = Pt(0)
            p_ref_h.paragraph_format.line_spacing = 1.0
            p_ref_h.paragraph_format.keep_with_next = True
            run_ref_h = p_ref_h.add_run("References")
            run_ref_h.font.name = 'Times New Roman'
            run_ref_h.font.size = Pt(12)
            run_ref_h.font.bold = False
            continue

        if in_references:
            # Add reference entry without indents per Cambridge JoN instructions ("Indents should NOT be used")
            p_ref = doc.add_paragraph()
            p_ref.paragraph_format.left_indent = Pt(0)
            p_ref.paragraph_format.first_line_indent = Pt(0)
            p_ref.paragraph_format.space_before = Pt(0)
            p_ref.paragraph_format.space_after = Pt(0)
            p_ref.paragraph_format.line_spacing = 1.0
            p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            run = p_ref.add_run(block.replace('\n', ' '))
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12)
            continue

        # Check for Declaration Headings (formatted as run-in paragraphs per Cambridge JoN style)
        declaration_headings = [
            "Data Availability Statement", "Code Availability Statement",
            "Acknowledgements", "Funding", "Competing Interests", "AI Disclosure Statement"
        ]
        if block in declaration_headings:
            pending_declaration = block
            continue

        if pending_declaration:
            p_decl = doc.add_paragraph()
            p_decl.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p_decl.paragraph_format.space_before = Pt(0)
            p_decl.paragraph_format.space_after = Pt(0)
            p_decl.paragraph_format.line_spacing = 1.0
            p_decl.paragraph_format.first_line_indent = Pt(0)
            r_dh = p_decl.add_run(f"{pending_declaration}. ")
            r_dh.font.name = 'Times New Roman'
            r_dh.font.size = Pt(12)
            r_dh.font.bold = False
            r_dh.font.italic = True
            add_runs_with_inline_math(p_decl, block, font_size=12)
            pending_declaration = None
            continue

        # Check for Section Headings
        main_sections = [
            "1. Introduction", "2. Dataset and Methods", "3. Experimental Protocol",
            "4. Results", "5. Discussion", "6. Limitations", "7. Conclusions"
        ]
        if block in main_sections:
            p_head = doc.add_paragraph()
            p_head.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p_head.paragraph_format.space_before = Pt(0)
            p_head.paragraph_format.space_after = Pt(0)
            p_head.paragraph_format.line_spacing = 1.0
            p_head.paragraph_format.keep_with_next = True
            run_h = p_head.add_run(block)
            run_h.font.name = 'Times New Roman'
            run_h.font.size = Pt(12)
            run_h.font.bold = False
            continue

        # Check for Subsection Headings (e.g. 2.1. Dataset and reference signals)
        if re.match(r'^[1-7]\.[0-9]+\.\s+[A-Za-z]', block) and '\n' not in block and len(block) < 100:
            p_subhead = doc.add_paragraph()
            p_subhead.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p_subhead.paragraph_format.space_before = Pt(0)
            p_subhead.paragraph_format.space_after = Pt(0)
            p_subhead.paragraph_format.line_spacing = 1.0
            p_subhead.paragraph_format.keep_with_next = True
            run_sub = p_subhead.add_run(block)
            run_sub.font.name = 'Times New Roman'
            run_sub.font.size = Pt(12)
            run_sub.font.italic = True
            continue

        m_sub = re.match(r'^([1-7]\.[0-9]+\.\s+[^\n]+)\n(.*)$', block, re.DOTALL)
        if m_sub and len(m_sub.group(1)) < 100:
            sub_title = m_sub.group(1).strip()
            rest_body = m_sub.group(2).strip()
            p_subhead = doc.add_paragraph()
            p_subhead.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p_subhead.paragraph_format.space_before = Pt(0)
            p_subhead.paragraph_format.space_after = Pt(0)
            p_subhead.paragraph_format.line_spacing = 1.0
            p_subhead.paragraph_format.keep_with_next = True
            run_sub = p_subhead.add_run(sub_title)
            run_sub.font.name = 'Times New Roman'
            run_sub.font.size = Pt(12)
            run_sub.font.italic = True

            if rest_body:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.first_line_indent = Pt(0)
                add_runs_with_inline_math(p, rest_body, font_size=12)
            continue

        # Check for Display Equation block ($$...$$)
        if block.startswith("$$") and block.endswith("$$"):
            raw_eq = block[2:-2].strip()
            equation_counter += 1
            eq_num_str = f"({equation_counter})"

            # Render in a borderless 2-column table: Left cell centered equation, Right cell right-aligned number
            eq_table = doc.add_table(rows=1, cols=2)
            eq_table.alignment = WD_TABLE_ALIGNMENT.CENTER
            
            # Prevent row split
            trPr = eq_table.rows[0]._tr.get_or_add_trPr()
            trPr.append(OxmlElement('w:cantSplit'))

            # Set widths: Left cell 5.7 inches, Right cell 0.8 inches (total 6.5 inches)
            cell_eq = eq_table.cell(0, 0)
            cell_num = eq_table.cell(0, 1)
            cell_eq.width = Inches(5.7)
            cell_num.width = Inches(0.8)

            set_cell_margins(cell_eq, top=0, bottom=0, left=0, right=0)
            set_cell_margins(cell_num, top=0, bottom=0, left=0, right=0)
            none_border = {'val': 'none'}
            set_cell_border(cell_eq, top=none_border, bottom=none_border, left=none_border, right=none_border)
            set_cell_border(cell_num, top=none_border, bottom=none_border, left=none_border, right=none_border)

            # Left cell: equation centered
            p_eq = cell_eq.paragraphs[0]
            p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_eq.paragraph_format.line_spacing = 1.0
            p_eq.paragraph_format.space_before = Pt(0)
            p_eq.paragraph_format.space_after = Pt(0)
            try:
                omml_eq = latex_to_omml_element(raw_eq)
                p_eq._p.append(omml_eq)
            except Exception as e:
                print(f"Warning: Display equation {equation_counter} OMML failed: {e}")
                r_fallback = p_eq.add_run(raw_eq)
                r_fallback.font.name = 'Times New Roman'
                r_fallback.font.size = Pt(12)
                r_fallback.font.italic = True

            # Right cell: number right-aligned
            p_num = cell_num.paragraphs[0]
            p_num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            p_num.paragraph_format.line_spacing = 1.0
            p_num.paragraph_format.space_before = Pt(0)
            p_num.paragraph_format.space_after = Pt(0)
            r_num = p_num.add_run(eq_num_str)
            r_num.font.name = 'Times New Roman'
            r_num.font.size = Pt(12)
            continue

        # Check for Figure insertion placeholders
        fig_match = re.search(r'\[Figure\s+([1-8])\s+about here', block, re.IGNORECASE)
        if fig_match:
            fig_num = int(fig_match.group(1))
            fig_path, fig_cap = figures_data[fig_num]
            if os.path.exists(fig_path):
                p_fig = doc.add_paragraph()
                p_fig.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_fig.paragraph_format.space_before = Pt(0)
                p_fig.paragraph_format.space_after = Pt(0)
                p_fig.paragraph_format.line_spacing = 1.0
                p_fig.paragraph_format.keep_with_next = True
                p_fig.add_run().add_picture(fig_path, width=Inches(2.25))

                # Figure caption beneath figure
                p_cap = doc.add_paragraph()
                p_cap.paragraph_format.space_before = Pt(0)
                p_cap.paragraph_format.space_after = Pt(0)
                p_cap.paragraph_format.line_spacing = 1.0
                p_cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                add_runs_with_inline_math(p_cap, fig_cap, font_size=10, italic_default=True)
            continue

        # Check for Table markdown blocks (e.g. Table 1. ...)
        tbl_match = re.match(r'^(Table\s+([1-7]))\.\s+(.*)', block)
        if tbl_match:
            tbl_key = tbl_match.group(1)
            tbl_title = block.split('\n')[0] # First line is table title
            
            # Table caption ABOVE table (Left-aligned)
            p_tcap = doc.add_paragraph()
            p_tcap.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p_tcap.paragraph_format.space_before = Pt(0)
            p_tcap.paragraph_format.space_after = Pt(0)
            p_tcap.paragraph_format.line_spacing = 1.0
            p_tcap.paragraph_format.keep_with_next = True
            run_tc = p_tcap.add_run(tbl_title)
            run_tc.font.name = 'Times New Roman'
            run_tc.font.size = Pt(11)
            run_tc.font.bold = False

            # Render Table from CSV data
            if tbl_key in tables_data:
                rows = tables_data[tbl_key]
                if rows:
                    num_rows = len(rows)
                    num_cols = len(rows[0])
                    table = doc.add_table(rows=num_rows, cols=num_cols)
                    table.alignment = WD_TABLE_ALIGNMENT.CENTER

                    top_border = {'sz': 12, 'val': 'single', 'color': '000000'}
                    header_bottom = {'sz': 6, 'val': 'single', 'color': '000000'}
                    bottom_border = {'sz': 12, 'val': 'single', 'color': '000000'}
                    none_border = {'val': 'none'}

                    for r_idx, row in enumerate(rows):
                        # Row properties: cantSplit on all rows, tblHeader on row 0
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
                            set_cell_margins(cell, top=20, bottom=20, left=50, right=50)
                            
                            # Format text in cell
                            p_cell = cell.paragraphs[0]
                            p_cell.alignment = WD_ALIGN_PARAGRAPH.CENTER if (c_idx > 0 or r_idx == 0) else WD_ALIGN_PARAGRAPH.LEFT
                            p_cell.paragraph_format.line_spacing = 1.0
                            p_cell.paragraph_format.space_after = Pt(0)
                            p_cell.paragraph_format.space_before = Pt(0)
                            add_runs_with_inline_math(p_cell, val, font_size=8.5)
                            if r_idx == 0 and len(p_cell.runs) > 0:
                                p_cell.runs[0].font.bold = True

                            # Apply booktabs style borders
                            if r_idx == 0:
                                set_cell_border(cell, top=top_border, bottom=header_bottom, left=none_border, right=none_border)
                            elif r_idx == num_rows - 1:
                                set_cell_border(cell, bottom=bottom_border, top=none_border, left=none_border, right=none_border)
                            else:
                                set_cell_border(cell, top=none_border, bottom=none_border, left=none_border, right=none_border)
            continue

        # Regular body paragraph: parse inline math $...$
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.first_line_indent = Pt(0)
        add_runs_with_inline_math(p, block, font_size=12)

    # 7. Save Word document
    out_docx = 'publication/journal_of_navigation_submission/01_JON_manuscript.docx'
    doc.save(out_docx)
    print(f"Successfully created: {out_docx}")
    print(f"Total sequentially numbered display equations: {equation_counter}")

    # 8. Export to PDF via Word COM Automation
    try:
        import win32com.client
        import pythoncom
        pythoncom.CoInitialize()
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        abs_docx = os.path.abspath(out_docx)
        abs_pdf = os.path.abspath('publication/journal_of_navigation_submission/02_JON_manuscript.pdf')
        
        doc_obj = word.Documents.Open(abs_docx)
        doc_obj.ExportAsFixedFormat(abs_pdf, 17) # 17 represents wdExportFormatPDF
        pages = doc_obj.ComputeStatistics(2)
        words = doc_obj.ComputeStatistics(0)
        doc_obj.Close(False)
        word.Quit()
        print(f"Successfully generated: {abs_pdf}")
        print(f"Official Word COM Page Count: {pages} pages (Budget: <= 20 pages)")
        print(f"Official Word COM Word Count: {words} words")
    except Exception as e:
        print(f"PDF export warning: {e}")

if __name__ == '__main__':
    build_manuscript()

"""Publication-Quality PDF Generator for Research Preprints.

Converts Markdown preprints into clean, professional academic PDFs with:
  - Standard ASCII math notation (guaranteed zero missing glyphs / zero black squares)
  - Styled multi-column tables with borders and headers
  - Proper paragraph spacing, section headers, and metadata rules
"""

import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def clean_markdown_text(text: str) -> str:
    """Replace ALL non-Latin1 / non-Helvetica unicode characters with clean ASCII equivalents.

    Prevents ReportLab black squares (■) caused by missing font glyphs.
    """
    replacements = {
        '⟨': '&lt;',
        '⟩': '&gt;',
        '₁': '_1',
        '₂': '_2',
        '₃': '_3',
        '₄': '_4',
        '₀': '_0',
        '†': '^dagger',
        'λ': 'lambda',
        'ρ': 'rho',
        'ε': 'epsilon',
        'σ': 'sigma',
        'θ': 'theta',
        'ψ': 'psi',
        'h̄': 'h_bar',
        '∏': 'PROD',
        '∑': 'SUM',
        '¹': '^1',
        '²': '^2',
        '³': '^3',
        '≤': '&lt;=',
        '≥': '&gt;=',
        '≠': '!=',
        '≈': '~=',
        '→': '-&gt;',
        'ℝ': 'R',
        'µ': 'u',
        '■': '',
        '`': '',
        '**': '',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Strip any remaining non-ASCII character that could break Helvetica
    text = ''.join(c if ord(c) < 128 else '' for c in text)
    return text


def build_academic_pdf(md_path: str, pdf_path: str):
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1a202c'),
        spaceAfter=6
    )
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor('#1a365d'),
        spaceBefore=14,
        spaceAfter=6
    )
    h3_style = ParagraphStyle(
        'DocH3',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#2c5282'),
        spaceBefore=10,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=6
    )
    code_style = ParagraphStyle(
        'DocCode',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1a202c'),
        backColor=colors.HexColor('#edf2f7'),
        borderColor=colors.HexColor('#cbd5e0'),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=6
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1
    )
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#2d3748'),
        alignment=1
    )

    story = []
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_code_block = False
    code_buffer = []

    in_table = False
    table_rows = []

    i = 0
    while i < len(lines):
        line_str = lines[i].rstrip()

        # Code block handling
        if line_str.startswith('```'):
            if in_code_block:
                code_text = "<br/>".join(code_buffer)
                story.append(Paragraph(code_text, code_style))
                code_buffer = []
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            safe_line = clean_markdown_text(line_str).replace(' ', '&nbsp;')
            code_buffer.append(safe_line)
            i += 1
            continue

        # Table handling
        if '|' in line_str and line_str.strip().startswith('|'):
            if ':---' in line_str or '---' in line_str:
                i += 1
                continue

            cells = [c.strip() for c in line_str.split('|')[1:-1]]
            if cells:
                table_rows.append(cells)
            in_table = True
            i += 1
            continue
        elif in_table:
            if table_rows:
                formatted_table_data = []
                for row_idx, row in enumerate(table_rows):
                    row_cells = []
                    style = table_header_style if row_idx == 0 else table_cell_style
                    for cell in row:
                        cell_clean = clean_markdown_text(cell)
                        row_cells.append(Paragraph(cell_clean, style))
                    formatted_table_data.append(row_cells)

                t = Table(formatted_table_data, hAlign='LEFT')
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')]),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(t)
                story.append(Spacer(1, 6))

            table_rows = []
            in_table = False

        # Regular Markdown handling
        if line_str.startswith('# '):
            story.append(Paragraph(clean_markdown_text(line_str[2:]), title_style))
            story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2b6cb0'), spaceAfter=8))
        elif line_str.startswith('## '):
            story.append(Paragraph(clean_markdown_text(line_str[3:]), h2_style))
        elif line_str.startswith('### '):
            story.append(Paragraph(clean_markdown_text(line_str[4:]), h3_style))
        elif line_str.startswith('---'):
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e2e8f0'), spaceBefore=4, spaceAfter=6))
        elif line_str.strip() == '':
            story.append(Spacer(1, 3))
        else:
            safe_text = clean_markdown_text(line_str)
            story.append(Paragraph(safe_text, body_style))

        i += 1

    doc.build(story)
    print(f"✅ Generated 100% clean PDF (zero black boxes): {pdf_path}")


if __name__ == "__main__":
    build_academic_pdf("docs/PREPRINT_DRAFT.md", "docs/PREPRINT_DRAFT.pdf")

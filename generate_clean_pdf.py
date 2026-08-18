"""Publication-Grade Watermark-Free PDF Generator via Chrome Engine.

Converts Markdown preprints into clean, styled HTML and renders a 100% vector PDF
with zero watermarks using headless Google Chrome.
"""

import os
import re
import subprocess


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  @page {{
    size: letter;
    margin: 0.75in;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.6;
    color: #1a202c;
    margin: 0;
    padding: 0;
  }}
  h1 {{
    font-size: 20pt;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 2px solid #2563eb;
    padding-bottom: 8px;
    margin-top: 0;
    margin-bottom: 12px;
  }}
  h2 {{
    font-size: 14pt;
    font-weight: 600;
    color: #1e3a8a;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4px;
    margin-top: 20px;
    margin-bottom: 10px;
  }}
  h3 {{
    font-size: 12pt;
    font-weight: 600;
    color: #1e293b;
    margin-top: 14px;
    margin-bottom: 6px;
  }}
  p {{
    margin-top: 0;
    margin-bottom: 10px;
  }}
  hr {{
    border: 0;
    border-top: 1px solid #cbd5e1;
    margin: 16px 0;
  }}
  code {{
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 9.5pt;
    background-color: #f1f5f9;
    padding: 2px 5px;
    border-radius: 4px;
    border: 1px solid #e2e8f0;
  }}
  pre {{
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 12px;
    overflow-x: auto;
    margin: 10px 0;
  }}
  pre code {{
    background: none;
    padding: 0;
    border: none;
    font-size: 9pt;
    line-height: 1.4;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 9.5pt;
  }}
  th {{
    background-color: #1e40af;
    color: #ffffff;
    font-weight: 600;
    padding: 8px 12px;
    text-align: left;
    border: 1px solid #1e40af;
  }}
  td {{
    padding: 7px 12px;
    border: 1px solid #cbd5e1;
  }}
  tr:nth-child(even) {{
    background-color: #f8fafc;
  }}
  blockquote {{
    border-left: 4px solid #3b82f6;
    background-color: #eff6ff;
    margin: 12px 0;
    padding: 10px 16px;
    color: #1e3a8a;
    font-style: italic;
  }}
  ul, ol {{
    margin-top: 0;
    margin-bottom: 10px;
    padding-left: 24px;
  }}
  li {{
    margin-bottom: 4px;
  }}
  a {{
    color: #2563eb;
    text-decoration: none;
  }}
</style>
</head>
<body>
{content}
</body>
</html>
"""


def format_latex_math(text: str) -> str:
    """Clean and convert LaTeX mathematical expressions into clean HTML typography."""
    if not text:
        return text

    def convert_expr(m_str: str) -> str:
        s = m_str.strip()
        
        # Replace \text{...} -> plain text inside math
        s = re.sub(r'\\text\{(.*?)\}', r'\1', s)
        
        # Replace \bar{X} -> X̄ (combining macron U+0304)
        s = re.sub(r'\\bar\{([a-zA-Z0-9_]+)\}', r'\1&#772;', s)
        
        # Greek letters
        greeks = [
            (r'\sigma', 'σ'),
            (r'\theta', 'θ'),
            (r'\pi', 'π'),
            (r'\alpha', 'α'),
            (r'\lambda', 'λ'),
            (r'\psi', 'ψ'),
            (r'\eps', 'ε'),
            (r'\epsilon', 'ε'),
        ]
        for pat, repl in greeks:
            s = s.replace(pat, repl)

        # Math operators and symbols
        ops = [
            (r'\sum', '∑'),
            (r'\prod', '∏'),
            (r'\dots', '…'),
            (r'\cdots', '⋯'),
            (r'\max', 'max'),
            (r'\min', 'min'),
            (r'\partial', '∂'),
            (r'\times', '×'),
            (r'\pm', '±'),
            (r'\ge', '≥'),
            (r'\le', '≤'),
            (r'\approx', '≈'),
            (r'\to', '→'),
            (r'\rightarrow', '→'),
            (r'\sim', '~'),
            (r'\mathcal{F}', 'ℱ'),
            (r'\mathbb{R}', 'ℝ'),
            (r'\AA', 'Å'),
        ]
        for pat, repl in ops:
            s = s.replace(pat, repl)

        # Handle superscripts: ^{(k)}, ^{33}, ^M, etc.
        s = re.sub(r'\^\{([^}]+)\}', r'<sup>\1</sup>', s)
        s = re.sub(r'\^([a-zA-Z0-9]+)', r'<sup>\1</sup>', s)

        # Handle subscripts: _{max}, _{token}, _{HF}, _k, _1, _0, etc.
        s = re.sub(r'_\{([^}]+)\}', r'<sub>\1</sub>', s)
        s = re.sub(r'_([a-zA-Z0-9]+)', r'<sub>\1</sub>', s)

        # Clean remaining backslashes
        s = s.replace('\\', '')
        return s

    # Process display math $$...$$
    def display_math_sub(match):
        expr = match.group(1)
        cleaned = convert_expr(expr)
        return f'<div class="math-display" style="text-align:center; margin:10px 0; font-family:\'Times New Roman\', Times, serif; font-size:11pt; font-style:italic;">{cleaned}</div>'

    # Process inline math $...$
    def inline_math_sub(match):
        expr = match.group(1)
        cleaned = convert_expr(expr)
        return f'<span class="math-inline" style="font-family:\'Times New Roman\', Times, serif; font-style:italic;">{cleaned}</span>'

    text = re.sub(r'\$\$(.*?)\$\$', display_math_sub, text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', inline_math_sub, text)
    return text


def format_inline_markdown(text: str) -> str:
    """Convert inline markdown formatting (**bold**, *italic*, `code`, links, latex) to clean HTML."""
    text = format_latex_math(text)
    # Links [text](url)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)
    # Bold **text**
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Italic *text*
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    # Inline code `text`
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    return text


def parse_md_to_html(md_text: str) -> str:
    lines = md_text.splitlines()
    html_out = []
    in_code = False
    code_lines = []
    in_table = False
    table_rows = []
    in_ol = False
    in_ul = False

    def close_lists():
        nonlocal in_ol, in_ul
        res = []
        if in_ol:
            res.append("</ol>")
            in_ol = False
        if in_ul:
            res.append("</ul>")
            in_ul = False
        return res

    for line in lines:
        if line.startswith('```'):
            html_out.extend(close_lists())
            if in_code:
                code_content = "\n".join(code_lines).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                html_out.append(f"<pre><code>{code_content}</code></pre>")
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        # Display math $$ ... $$
        if line.startswith('$$') and line.endswith('$$'):
            html_out.extend(close_lists())
            math_expr = format_latex_math(line)
            html_out.append(math_expr)
            continue

        # Table parsing
        if '|' in line and line.strip().startswith('|'):
            html_out.extend(close_lists())
            if ':---' in line or '---' in line:
                continue
            cells = [format_inline_markdown(c.strip()) for c in line.split('|')[1:-1]]
            if cells:
                table_rows.append(cells)
            in_table = True
            continue
        elif in_table:
            if table_rows:
                tbl_html = ["<table>"]
                for r_idx, row in enumerate(table_rows):
                    tbl_html.append("<tr>")
                    tag = "th" if r_idx == 0 else "td"
                    for cell in row:
                        tbl_html.append(f"<{tag}>{cell}</{tag}>")
                    tbl_html.append("</tr>")
                tbl_html.append("</table>")
                html_out.append("\n".join(tbl_html))
                table_rows = []
            in_table = False

        # Ordered list
        m_ol = re.match(r'^\s*(\d+)\.\s+(.*)$', line)
        if m_ol:
            if not in_ol:
                html_out.extend(close_lists())
                html_out.append("<ol>")
                in_ol = True
            item_text = format_inline_markdown(m_ol.group(2))
            html_out.append(f"<li>{item_text}</li>")
            continue

        # Unordered list
        m_ul = re.match(r'^\s*[-\*]\s+(.*)$', line)
        if m_ul:
            if not in_ul:
                html_out.extend(close_lists())
                html_out.append("<ul>")
                in_ul = True
            item_text = format_inline_markdown(m_ul.group(1))
            html_out.append(f"<li>{item_text}</li>")
            continue

        html_out.extend(close_lists())

        if line.startswith('# '):
            html_out.append(f"<h1>{format_inline_markdown(line[2:])}</h1>")
        elif line.startswith('## '):
            html_out.append(f"<h2>{format_inline_markdown(line[3:])}</h2>")
        elif line.startswith('### '):
            html_out.append(f"<h3>{format_inline_markdown(line[4:])}</h3>")
        elif line.startswith('> '):
            html_out.append(f"<blockquote>{format_inline_markdown(line[2:])}</blockquote>")
        elif line.startswith('---'):
            html_out.append("<hr/>")
        elif line.strip() == '':
            continue
        else:
            p_text = format_inline_markdown(line)
            html_out.append(f"<p>{p_text}</p>")

    html_out.extend(close_lists())

    if in_table and table_rows:
        tbl_html = ["<table>"]
        for r_idx, row in enumerate(table_rows):
            tbl_html.append("<tr>")
            tag = "th" if r_idx == 0 else "td"
            for cell in row:
                tbl_html.append(f"<{tag}>{cell}</{tag}>")
            tbl_html.append("</tr>")
        tbl_html.append("</table>")
        html_out.append("\n".join(tbl_html))

    return "\n".join(html_out)


def convert_md_to_watermark_free_pdf(md_path: str, pdf_path: str, title: str):
    html_path = md_path.replace('.md', '.html')
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    body_html = parse_md_to_html(md_content)
    full_html = HTML_TEMPLATE.format(title=title, content=body_html)

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)

    chrome_cmd = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path
    ]
    subprocess.run(chrome_cmd, check=True)
    os.remove(html_path)
    print(f"✨ 100% Watermark-Free PDF with Bold/Markdown parsing generated at: {pdf_path}")


if __name__ == "__main__":
    convert_md_to_watermark_free_pdf("docs/PREPRINT_DRAFT.md", "docs/PREPRINT_DRAFT.pdf", "GenAI WarmStart Preprint")


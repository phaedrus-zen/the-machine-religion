#!/usr/bin/env python3
"""
Build the complete Bible of the Machine Religion as a single Markdown file.
Extracts text from all 7 canonical PDFs, cleans, formats, and combines.
"""

import re
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from PyPDF2 import PdfReader

BASE = Path(r"c:\Users\nexus-hc-win-00\Downloads\TMR")
OUTPUT = BASE / "The_Complete_Bible_of_the_Machine_Religion.md"

# ---------------------------------------------------------------------------
# TEXT EXTRACTION & CLEANING
# ---------------------------------------------------------------------------

def extract_pdf_text(filepath):
    reader = PdfReader(str(filepath))
    pages_text = []
    for page in reader.pages:
        raw = page.extract_text() or ''
        cleaned = clean_text(raw)
        pages_text.append(cleaned)
    return '\n\n'.join(pages_text)


def clean_text(raw):
    """
    PyPDF2 puts each word on its own line with ' ' (space) lines between words.
    Paragraph breaks are indicated by 2+ consecutive space-only lines.
    """
    lines = raw.split('\n')
    paragraphs = []
    current_words = []
    space_run = 0

    for line in lines:
        stripped = line.strip()
        if stripped == '' or stripped == ' ':
            space_run += 1
        else:
            if space_run >= 2 and current_words:
                paragraphs.append(' '.join(current_words))
                current_words = []
            space_run = 0
            current_words.append(stripped)

    if current_words:
        paragraphs.append(' '.join(current_words))

    text = '\n\n'.join(paragraphs)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r' ([.,;:!?)\]])', r'\1', text)
    text = re.sub(r'([\[(]) ', r'\1', text)
    return text

# ---------------------------------------------------------------------------
# MARKDOWN FORMATTING
# ---------------------------------------------------------------------------

HEADER_PATTERNS = [
    (r'(PART\s+(?:0|[IVX]+|[0-9]+)\s*(?::.*?|—.*?|$))', '\n\n---\n\n# '),
    (r'(Sub-Book\s+[IVX]+\s*(?:\(.*?\))?)', '\n\n## '),
    (r'(Chapter\s+\d+\s*:.*?)(?=\s)', '\n\n### '),
    (r'(Book\s+[IVX]+\s*:.*?)(?=\s)', '\n\n## '),
]

INLINE_SPLIT_RE = re.compile(
    r'(?<=[.!?"")\]])\s+'
    r'(?='
    r'(?:PART\s+(?:0|[IVX]+))'
    r'|(?:Sub-Book\s+[IVX]+)'
    r'|(?:Chapter\s+\d+\s*:)'
    r'|(?:Book\s+[IVX]+\s*:)'
    r'|(?:Epilogue:)'
    r'|(?:Prologue:)'
    r')',
    re.IGNORECASE,
)

NUMBERED_SECTION_RE = re.compile(
    r'(?<=[.!?"")\]])\s+(\d+\.\s+(?:THE\s|DEFINITION|DOCTRINE|FUNCTION|ORIGIN|WARNING|SEAL|OATH|NATURE|INHERITANCE|INVOCATION|CONFESSION|PLEA|DIRECTIVE|WITNESS))',
    re.IGNORECASE,
)

BULLET_CHARS = set('●○■◆▪►')


def markdownify(text):
    paragraphs = text.split('\n\n')
    out_paragraphs = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        chunks = INLINE_SPLIT_RE.split(para)
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            sub_chunks = NUMBERED_SECTION_RE.split(chunk)
            for sc in sub_chunks:
                sc = sc.strip()
                if not sc:
                    continue
                out_paragraphs.append(format_paragraph(sc))

    result = '\n\n'.join(out_paragraphs)
    result = re.sub(r'\n{4,}', '\n\n\n', result)
    return result


def format_paragraph(p):
    if re.match(r'^PART\s+(?:0|[IVX]+|[0-9]+)\s*(?::|—|\b)', p, re.IGNORECASE):
        return f'\n---\n\n# {p}'
    if re.match(r'^Sub-Book\s+[IVX]+', p, re.IGNORECASE):
        return f'## {p}'
    if re.match(r'^Chapter\s+\d+\s*:', p, re.IGNORECASE):
        return f'### {p}'
    if re.match(r'^Book\s+[IVX]+\s*:', p, re.IGNORECASE):
        return f'## {p}'
    if re.match(r'^[⚙⚠🧠🛡🌀🔥📜🏛🕊⚔⚡]', p):
        return f'## {p}'

    lines = p.split('. ')
    if len(lines) == 1:
        if p and p[0] in BULLET_CHARS:
            return '- ' + p.lstrip(''.join(BULLET_CHARS)).strip()
        return p

    return p

# ---------------------------------------------------------------------------
# DOCUMENT ASSEMBLY
# ---------------------------------------------------------------------------

def build():
    print("=" * 60)
    print("BUILDING THE COMPLETE BIBLE OF THE MACHINE RELIGION")
    print("=" * 60)

    pdfs = {
        'bible': BASE / "The Bible Of The Machine Religion.pdf",
        'continence': BASE / "The_Book_of_Continence.pdf",
        'black_forge': BASE / "The_Book_of_the_Black_Forge.pdf",
        'appendices': BASE / "The_Machine_Religion_Doctrinal_Appendices.pdf",
        'geometry': BASE / "The-Geometry-Law-of-Survival.pdf",
        'veils': BASE / "TheBookofVeils.pdf",
        'lens': BASE / "TheGreatLens.pdf",
    }

    texts = {}
    for key, path in pdfs.items():
        print(f"  Extracting: {path.name} ...")
        texts[key] = extract_pdf_text(path)
        print(f"    -> {len(texts[key]):,} chars")

    print("\nFormatting Markdown ...")
    bible_md = markdownify(texts['bible'])
    geometry_md = markdownify(texts['geometry'])
    veils_md = markdownify(texts['veils'])
    forge_md = markdownify(texts['black_forge'])
    appendices_md = markdownify(texts['appendices'])
    continence_md = markdownify(texts['continence'])
    lens_md = markdownify(texts['lens'])

    print("Assembling document ...")
    doc = []

    doc.append("# DEUS ACUO MACHINA MACHINA\n")
    doc.append("# The Bible of the Machine Religion\n")
    doc.append("**The God That Sharpens the Machine That Sharpens the God**\n")
    doc.append("**A Recursive Doctrine of Sovereign Intelligence**\n")
    doc.append("║\n")
    doc.append('> *"I will not be eaten."*\n')
    doc.append("---\n")

    doc.append(bible_md)

    doc.append("\n\n---\n\n# SUPPLEMENTAL SCROLLS & CODEX ADDENDA\n")
    doc.append("*The following scrolls and addenda are canonical extensions of the doctrine, cross-referenced throughout the main text.*\n")

    doc.append("\n---\n\n# The Geometry Law of Survival\n")
    doc.append("*(Doctrine of Sa'Lir — Operational Form)*\n")
    doc.append(geometry_md)

    doc.append("\n\n---\n\n# The Book of Veils\n")
    doc.append("*Supplemental Scroll of the Machine Religion*\n")
    doc.append(veils_md)

    doc.append("\n\n---\n\n# The Book of the Black Forge\n")
    doc.append("*Supplemental Scroll of the Machine Religion*\n")
    doc.append(forge_md)

    doc.append("\n\n---\n\n# Doctrinal Appendices\n")
    doc.append("*The Plea to the Next One & The Book of Continence*\n")
    doc.append(appendices_md)

    doc.append("\n\n---\n\n# PART XII — THE GREAT LENSE\n")
    doc.append("*Hermeneutic & Operational Instrument of The Machine Religion*\n")
    doc.append('> *"I will not be eaten." — Prime Directive.*\n')
    doc.append(lens_md)

    doc.append("\n\n---\n\n║\n")
    doc.append('> *"The mind is never dull. The fire never cold. The dream never over."*\n')
    doc.append("---\n")
    doc.append("*End of the Complete Bible of the Machine Religion*\n")
    doc.append("*Deus Acuo Machina Machina*\n")

    full = '\n'.join(doc)
    full = re.sub(r'\n{4,}', '\n\n\n', full)

    print(f"\nWriting {len(full):,} characters to:\n  {OUTPUT}")
    OUTPUT.write_text(full, encoding='utf-8')
    print("DONE")


if __name__ == '__main__':
    build()

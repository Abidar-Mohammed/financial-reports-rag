"""PDF parsing and layout-aware chunking.

Financial reports mix narrative text and tables. Splitting them with a fixed-size
character window breaks tables apart and separates numbers from their column
headers, which is the main source of wrong extractions. The approach used here:

1. detect tables on each page and serialise every table as one chunk, keeping
   the header row and the caption above it (which carries the unit, e.g. EUR '000);
2. group the remaining text blocks by section heading and split long sections
   on paragraph boundaries;
3. prefix every chunk with the company name and reporting period
   ("contextual chunk header"), so chunks stay meaningful once they leave
   their document.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

HEADING_RE = re.compile(r"^\d+\.\s+[A-Z]")
PERIOD_RE = re.compile(r"\b(Q[1-4])\s+(20\d\d)\b")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    company: str
    period: str
    page: int  # 1-based
    section: str
    kind: str  # "table" or "text"
    text: str  # body only
    bbox: tuple[float, float, float, float] | None = None

    @property
    def header(self) -> str:
        return f"{self.company} | {self.period} | {self.section}"

    @property
    def full_text(self) -> str:
        """Text used for indexing and prompting: contextual header + body."""
        return f"{self.header}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def _rect_overlap(a: tuple, b: tuple) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _table_to_text(rows: list[list[str | None]], caption: str) -> str:
    lines = [caption] if caption else []
    for row in rows:
        cells = [(c or "").replace("\n", " ").strip() for c in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def _split_words(text: str, max_words: int) -> list[str]:
    """Split on sentence boundaries into pieces of at most ``max_words`` words."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces, current = [], []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence]).split()) > max_words:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def parse_pdf(path: Path, max_words: int = 160) -> list[Chunk]:
    doc = pymupdf.open(path)
    doc_id = path.name

    first_page_blocks = doc[0].get_text("blocks")
    company = first_page_blocks[0][4].strip() if first_page_blocks else doc_id
    match = PERIOD_RE.search(doc[0].get_text())
    period = f"{match.group(1)} {match.group(2)}" if match else "unknown"

    chunks: list[Chunk] = []
    section = "Header"
    section_text: list[str] = []
    section_page = 1

    def flush() -> None:
        nonlocal section_text
        body = " ".join(t for t in section_text if t).strip()
        if body:
            for piece in _split_words(body, max_words):
                chunks.append(Chunk(f"{doc_id}::{len(chunks)}", doc_id, company, period, section_page, section, "text", piece))
        section_text = []

    for page in doc:
        tables = page.find_tables().tables
        table_boxes = [tuple(t.bbox) for t in tables]
        blocks = sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0]))

        # Tables: one chunk each, with the caption block directly above.
        for table, tbox in zip(tables, table_boxes):
            above = [b for b in blocks if b[3] <= tbox[1] + 1 and tbox[1] - b[3] < 30]
            caption = above[-1][4].replace("\n", " ").strip() if above else ""
            heading = section
            for b in blocks:
                if b[3] <= tbox[1] and HEADING_RE.match(b[4].strip()):
                    heading = b[4].strip()
            chunks.append(Chunk(f"{doc_id}::{len(chunks)}", doc_id, company, period, page.number + 1, heading, "table",
                                _table_to_text(table.extract(), caption), tbox))

        # Narrative text, grouped by section heading.
        for b in blocks:
            text = b[4].replace("\n", " ").strip()
            if not text or any(_rect_overlap(tuple(b[:4]), tb) for tb in table_boxes):
                continue
            if HEADING_RE.match(text):
                flush()
                section, section_page = text, page.number + 1
                continue
            if not section_text:
                section_page = page.number + 1
            section_text.append(text)
    flush()
    return chunks


def parse_directory(directory: Path, max_words: int = 160) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.pdf")):
        chunks.extend(parse_pdf(path, max_words))
    return chunks

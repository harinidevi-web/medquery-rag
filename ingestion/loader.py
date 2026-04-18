import pdfplumber
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RawPage:
    page_num: int
    text: str
    has_table: bool
    tables: list[list] = field(default_factory=list)
    source_path: str = ""

def load_pdf(pdf_path: str) -> list[RawPage]:
    """
    Load a PDF page by page. Tables are extracted separately
    and flagged so the chunker never splits them.
    """
    pages = []
    path = Path(pdf_path)

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            has_table = len(tables) > 0

            # Remove table regions from prose text to avoid duplication
            if has_table:
                # Bounding boxes of tables — mask them out
                table_bboxes = [t.bbox for t in page.find_tables()]
                text = page.filter(
                    lambda obj: not _inside_any_bbox(obj, table_bboxes)
                ).extract_text() or ""
            else:
                text = page.extract_text() or ""

            pages.append(RawPage(
                page_num=i + 1,
                text=text.strip(),
                has_table=has_table,
                tables=tables,
                source_path=str(path),
            ))

    return pages


def _inside_any_bbox(obj, bboxes: list) -> bool:
    """Check if a PDF object falls inside any table bounding box."""
    x0, top, x1, bottom = obj.get("x0"), obj.get("top"), obj.get("x1"), obj.get("bottom")
    if None in (x0, top, x1, bottom):
        return False
    for bbox in bboxes:
        if x0 >= bbox[0] and top >= bbox[1] and x1 <= bbox[2] and bottom <= bbox[3]:
            return True
    return False


def table_to_text(table: list[list]) -> str:
    """Convert a pdfplumber table into readable text for the LLM."""
    rows = []
    for row in table:
        cleaned = [str(cell).strip() if cell else "" for cell in row]
        rows.append(" | ".join(cleaned))
    return "\n".join(rows)
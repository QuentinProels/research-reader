"""Layout-aware extraction with PyMuPDF.

This is the acknowledged bottleneck of the project. v0 handles single-column and
simple two-column layouts; dense/rotated/multi-panel layouts will need iteration
against a real corpus. Everything here is deterministic and unit-testable on
purpose -- the LLM pass downstream should be fixing prose, not rescuing a parse.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

CAPTION_RE = re.compile(r"^\s*(fig(?:ure)?\.?|table|tab\.)\s*(\d+|[IVXLC]+)", re.IGNORECASE)
SECTION_RE = re.compile(
    r"^\s*(?:(\d+(?:\.\d+)*)\s+)?"
    r"(abstract|introduction|background|related work|method(?:s|ology)?|approach|"
    r"experiments?|results?|evaluation|discussion|conclusions?|limitations?|"
    r"[A-Z][A-Za-z\s\-]{2,50})\s*$"
)
END_MATTER_RE = re.compile(
    r"^\s*(references|bibliography|acknowledge?ments?|appendix|supplementary)\b", re.IGNORECASE
)
FIGURE_MIN_PIXELS = 120  # ignore rules, logos, and separator hairlines


@dataclass
class Figure:
    id: str
    label: str
    page: int
    image_path: Path
    printed_caption: str = ""


@dataclass
class Section:
    title: str
    paragraphs: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


@dataclass
class ParsedPaper:
    title: str
    sections: list[Section]
    figures: list[Figure]
    page_count: int


class ParseError(RuntimeError):
    pass


def _is_two_column(page: fitz.Page, blocks: list[dict]) -> bool:
    """Two-column if text blocks cluster on either side of the page midline and
    few blocks straddle it."""
    if len(blocks) < 6:
        return False
    midline = page.rect.width / 2
    straddling = sum(1 for b in blocks if b["bbox"][0] < midline - 20 < b["bbox"][2])
    left = sum(1 for b in blocks if b["bbox"][2] <= midline + 20)
    right = sum(1 for b in blocks if b["bbox"][0] >= midline - 20)
    return straddling / len(blocks) < 0.15 and left >= 2 and right >= 2


def _reading_order(page: fitz.Page, blocks: list[dict]) -> list[dict]:
    """Sort text blocks the way a human reads them."""
    if _is_two_column(page, blocks):
        midline = page.rect.width / 2
        left = sorted((b for b in blocks if b["bbox"][0] < midline - 20), key=lambda b: b["bbox"][1])
        right = sorted(
            (b for b in blocks if b["bbox"][0] >= midline - 20), key=lambda b: b["bbox"][1]
        )
        return left + right
    return sorted(blocks, key=lambda b: (round(b["bbox"][1]), b["bbox"][0]))


def _block_text(block: dict) -> str:
    lines = []
    for line in block.get("lines", []):
        lines.append("".join(span["text"] for span in line.get("spans", [])))
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # rejoin hyphenated line breaks
    return re.sub(r"\s*\n\s*", " ", text).strip()


def _repeated_lines(pages_text: list[list[str]], page_count: int) -> set[str]:
    """Headers and footers: short lines that recur on most pages."""
    counts = Counter(
        line
        for page_lines in pages_text
        for line in {ln for ln in page_lines if len(ln) < 90}
    )
    threshold = max(3, int(page_count * 0.5))
    return {line for line, n in counts.items() if n >= threshold}


def _extract_figures(doc: fitz.Document, out_dir: Path) -> list[Figure]:
    """Render each embedded raster image large enough to be a real figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: list[Figure] = []
    for page_index, page in enumerate(doc):
        captions = [
            _block_text(b)
            for b in page.get_text("dict")["blocks"]
            if b.get("type") == 0 and CAPTION_RE.match(_block_text(b))
        ]
        for image_index, image in enumerate(page.get_images(full=True)):
            xref = image[0]
            try:
                pixmap = fitz.Pixmap(doc, xref)
            except (RuntimeError, ValueError):
                continue
            if pixmap.width < FIGURE_MIN_PIXELS or pixmap.height < FIGURE_MIN_PIXELS:
                continue
            if pixmap.colorspace and pixmap.colorspace.n > 3:  # CMYK -> RGB
                pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
            figure_id = f"p{page_index + 1}i{image_index}"
            image_path = out_dir / f"{figure_id}.png"
            pixmap.save(image_path)
            printed = captions[image_index] if image_index < len(captions) else ""
            match = CAPTION_RE.match(printed)
            label = (
                f"{match.group(1).rstrip('.').title()} {match.group(2)}"
                if match
                else f"Figure on page {page_index + 1}"
            )
            figures.append(
                Figure(
                    id=figure_id,
                    label=label,
                    page=page_index + 1,
                    image_path=image_path,
                    printed_caption=printed,
                )
            )
    return figures


def parse(pdf_path: Path, assets_dir: Path, max_pages: int = 500) -> ParsedPaper:
    doc = fitz.open(pdf_path)
    try:
        if doc.page_count > max_pages:
            raise ParseError(
                f"{doc.page_count} pages exceeds the {max_pages}-page cap. "
                "Refusing to parse -- this is the decompression-bomb guard."
            )
        if doc.page_count == 0:
            raise ParseError("PDF has no pages.")

        pages_blocks: list[list[dict]] = []
        pages_lines: list[list[str]] = []
        for page in doc:
            blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
            ordered = _reading_order(page, blocks)
            pages_blocks.append(ordered)
            pages_lines.append([_block_text(b) for b in ordered])

        boilerplate = _repeated_lines(pages_lines, doc.page_count)

        title = (doc.metadata or {}).get("title") or ""
        if not title.strip() and pages_lines and pages_lines[0]:
            title = pages_lines[0][0][:200]
        title = title.strip() or pdf_path.stem

        sections: list[Section] = [Section(title="Body")]
        for page_lines in pages_lines:
            for text in page_lines:
                if not text or text in boilerplate:
                    continue
                if END_MATTER_RE.match(text):
                    return ParsedPaper(
                        title=title,
                        sections=[s for s in sections if s.paragraphs],
                        figures=_extract_figures(doc, assets_dir),
                        page_count=doc.page_count,
                    )
                if CAPTION_RE.match(text):
                    continue  # captions are re-inserted from the vision pass
                if len(text) < 80 and SECTION_RE.match(text):
                    sections.append(Section(title=text))
                    continue
                sections[-1].paragraphs.append(text)

        return ParsedPaper(
            title=title,
            sections=[s for s in sections if s.paragraphs],
            figures=_extract_figures(doc, assets_dir),
            page_count=doc.page_count,
        )
    finally:
        doc.close()

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
FIGURE_MIN_POINTS = 40  # minimum on-page size of a placed image, in PDF points
MAX_CAPTION_GAP = 150  # a figure further than this from a caption is not its figure
MIN_REGION_POINTS = 40  # a rendered fallback region smaller than this is not a figure
MAX_REGION_POINTS = 420  # and one larger than this is most of a page, not a figure
PROSE_MIN_LINE_CHARS = 35  # below this a "paragraph" is really a column of table cells
RENDER_DPI = 150
BLANK_STD = 3.0  # pixel std below this means the region is blank paper


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


def _caption_label(text: str) -> str:
    """'Fig. 3: The architecture' -> 'Figure 3'. The label has to round-trip through
    pipeline._reference_pattern to find its in-text mentions, so normalise the kind."""
    match = CAPTION_RE.match(text)
    if not match:
        return ""
    kind = "Figure" if match.group(1).lower().startswith("fig") else "Table"
    return f"{kind} {match.group(2)}"


def _horizontally_overlaps(a: fitz.Rect, b: fitz.Rect) -> bool:
    """Same column? A figure and its caption always share horizontal space."""
    overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
    return overlap > 0.3 * min(a.width, b.width)


def _candidate_images(caption: fitz.Rect, images: list[list]) -> list[list]:
    """Every unclaimed image that plausibly belongs to this caption, nearest first.

    Captions sit below their figure far more often than above, so an image above the
    caption wins ties against an equally distant one below. Index-based pairing (the
    Nth image with the Nth caption) got this wrong whenever a page held a logo, a
    rotated panel, or figures in an order that did not match the captions.

    Returns a list rather than a single best match because multi-panel figures are one
    caption over several images -- "Figure 2: (left) ... (right) ..." is two embedded
    images, and handing the vision model only the left panel while the caption
    describes both is worse than handing it a render of the whole region.
    """
    scored: list[tuple[float, list]] = []
    for entry in images:
        rect, _xref, claimed = entry
        if claimed or not _horizontally_overlaps(rect, caption):
            continue
        if rect.y1 <= caption.y0 + 2:  # figure above its caption -- the normal case
            gap, penalty = caption.y0 - rect.y1, 1.0
        elif rect.y0 >= caption.y1 - 2:  # caption above its figure
            gap, penalty = rect.y0 - caption.y1, 1.6
        else:
            gap, penalty = 0.0, 2.0  # caption overlaps the image box
        if gap > MAX_CAPTION_GAP:
            continue
        scored.append((gap * penalty, entry))
    scored.sort(key=lambda pair: pair[0])
    return [entry for _score, entry in scored]


def _is_blank(pixmap: fitz.Pixmap) -> bool:
    import numpy as np

    return float(np.frombuffer(pixmap.samples, dtype=np.uint8).std()) < BLANK_STD


def _looks_like_prose(block: dict, text: str) -> bool:
    """Running text, as opposed to a table PyMuPDF merged into a single block.

    The region a caption owns must stop at real paragraphs but must NOT stop at the
    contents of a table -- and a whole table often arrives as one text block that is
    long enough to pass any length test. Bounding on that collapsed every table region
    to nothing and silently dropped all four tables in a test paper.

    Mean line length separates them cleanly. Measured on that paper: table blocks
    averaged 10 and 15 characters per line, body paragraphs 63 to 100.
    """
    if len(text) <= 120 or len(text.split()) <= 20:
        return False
    lines = block.get("lines", [])
    if not lines:
        return False
    mean_line = sum(
        len("".join(span["text"] for span in line.get("spans", []))) for line in lines
    ) / len(lines)
    return mean_line >= PROSE_MIN_LINE_CHARS


def _neighbour_bounds(
    caption: fitz.Rect, body: list[tuple[fitz.Rect, bool]], page: fitz.Rect
) -> tuple[float, float]:
    """How far above and below the caption we can go before hitting a real paragraph."""
    above = page.y0 + 24
    below = page.y1 - 24
    for rect, is_prose in body:
        if not is_prose or not _horizontally_overlaps(rect, caption):
            continue
        if rect.y1 <= caption.y0:
            above = max(above, rect.y1)
        elif rect.y0 >= caption.y1:
            below = min(below, rect.y0)
    return max(above, caption.y0 - MAX_REGION_POINTS), min(below, caption.y1 + MAX_REGION_POINTS)


def _render_fallback(
    page: fitz.Page, caption: fitz.Rect, body: list[tuple[fitz.Rect, bool]], path: Path
):
    """Rasterise the page region belonging to a caption that owns no embedded image.

    Most academic plots are vector drawings, so get_images() returns nothing for them
    and the paper silently loses its figures. Tables are pure text and are missed the
    same way -- rendering them is a bonus, since the vision model can then read one.

    Tries the region above the caption and the region below, and keeps whichever holds
    more ink, which is what makes this work regardless of caption convention.
    """
    above, below = _neighbour_bounds(caption, body, page.rect)
    candidates = [
        fitz.Rect(caption.x0, above + 2, caption.x1, caption.y0 - 2),
        fitz.Rect(caption.x0, caption.y1 + 2, caption.x1, below - 2),
    ]
    best, best_ink = None, 0.0
    for clip in candidates:
        if clip.height < MIN_REGION_POINTS or clip.width < MIN_REGION_POINTS:
            continue
        pixmap = page.get_pixmap(clip=clip, dpi=RENDER_DPI)
        if _is_blank(pixmap):
            continue
        import numpy as np

        ink = float(np.frombuffer(pixmap.samples, dtype=np.uint8).std())
        if ink > best_ink:
            best, best_ink = pixmap, ink
    if best is None:
        return False
    best.save(path)
    return True


def _save_embedded(doc: fitz.Document, xref: int, path: Path) -> bool:
    try:
        pixmap = fitz.Pixmap(doc, xref)
    except (RuntimeError, ValueError):
        return False
    if pixmap.width < FIGURE_MIN_PIXELS or pixmap.height < FIGURE_MIN_PIXELS:
        return False
    if pixmap.colorspace is None or pixmap.colorspace.n > 3:  # CMYK/gray oddities -> RGB
        pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
    pixmap.save(path)
    return True


def _extract_figures(doc: fitz.Document, out_dir: Path) -> list[Figure]:
    """Find figures by their captions, then attach the right picture to each.

    Driven from captions rather than from images: the caption is what carries the label
    the narration needs ("Figure 3"), and a picture with no caption is nearly always a
    logo or a rule rather than something worth describing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: list[Figure] = []
    for page_index, page in enumerate(doc):
        captions: list[tuple[fitz.Rect, str]] = []
        body: list[tuple[fitz.Rect, bool]] = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            text = _block_text(block)
            if not text:
                continue
            if CAPTION_RE.match(text):
                captions.append((fitz.Rect(block["bbox"]), text))
            else:
                body.append((fitz.Rect(block["bbox"]), _looks_like_prose(block, text)))
        if not captions:
            continue

        images: list[list] = []
        for info in page.get_image_info(xrefs=True):
            rect = fitz.Rect(info["bbox"])
            if rect.width >= FIGURE_MIN_POINTS and rect.height >= FIGURE_MIN_POINTS:
                images.append([rect, info["xref"], False])

        for order, (caption_rect, caption_text) in enumerate(
            sorted(captions, key=lambda c: c[0].y0)
        ):
            figure_id = f"p{page_index + 1}c{order}"
            image_path = out_dir / f"{figure_id}.png"
            candidates = _candidate_images(caption_rect, images)
            for entry in candidates:
                entry[2] = True
            if len(candidates) == 1:
                saved = _save_embedded(doc, candidates[0][1], image_path)
            else:
                # Zero images means a vector figure or a table; more than one means a
                # multi-panel figure. Both are served by rendering the whole region.
                saved = _render_fallback(page, caption_rect, body, image_path)
            if not saved and candidates:
                saved = _render_fallback(page, caption_rect, body, image_path)
            if not saved:
                continue
            figures.append(
                Figure(
                    id=figure_id,
                    label=_caption_label(caption_text) or f"Figure on page {page_index + 1}",
                    page=page_index + 1,
                    image_path=image_path,
                    printed_caption=caption_text,
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

"""Generate synthetic PDF fixtures for testing pdf.py layout extraction."""

from pathlib import Path

import fitz


def _new_doc() -> fitz.Document:
    return fitz.open()


def _save_and_close(doc: fitz.Document, path: Path) -> None:
    doc.save(path)
    doc.close()


def _add_text_block(page: fitz.Page, x0: float, y0: float, text: str,
                    fontsize: float = 11, width: float = 500) -> None:
    """Insert text, wrapping at width."""
    page.insert_text(fitz.Point(x0, y0), text, fontsize=fontsize, color=(0, 0, 0))


def single_column(doc: fitz.Document) -> None:
    """A simple single-column page with title and paragraphs."""
    page = doc.new_page(width=612, height=792)
    _add_text_block(page, 72, 72, "Single Column Title", fontsize=18)
    _add_text_block(page, 72, 110, 
        "This is the first paragraph of a single-column document. "
        "It contains enough text to form a meaningful block for testing. "
        "The reading order should be straightforward top-to-bottom.",
        width=468)
    _add_text_block(page, 72, 170,
        "This is the second paragraph. It follows the first one in a "
        "simple vertical flow. There are no columns or complex layouts here.",
        width=468)
    _add_text_block(page, 72, 230,
        "A third paragraph to ensure multi-paragraph extraction works correctly.",
        width=468)


def two_column(doc: fitz.Document) -> None:
    """A clean two-column layout."""
    page = doc.new_page(width=612, height=792)
    _add_text_block(page, 72, 72, "Two Column Title", fontsize=16)
    # Left column
    _add_text_block(page, 72, 120, 
        "Left column paragraph one. This text sits in the left column "
        "of a two-column layout. It should be read before the right column.",
        width=200)
    _add_text_block(page, 72, 200,
        "Left column paragraph two. More text in the left column to "
        "ensure multiple blocks are detected.",
        width=200)
    # Right column
    _add_text_block(page, 340, 120,
        "Right column paragraph one. This text sits in the right column "
        "and should be read after the left column is complete.",
        width=200)
    _add_text_block(page, 340, 200,
        "Right column paragraph two. More text in the right column.",
        width=200)


def two_column_with_fullwidth_figure(doc: fitz.Document) -> None:
    """Two-column page with a full-width figure spanning both columns."""
    page = doc.new_page(width=612, height=792)
    _add_text_block(page, 72, 72, "Two Column With Figure", fontsize=16)
    # Left column top
    _add_text_block(page, 72, 120, "Left column text before the figure.", width=200)
    # Right column top
    _add_text_block(page, 340, 120, "Right column text before the figure.", width=200)
    # Full-width figure spanning both columns
    page.draw_rect(fitz.Rect(72, 200, 540, 300), color=(0, 0, 0), width=1, fill=(0.9, 0.9, 0.9))
    _add_text_block(page, 72, 310, "Figure 1: A full-width figure spanning both columns.", fontsize=10)
    # Left column bottom
    _add_text_block(page, 72, 350, "Left column text after the figure.", width=200)
    # Right column bottom
    _add_text_block(page, 340, 350, "Right column text after the figure.", width=200)


def headers_footers(doc: fitz.Document) -> None:
    """A page with repeated running header and footer plus page numbers."""
    for page_num in range(1, 4):
        page = doc.new_page(width=612, height=792)
        # Running header
        _add_text_block(page, 72, 50, "Journal of AI Research, Vol. 42", fontsize=10)
        # Page number in footer
        _add_text_block(page, 300, 770, f"{page_num}", fontsize=10)
        # Running footer
        _add_text_block(page, 72, 755, "Published under CC-BY 4.0", fontsize=9)
        # Body text
        _add_text_block(page, 72, 100, 
            f"This is body text on page {page_num}. The headers and footers "
            "should be stripped as boilerplate.",
            width=468)
        _add_text_block(page, 72, 150,
            "More body text that varies per page so it is not stripped.",
            width=468)


def references_section(doc: fitz.Document) -> None:
    """A document with a references section partway through."""
    page = doc.new_page(width=612, height=792)
    _add_text_block(page, 72, 72, "Paper With References", fontsize=16)
    _add_text_block(page, 72, 120, 
        "This is the body of the paper. It discusses interesting results "
        "and methods. Everything before references should be kept.",
        width=468)
    _add_text_block(page, 72, 180,
        "More content in the body section of the paper.",
        width=468)
    _add_text_block(page, 72, 240, "References", fontsize=14)
    _add_text_block(page, 72, 270, "[1] Smith et al., A Great Paper, 2023", width=468)
    _add_text_block(page, 72, 290, "[2] Jones et al., Another Paper, 2024", width=468)
    _add_text_block(page, 72, 310, "[3] Lee et al., Yet Another Paper, 2022", width=468)


def hyphenated_words(doc: fitz.Document) -> None:
    """Text with hyphenated words broken across lines."""
    page = doc.new_page(width=612, height=792)
    _add_text_block(page, 72, 72, "Hyphenated Words Test", fontsize=16)
    # Use narrow width to force hyphenation
    _add_text_block(page, 72, 120,
        "This paragraph contains a word that is broken across lines like "
        "experi-\nment and devel-\noped. The extraction should rejoin these "
        "into 'experiment' and 'developed'.",
        width=300)
    _add_text_block(page, 72, 200,
        "Another example: the word docu-\nmentation appears here broken.",
        width=300)


def caption_above_figure(doc: fitz.Document) -> None:
    """A figure caption sitting above its figure rather than below."""
    page = doc.new_page(width=612, height=792)
    _add_text_block(page, 72, 72, "Caption Above Figure Test", fontsize=16)
    # Caption ABOVE the figure
    _add_text_block(page, 72, 130, "Figure 2: A chart showing experimental results.", fontsize=10)
    # Figure below the caption
    page.draw_rect(fitz.Rect(72, 160, 350, 300), color=(0, 0, 0), width=1, fill=(0.85, 0.85, 0.85))
    _add_text_block(page, 72, 330, "Text after the figure and its caption.", width=468)


def title_with_font_size(doc: fitz.Document) -> None:
    """A page where the first text block is a small license stamp and the real title is larger."""
    page = doc.new_page(width=612, height=792)
    # Small-font license stamp at top
    _add_text_block(page, 72, 72, 
        "Provided proper attribution is provided, Google hereby grants permission...",
        fontsize=8)
    # Large-font real title below
    _add_text_block(page, 72, 110, "Attention Is All You Need", fontsize=22)
    # Body text
    _add_text_block(page, 72, 160,
        "The dominant sequence transduction models are based on complex "
        "recurrent or convolutional neural networks that include an encoder "
        "and a decoder.",
        width=468)


def generate_all(output_dir: Path) -> dict[str, Path]:
    """Generate all fixture PDFs and return a dict mapping scenario name -> path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = {
        "single_column": single_column,
        "two_column": two_column,
        "two_column_with_fullwidth_figure": two_column_with_fullwidth_figure,
        "headers_footers": headers_footers,
        "references_section": references_section,
        "hyphenated_words": hyphenated_words,
        "caption_above_figure": caption_above_figure,
        "title_with_font_size": title_with_font_size,
    }
    result = {}
    for name, builder in scenarios.items():
        doc = _new_doc()
        builder(doc)
        path = output_dir / f"{name}.pdf"
        _save_and_close(doc, path)
        result[name] = path
    return result

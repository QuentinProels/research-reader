"""Caption-to-figure association.

Fixtures are generated with PyMuPDF rather than checked in, so the expected geometry is
visible in the test rather than buried in a binary. They draw figures with draw_rect,
which produces vector content -- so they also exercise the render fallback, since
get_images() returns nothing for a vector plot.
"""

import fitz
import pytest

from app import pdf
from tests.fixtures.make_pdfs import generate_all


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    return generate_all(tmp_path_factory.mktemp("pdfs"))


def _parse(path, tmp_path):
    return pdf.parse(path, tmp_path / "figs")


class TestCaptionLabel:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Figure 1: The Transformer architecture.", "Figure 1"),
            ("Fig. 2 Attention weights over time", "Figure 2"),
            ("fig 3. Something lowercase", "Figure 3"),
            ("Table 4: Ablation results", "Table 4"),
            ("Tab. 5 Comparison", "Table 5"),
        ],
    )
    def test_normalises_the_kind_and_number(self, text, expected):
        assert pdf._caption_label(text) == expected

    def test_returns_empty_for_ordinary_prose(self):
        assert pdf._caption_label("The figure shows a rising curve.") == ""

    def test_label_round_trips_through_the_reference_matcher(self):
        """The label is only useful if it can find its own in-text mentions."""
        from app.pipeline import _reference_pattern

        label = pdf._caption_label("Fig. 7: Results")
        assert _reference_pattern(label).search("As shown in Figure 7, accuracy improves.")


class TestGeometricAssociation:
    def _img(self, x0, y0, x1, y1, xref=1):
        return [fitz.Rect(x0, y0, x1, y1), xref, False]

    def test_prefers_the_image_directly_above_the_caption(self):
        caption = fitz.Rect(72, 310, 540, 325)
        far = self._img(72, 60, 540, 150, xref=1)
        near = self._img(72, 200, 540, 300, xref=2)
        assert pdf._candidate_images(caption, [far, near])[0][1] == 2

    def test_finds_an_image_below_when_the_caption_sits_above_it(self):
        caption = fitz.Rect(72, 130, 350, 145)
        below = self._img(72, 160, 350, 300, xref=9)
        assert pdf._candidate_images(caption, [below])[0][1] == 9

    def test_an_image_above_beats_an_equally_distant_one_below(self):
        caption = fitz.Rect(72, 200, 400, 215)
        above = self._img(72, 100, 400, 180, xref=1)
        below = self._img(72, 235, 400, 315, xref=2)
        assert pdf._candidate_images(caption, [above, below])[0][1] == 1

    def test_ignores_an_image_in_another_column(self):
        caption = fitz.Rect(72, 310, 280, 325)
        other_column = self._img(320, 200, 540, 300)
        assert pdf._candidate_images(caption, [other_column]) == []

    def test_ignores_an_image_on_the_far_side_of_the_page(self):
        caption = fitz.Rect(72, 700, 540, 715)
        distant = self._img(72, 60, 540, 150)
        assert pdf._candidate_images(caption, [distant]) == []

    def test_an_image_is_claimed_by_only_one_caption(self):
        images = [self._img(72, 200, 540, 300, xref=5)]
        first = fitz.Rect(72, 310, 540, 325)
        second = fitz.Rect(72, 330, 540, 345)
        first_match = pdf._candidate_images(first, images)
        assert first_match[0][1] == 5
        first_match[0][2] = True  # caller marks it claimed
        assert pdf._candidate_images(second, images) == []


class TestVectorFigureFallback:
    def test_a_vector_figure_is_captured_even_though_it_is_not_an_embedded_image(
        self, fixtures, tmp_path
    ):
        """Regression: get_images() returns nothing for a vector plot, so these figures
        used to vanish entirely from the narration."""
        path = fixtures["two_column_with_fullwidth_figure"]
        assert not fitz.open(path)[0].get_images(), "fixture should be vector, not raster"

        figures = _parse(path, tmp_path).figures
        assert [f.label for f in figures] == ["Figure 1"]
        assert figures[0].image_path.exists()
        assert figures[0].image_path.stat().st_size > 0

    def test_it_works_when_the_caption_sits_above_the_figure(self, fixtures, tmp_path):
        figures = _parse(fixtures["caption_above_figure"], tmp_path).figures
        assert [f.label for f in figures] == ["Figure 2"]
        assert figures[0].image_path.exists()

    def test_the_rendered_region_is_not_blank(self, fixtures, tmp_path):
        figures = _parse(fixtures["caption_above_figure"], tmp_path).figures
        rendered = fitz.Pixmap(str(figures[0].image_path))
        assert not pdf._is_blank(rendered)

    def test_the_printed_caption_is_kept_for_the_vision_prompt(self, fixtures, tmp_path):
        figures = _parse(fixtures["caption_above_figure"], tmp_path).figures
        assert "chart showing experimental results" in figures[0].printed_caption

    def test_a_page_with_no_captions_yields_no_figures(self, fixtures, tmp_path):
        assert _parse(fixtures["single_column"], tmp_path).figures == []


class TestProseDetection:
    """Separating a real paragraph from a table that PyMuPDF merged into one block.

    Regression: bounding a caption's region on any long text block collapsed every
    table region to zero height, and all four tables in a real paper were dropped.
    Measured on that paper, table blocks averaged 10 and 15 characters per line while
    body paragraphs averaged 63 to 100.
    """

    def _block(self, lines):
        return {"lines": [{"spans": [{"text": line}]} for line in lines]}

    def test_a_paragraph_is_prose(self):
        lines = [
            "The dominant sequence transduction models are based on complex recurrent",
            "or convolutional neural networks that include an encoder and a decoder.",
            "The best performing models also connect the encoder and decoder through",
        ]
        assert pdf._looks_like_prose(self._block(lines), " ".join(lines))

    def test_a_merged_table_is_not_prose(self):
        lines = ["Layer Type", "Self-Attention", "O(1)", "Recurrent", "O(n)"] * 5
        assert not pdf._looks_like_prose(self._block(lines), " ".join(lines))

    def test_a_short_block_is_not_prose(self):
        assert not pdf._looks_like_prose(self._block(["Encoder:"]), "Encoder:")

    def test_a_block_with_no_lines_is_not_prose(self):
        assert not pdf._looks_like_prose({"lines": []}, "x " * 200)


class TestMultiPanelFigures:
    def test_several_images_under_one_caption_are_all_claimed(self):
        """"Figure 2: (left) ... (right) ..." is two embedded images under one caption.
        Claiming only the nearest handed the vision model half the figure."""
        caption = fitz.Rect(72, 320, 540, 340)
        left = [fitz.Rect(72, 200, 280, 300), 1, False]
        right = [fitz.Rect(300, 200, 540, 300), 2, False]
        assert len(pdf._candidate_images(caption, [left, right])) == 2

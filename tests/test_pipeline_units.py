"""Unit tests for the deterministic pieces: chunking, caption placement, URL resolution.

The LLM and TTS calls are not tested here -- they need a live model server.
"""

import pytest

from app import ingest
from app.pipeline import _insert_captions, _reference_pattern
from app.tts import chunk_text


class TestChunkText:
    def test_splits_on_sentence_boundaries(self):
        text = " ".join(f"Sentence number {i} goes here." for i in range(60))
        chunks = chunk_text(text, max_chars=200)
        assert all(len(c) <= 200 for c in chunks)
        assert "".join(chunks).replace(" ", "") == text.replace(" ", "")

    def test_short_text_is_one_chunk(self):
        assert chunk_text("Just one sentence.") == ["Just one sentence."]

    def test_run_on_longer_than_limit_is_hard_split(self):
        chunks = chunk_text("word " * 400, max_chars=100)
        assert all(len(c) <= 100 for c in chunks)

    def test_empty_text_yields_nothing(self):
        assert chunk_text("   ") == []


class TestReferencePattern:
    @pytest.mark.parametrize(
        "label,sentence",
        [
            ("Figure 3", "As shown in Figure 3, the loss drops."),
            ("Figure 3", "See fig. 3 for details."),
            ("Table 2", "Table 2 reports the ablation."),
        ],
    )
    def test_matches_in_text_references(self, label, sentence):
        assert _reference_pattern(label).search(sentence)

    def test_does_not_match_a_different_number(self):
        assert not _reference_pattern("Figure 3").search("Figure 30 shows something else.")


class TestInsertCaptions:
    def test_caption_lands_after_the_referencing_sentence(self):
        figures = [{"id": "f1", "label": "Figure 1", "caption": "A rising curve."}]
        segments = _insert_captions("First line. Figure 1 shows the trend. Then more.", figures)
        texts = [s["text"] for s in segments]
        assert texts[1].startswith("Figure 1 shows")
        assert texts[2] == "Figure 1. A rising curve."
        assert segments[2]["figure_id"] == "f1"

    def test_unreferenced_figure_goes_to_the_end(self):
        figures = [{"id": "f9", "label": "Figure 9", "caption": "An orphan chart."}]
        segments = _insert_captions("No mention here at all.", figures)
        assert segments[-1]["figure_id"] == "f9"

    def test_figure_is_inserted_only_once(self):
        figures = [{"id": "f1", "label": "Figure 1", "caption": "A curve."}]
        segments = _insert_captions("Figure 1 here. And Figure 1 again.", figures)
        assert sum(1 for s in segments if s["figure_id"] == "f1") == 1

    def test_figure_without_a_caption_is_skipped(self):
        segments = _insert_captions("Figure 1 here.", [{"id": "f1", "label": "Figure 1", "caption": ""}])
        assert all(s["figure_id"] is None for s in segments)


class TestIngest:
    def test_arxiv_abs_resolves_to_pdf(self):
        assert ingest.resolve_url("https://arxiv.org/abs/2401.12345") == "https://arxiv.org/pdf/2401.12345"

    def test_direct_pdf_url_passes_through(self):
        url = "https://example.org/paper.pdf"
        assert ingest.resolve_url(url) == url

    def test_non_http_is_rejected(self):
        with pytest.raises(ingest.IngestError):
            ingest.resolve_url("ftp://example.org/paper.pdf")

    def test_magic_bytes_reject_non_pdf(self):
        with pytest.raises(ingest.IngestError):
            ingest.check_magic_bytes(b"PK\x03\x04\x00")  # a zip pretending to be a PDF

    def test_magic_bytes_accept_pdf(self):
        ingest.check_magic_bytes(b"%PDF-1.7")

"""Unit tests for the deterministic pieces: chunking, caption placement, URL resolution.

The LLM and TTS calls are not tested here -- they need a live model server.
"""

import pytest

from app import ingest
from app.pipeline import _insert_captions, _reference_pattern, _trailing_captions
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
        segments = _insert_captions(
            "First line. Figure 1 shows the trend. Then more.", figures, set()
        )
        texts = [s["text"] for s in segments]
        assert texts[1].startswith("Figure 1 shows")
        assert texts[2] == "Figure 1. A rising curve."
        assert segments[2]["figure_id"] == "f1"

    def test_figure_is_inserted_only_once_within_a_section(self):
        figures = [{"id": "f1", "label": "Figure 1", "caption": "A curve."}]
        segments = _insert_captions("Figure 1 here. And Figure 1 again.", figures, set())
        assert sum(1 for s in segments if s["figure_id"] == "f1") == 1

    def test_figure_is_not_reread_in_a_later_section(self):
        """Regression: `placed` used to be local to each call, so a paper that referred
        back to Figure 1 in 27 sections narrated its full description 27 times."""
        figures = [{"id": "f1", "label": "Figure 1", "caption": "A curve."}]
        placed: set[str] = set()
        sections = ["Figure 1 introduced here.", "Recall Figure 1.", "Figure 1 once more."]
        total = sum(
            1
            for section in sections
            for segment in _insert_captions(section, figures, placed)
            if segment["figure_id"] == "f1"
        )
        assert total == 1

    def test_figure_without_a_caption_is_skipped(self):
        segments = _insert_captions(
            "Figure 1 here.", [{"id": "f1", "label": "Figure 1", "caption": ""}], set()
        )
        assert all(s["figure_id"] is None for s in segments)


class TestTrailingCaptions:
    def test_unreferenced_figure_is_read_once_at_the_end(self):
        figures = [{"id": "f9", "label": "Figure 9", "caption": "An orphan chart."}]
        assert [s["figure_id"] for s in _trailing_captions(figures, set())] == ["f9"]

    def test_already_placed_figure_is_not_repeated(self):
        figures = [{"id": "f1", "label": "Figure 1", "caption": "A curve."}]
        assert _trailing_captions(figures, {"f1"}) == []

    def test_figure_without_a_caption_is_not_appended(self):
        assert _trailing_captions([{"id": "f1", "label": "Figure 1", "caption": ""}], set()) == []


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


class TestReflowPreservesReferences:
    """Runs against the live llama-server; skipped when it is not up.

    Regression: the reflow prompt told the model to strip parenthetical citation
    markers, and it generalised that to "(Figure 2)" and rewrote "Table 1" as "the
    corresponding table". Every figure reference vanished, so every figure description
    fell through to the end of the recording instead of landing next to the text that
    explains it -- which is the one thing this project exists to do.
    """

    @pytest.fixture(autouse=True)
    def _require_llm(self):
        from app import llm

        ok, detail = llm.health()
        if not ok:
            pytest.skip(f"llama-server unavailable: {detail}")

    def test_figure_and_table_references_survive(self):
        from app import llm

        out = llm.reflow(
            'We call our attention "Scaled Dot-Product Attention" (Figure 2). '
            "The input has dimension dk [5]. The model follows this architecture "
            "(Figure 1), using stacked layers (Smith et al., 2020). See Table 1."
        )
        for label in ("Figure 1", "Figure 2", "Table 1"):
            assert _reference_pattern(label).search(out), f"{label} was dropped by reflow: {out!r}"

    def test_citations_are_still_stripped(self):
        from app import llm

        out = llm.reflow("The model has dimension dk [5], improving on prior art (Smith et al., 2020).")
        assert "[5]" not in out and "Smith" not in out


class TestRefusalDetection:
    """The reflow model sometimes answers about a passage instead of returning it, and
    the answer then gets narrated in the same voice as the paper. Found in the wild:
    2 chunks in 2244 said "no substantive prose to clean" mid-section."""


    @pytest.mark.parametrize(
        "text",
        [
            "As there is no substantive prose to clean, summarize, or read aloud, no output can be generated",
            "There is no prose here to reflow.",
            "I cannot process this block.",
            "The provided text contains no readable sentences.",
        ],
    )
    def test_model_commentary_is_recognised(self, text):
        from app.pipeline import _is_refusal

        assert _is_refusal(text)

    def test_real_prose_is_kept(self):
        from app.pipeline import _is_refusal

        assert not _is_refusal(
            "The encoder is composed of a stack of six identical layers, each with a "
            "multi-head self attention mechanism and a position wise feed forward network."
        )

    def test_a_long_passage_mentioning_the_phrase_is_kept(self):
        """A paper discussing what a model cannot do is prose, not a refusal."""
        from app.pipeline import _is_refusal

        long_passage = (
            "Prior work observed that the model cannot represent positions beyond its "
            "training length, and that no output can be generated past that horizon "
            "without extrapolation. " * 4
        )
        assert not _is_refusal(long_passage)

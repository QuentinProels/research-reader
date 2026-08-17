"""PDF -> narrated audio, run synchronously in a worker thread.

Order matches the spec: parse, caption figures, reflow prose, insert captions at
their in-text reference points, chunk, render, concatenate with chapter markers.
"""

import logging
import re
import traceback
from pathlib import Path

from app import llm, pdf, store, tts
from app.config import settings

log = logging.getLogger(__name__)

REFLOW_BUDGET_CHARS = 6000  # per LLM call; well inside the 250k context, keeps latency sane


def _reference_pattern(label: str) -> re.Pattern | None:
    """Match in-text references to 'Figure 3' / 'Fig. 3' / 'Table 2'."""
    match = re.match(r"(figure|table)\s+(\S+)", label, re.IGNORECASE)
    if not match:
        return None
    kind, number = match.groups()
    prefix = "fig(?:ure)?\\.?" if kind.lower() == "figure" else "tab(?:le)?\\.?"
    return re.compile(rf"\b{prefix}\s*{re.escape(number)}\b", re.IGNORECASE)


def _insert_captions(prose: str, figures: list[dict], placed: set[str]) -> list[dict]:
    """Split prose into segments, dropping each figure description in at the first
    sentence that mentions it.

    `placed` is owned by the caller and spans the whole paper, not one section. It has
    to: a paper refers back to Figure 1 in half its sections, and a per-section set
    would re-read the full description every time.
    """
    sentences = re.split(r"(?<=[.!?])\s+", prose.strip())
    segments: list[dict] = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        segments.append({"text": sentence, "figure_id": None})
        for figure in figures:
            if figure["id"] in placed or not figure.get("caption"):
                continue
            pattern = _reference_pattern(figure["label"])
            if pattern and pattern.search(sentence):
                segments.append(
                    {
                        "text": f"{figure['label']}. {figure['caption']}",
                        "figure_id": figure["id"],
                    }
                )
                placed.add(figure["id"])
    return segments


def _trailing_captions(figures: list[dict], placed: set[str]) -> list[dict]:
    """Figures the prose never referred to. Read once, after the last section."""
    return [
        {"text": f"{figure['label']}. {figure['caption']}", "figure_id": figure["id"]}
        for figure in figures
        if figure["id"] not in placed and figure.get("caption")
    ]


def run(paper_id: str, pdf_path: Path) -> None:
    paper_dir = settings.papers_dir / paper_id
    try:
        # 1. parse
        store.set_status(paper_id, "parsing", 0.05, "Extracting text and figures")
        parsed = pdf.parse(pdf_path, paper_dir / "figures", max_pages=settings.max_pages)
        store.update_paper(paper_id, title=parsed.title)

        # 2. caption figures with the vision model
        figures: list[dict] = []
        for index, figure in enumerate(parsed.figures):
            store.set_status(
                paper_id,
                "captioning",
                0.10 + 0.25 * (index / max(len(parsed.figures), 1)),
                f"Describing {figure.label} ({index + 1} of {len(parsed.figures)})",
            )
            try:
                caption = llm.caption_figure(figure.image_path, figure.printed_caption)
            except llm.LLMUnavailable:
                raise  # the whole job should retry, not lose every figure description
            except llm.LLMError as exc:
                log.warning("caption failed for %s: %s", figure.id, exc)
                caption = figure.printed_caption
            figures.append(
                {
                    "id": figure.id,
                    "label": figure.label,
                    "page": figure.page,
                    "image_path": str(figure.image_path.relative_to(settings.data_dir)),
                    "caption": caption,
                }
            )

        # 3. reflow prose, section by section
        store.set_status(paper_id, "reflowing", 0.40, "Cleaning up reading order")
        chapters: list[dict] = []
        segments: list[dict] = []
        placed: set[str] = set()
        for index, section in enumerate(parsed.sections):
            store.set_status(
                paper_id,
                "reflowing",
                0.40 + 0.25 * (index / max(len(parsed.sections), 1)),
                f"Cleaning section {index + 1} of {len(parsed.sections)}",
            )
            cleaned_parts = []
            for start in range(0, len(section.text), REFLOW_BUDGET_CHARS):
                block = section.text[start : start + REFLOW_BUDGET_CHARS]
                try:
                    cleaned_parts.append(llm.reflow(block))
                except llm.LLMUnavailable:
                    raise  # narrating a whole paper from unreflowed text is worse
                except llm.LLMError as exc:
                    log.warning("reflow failed, using raw text: %s", exc)
                    cleaned_parts.append(block)
            chapters.append({"title": section.title, "segment_index": len(segments)})
            segments.extend(_insert_captions("\n\n".join(cleaned_parts), figures, placed))
        segments.extend(_trailing_captions(figures, placed))

        # 4. chunk + render
        store.set_status(paper_id, "synthesizing", 0.68, "Rendering audio")
        rendered: list[dict] = []
        for segment in segments:
            for piece in tts.chunk_text(segment["text"]):
                rendered.append({"text": piece, "figure_id": segment["figure_id"]})

        audio_path = paper_dir / "audio.wav"
        offsets = tts.render(
            [item["text"] for item in rendered],
            audio_path,
            on_progress=lambda done, total: store.set_status(
                paper_id,
                "synthesizing",
                0.68 + 0.30 * (done / max(total, 1)),
                f"Rendering audio ({done} of {total} segments)",
            ),
        )

        # 5. map chapter + figure markers onto real timestamps
        segment_starts = _segment_starts(segments, rendered, offsets)
        for chapter in chapters:
            chapter["start_s"] = segment_starts.get(chapter["segment_index"], 0.0)
            chapter.pop("segment_index")
        for figure in figures:
            figure["start_s"] = next(
                (
                    offsets[i]
                    for i, item in enumerate(rendered)
                    if item["figure_id"] == figure["id"]
                ),
                None,
            )

        duration = _duration(audio_path)
        store.update_paper(
            paper_id,
            status="ready",
            progress=1.0,
            detail="Ready",
            audio_path=str(audio_path.relative_to(settings.data_dir)),
            duration_s=duration,
            chapters=chapters,
            figures=figures,
            error=None,
        )
    except llm.LLMUnavailable:
        # Not this paper's fault. The worker decides whether to requeue or give up.
        log.warning("model server unreachable during %s; handing back to the worker", paper_id)
        raise
    except Exception as exc:  # noqa: BLE001 -- the status row is the error channel
        log.error("pipeline failed for %s: %s", paper_id, traceback.format_exc())
        store.update_paper(paper_id, status="failed", detail="", error=str(exc))


def _segment_starts(
    segments: list[dict], rendered: list[dict], offsets: list[float]
) -> dict[int, float]:
    """Which audio offset does each pre-chunk segment begin at?"""
    starts: dict[int, float] = {}
    cursor = 0
    for index, segment in enumerate(segments):
        pieces = len(tts.chunk_text(segment["text"]))
        if cursor < len(offsets):
            starts[index] = offsets[cursor]
        cursor += pieces
    return starts


def _duration(audio_path: Path) -> float:
    import soundfile as sf

    info = sf.info(audio_path)
    return info.frames / info.samplerate

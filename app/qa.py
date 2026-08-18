"""Answering a spoken question about the paper being listened to.

Context comes from where the listener actually is. A question during narration is nearly
always about the passage just heard, so the transcript around the current playback offset
answers most of them without any search at all. Step 7 adds a vector fallback for the
rest; until then, an unanswerable question says so rather than inventing something.

Answers do not use the model's reasoning mode, and that is a measured decision rather
than a latency compromise. Asked the same grounded question over the same passage, the
fast path answered in 4s and the reasoning path in 53s, and the fast answer was the more
specific of the two. When the answer is already in the passage the task is reading
comprehension, not deduction, so there is nothing for reasoning to do -- the same finding
as the reflow pass, which was 32x slower for byte-identical output.

Reasoning also has a failure mode here: it consumed an entire 800-token budget on its own
thinking and returned an empty answer after 48 seconds of silence.

This should be revisited when step 7 adds synthesis across retrieved passages, which is a
genuinely different task shape. Measure it again then rather than assuming either way.
"""

import logging
import re

from app import llm, store

log = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 6
MAX_CONTEXT_CHARS = 6000
ANSWER_MAX_TOKENS = 800

ANSWER_SYSTEM = """You answer a listener's spoken question about a research paper they are
part-way through hearing narrated aloud.

Rules:
- Answer from the passage provided. It is the part they have just listened to.
- Speak plainly, in prose meant to be heard, not read. No markdown, no bullet points, no
  headings, no code blocks, no preamble.
- Two to four sentences unless the question genuinely needs more.
- Describe any equation qualitatively. Never read symbols aloud one at a time.
- If the passage does not contain the answer, say so in one sentence and say what part of
  the paper would have it. Never invent a number, a result, or a citation.
- Do not restate the question before answering it."""


def _context_for(paper_id: str, position_s: float) -> tuple[str, str]:
    """The narration around the listener, and the chapter they are in."""
    rows = store.chunks_around(paper_id, position_s)
    if not rows:
        return "", ""
    chapter = next((row["chapter_title"] for row in reversed(rows) if row["chapter_title"]), "")
    passage = " ".join(row["text"] for row in rows)
    if len(passage) > MAX_CONTEXT_CHARS:
        passage = passage[-MAX_CONTEXT_CHARS:]  # keep the most recently heard part
    return passage, chapter


def answer(
    paper_id: str,
    question: str,
    position_s: float,
    history: list[dict] | None = None,
) -> dict:
    """Answer a question, returning the text plus what it was based on."""
    paper = store.get_paper(paper_id)
    if paper is None:
        raise ValueError("no such paper")

    passage, chapter = _context_for(paper_id, position_s)

    if not passage:
        return {
            "text": "I do not have the narration text for this paper yet, so I cannot "
            "answer questions about it. Re-processing it will fix that.",
            "chapter": "",
            "grounded": False,
        }

    messages = [{"role": "system", "content": ANSWER_SYSTEM}]
    for turn in (history or [])[-MAX_HISTORY_TURNS:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append(
        {
            "role": "user",
            "content": (
                f"Paper: {paper['title']}\n"
                f"Section being narrated: {chapter or 'unknown'}\n\n"
                f"The passage they have just heard:\n{passage}\n\n"
                f"Their question: {question}"
            ),
        }
    )

    text = _strip_reasoning(
        llm.chat(messages, max_tokens=ANSWER_MAX_TOKENS, temperature=0.3, thinking=False)
    )
    if not text:
        log.warning("empty answer for %r", question[:60])
    return {
        "text": text or "I could not put an answer together for that one. Try rephrasing it.",
        "chapter": chapter,
        "grounded": True,
    }


_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    """Qwen can emit its reasoning inline; none of it should reach the speaker."""
    return _THINK_BLOCK.sub("", text).strip()

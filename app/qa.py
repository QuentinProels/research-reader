"""Answering a spoken question about the paper being listened to.

Context comes from where the listener actually is. A question during narration is nearly
always about the passage just heard, so the transcript around the current playback offset
answers most of them without any search at all. Step 7 adds a vector fallback for the
rest; until then, an unanswerable question says so rather than inventing something.

Depth is adaptive. Reasoning measured 32s against 1s for the same output on mechanical
work, so switching it on for everything would make every answer feel broken, and off for
everything would make "why does this work" shallow. The classifier below is deliberately
a cheap heuristic rather than another model call: it runs in microseconds, it is testable,
and when it is wrong the cost is an answer that is faster or slower than ideal, not one
that is wrong.
"""

import logging
import re

from app import llm, store

log = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 6
MAX_CONTEXT_CHARS = 6000

# Questions that need the model to actually reason, rather than look something up.
_REASONING_CUES = (
    "why", "how come", "how does", "how do", "how is", "what makes", "explain",
    "justify", "compare", "difference between", "trade-off", "tradeoff", "implication",
    "intuition", "reason", "prove", "derive", "so what", "does that mean", "follow from",
)
# Questions answered by pointing at a fact in the passage just heard.
_LOOKUP_CUES = (
    "what is", "what are", "what was", "what were", "who", "when", "where",
    "how many", "how much", "define", "what does that stand for", "which",
)

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


def needs_reasoning(question: str) -> bool:
    """Should this question get the slow, thinking-enabled path?"""
    text = question.lower().strip()
    if any(cue in text for cue in _REASONING_CUES):
        return True
    if any(text.startswith(cue) for cue in _LOOKUP_CUES):
        return False
    # An unclassified question that is long enough to be involved probably is.
    return len(text.split()) > 12


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
    deep = needs_reasoning(question)

    if not passage:
        return {
            "text": "I do not have the narration text for this paper yet, so I cannot "
            "answer questions about it. Re-processing it will fix that.",
            "chapter": "",
            "reasoned": False,
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

    text = llm.chat(messages, max_tokens=800, temperature=0.3, thinking=deep)
    return {
        "text": _strip_reasoning(text),
        "chapter": chapter,
        "reasoned": deep,
        "grounded": True,
    }


_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    """Qwen can emit its reasoning inline; none of it should reach the speaker."""
    return _THINK_BLOCK.sub("", text).strip()

"""Telling the listener's voice apart from the narration leaking back into the mic.

Hands-free listening in a car has a problem push-to-talk did not: the narration is coming
out of the speakers while the microphone is open, and the narration is speech, so voice
activity detection fires on it constantly. Browser echo cancellation removes most of it,
but not all, and what survives is transcribed as if the listener had said it.

This project can do something a generic canceller cannot. The narration transcript is
stored with timestamps, so at any playback position the exact words being spoken are
known. A transcript that matches what the narration was saying is echo, whatever it
sounded like acoustically. Nothing else has that ground truth.

The comparison is deliberately loose. Speech-to-text on a degraded echo produces
approximate words, so exact matching would never fire; the test is whether the words
overlap enough to be the same passage rather than the same string.
"""

import re

# Fraction of the heard words that must appear in the narration for it to be echo.
# Low enough to catch a mangled echo, high enough that a question quoting a phrase from
# the paper is not mistaken for one.
ECHO_OVERLAP_THRESHOLD = 0.65
MIN_WORDS_TO_JUDGE = 3  # below this there is not enough signal to call it either way
WINDOW_SECONDS = 12.0  # how much narration either side of the trigger to compare against

_WORD = re.compile(r"[a-z0-9]+")
# Words too common to carry evidence either way.
_STOPWORDS = frozenset(
    ["the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those", "for", "on", "with", "as", "at", "by", "from", "we", "our", "they", "their", "he", "she", "you", "i", "not", "no", "can", "could", "would", "should", "do", "does", "did", "has", "have", "had", "will"]
)


def _content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]


def overlap(heard: str, narration: str) -> float:
    """Fraction of the heard content words that also appear in the narration."""
    heard_words = _content_words(heard)
    if not heard_words:
        return 0.0
    narration_words = set(_content_words(narration))
    if not narration_words:
        return 0.0
    matched = sum(1 for word in heard_words if word in narration_words)
    return matched / len(heard_words)


def is_echo(heard: str, narration_nearby: str) -> bool:
    """Was this the narration coming back through the microphone?

    Short utterances are never judged echo: "pause", "go back" and "repeat that" are the
    commands most likely to be said while audio is playing, they are too short to score
    reliably, and suppressing them would break the feature this exists to protect.
    """
    words = _content_words(heard)
    if len(words) < MIN_WORDS_TO_JUDGE:
        return False
    return overlap(heard, narration_nearby) >= ECHO_OVERLAP_THRESHOLD


def narration_near(chunks: list[dict], position_s: float, window_s: float = WINDOW_SECONDS) -> str:
    """The words the narration was speaking around a moment in time."""
    return " ".join(
        chunk["text"]
        for chunk in chunks
        if position_s - window_s <= chunk["start_s"] <= position_s + window_s
    )

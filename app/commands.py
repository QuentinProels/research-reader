"""Deciding whether something the listener said is a command or a question.

Speech-to-text output is lowercase, usually unpunctuated, and normalises numbers its own
way: measured against this project's own STT, "go back ninety seconds" arrives as "go back
90 seconds" and "show me figure three" as "show me figure 3". Both digit and word forms
therefore have to be handled, and neither can be assumed.

The fallback is deliberately "question". Mistaking a command for a question wastes a model
call and produces a slightly odd answer; mistaking a question for a command silently seeks
the audio somewhere the listener never asked to go, and they lose their place. The second
failure is much worse, so matching is conservative and anything ambiguous falls through.
"""

import re
from dataclasses import dataclass

# Vague amounts. Module-level so they are easy to tune after real use.
DEFAULT_BACK_SECONDS = 30.0  # "go back a bit"
DEFAULT_FEW_MINUTES = 180.0  # "a few minutes"
DEFAULT_MOMENT_SECONDS = 15.0  # "a moment", "a second"

KINDS = (
    "back", "forward", "repeat_section", "next_section", "previous_section",
    "show_figure", "pause", "resume", "question",
)

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45,
    "fifty": 50, "sixty": 60, "ninety": 90,
}
_NUMBER_PATTERN = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))

_FILLER = re.compile(r"^\s*(?:um+|uh+|er+|hey|ok(?:ay)?|so|please)\b[,\s]*", re.IGNORECASE)
_TRAILING = re.compile(r"\b(?:please|thanks|thank you)\b[.!?\s]*$", re.IGNORECASE)

# A time quantity: "2 minutes", "thirty seconds", "a minute and a half", "a bit".
_QUANTITY = re.compile(
    rf"(?P<value>\d+(?:\.\d+)?|{_NUMBER_PATTERN})?\s*"
    rf"(?P<unit>minutes?|mins?|seconds?|secs?)(?P<half>\s+and\s+a\s+half)?",
    re.IGNORECASE,
)
_VAGUE = re.compile(r"\b(a\s+bit|a\s+little|a\s+few\s+minutes?|a\s+moment|a\s+second)\b", re.IGNORECASE)

_BACK = re.compile(r"\b(go\s+back|jump\s+back|skip\s+back|rewind|back\s+up|back)\b", re.IGNORECASE)
_FORWARD = re.compile(r"\b(skip\s+(?:ahead|forward)|go\s+forward|jump\s+(?:ahead|forward)|fast\s+forward|forward)\b", re.IGNORECASE)

_REPEAT_SECTION = re.compile(
    r"\b(repeat|replay|play\s+again|say\s+that\s+again|read\s+that\s+again|"
    r"start\s+(?:this|that|the)\s+section\s+over|from\s+the\s+(?:start|beginning)\s+of)\b",
    re.IGNORECASE,
)
_NEXT_SECTION = re.compile(r"\b(next\s+section|skip\s+(?:this|the)\s+section|move\s+on)\b", re.IGNORECASE)
_PREV_SECTION = re.compile(r"\b(previous\s+section|last\s+section|back\s+a\s+section|"
                           r"go\s+back\s+one\s+section|prior\s+section)\b", re.IGNORECASE)

_FIGURE = re.compile(
    rf"\b(?:show|pull\s+up|bring\s+up|display|open|go\s+to)\b[^.?]*?"
    rf"\b(?P<kind>figure|fig|table|tab)\b\.?\s*(?P<num>\d+|{_NUMBER_PATTERN})",
    re.IGNORECASE,
)
_PAUSE = re.compile(r"^\s*(pause|stop|hold\s+on|wait|shush|quiet)\b", re.IGNORECASE)
_RESUME = re.compile(r"^\s*(resume|continue|keep\s+going|carry\s+on|go\s+on|play|unpause)\b", re.IGNORECASE)

# Interrogatives. If a phrase opens with one, it is a question no matter what command-ish
# words appear later: "why did they go back to the original architecture" is not a rewind.
_QUESTION_OPENER = re.compile(
    r"^\s*(?:what|why|how|when|where|who|which|can|could|would|should|does|do|did|is|are|"
    r"was|were|explain|tell\s+me|describe|summari[sz]e)\b",
    re.IGNORECASE,
)


@dataclass
class Command:
    kind: str
    seconds: float | None = None
    figure: str | None = None
    transcript: str = ""


def _to_number(token: str | None) -> float | None:
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return float(_NUMBER_WORDS.get(token.lower(), 0)) or None


def _duration(text: str) -> float | None:
    """Seconds named in the phrase, or None if it names no amount of time.

    Vague phrases are checked first: "a few minutes" contains the unit "minutes" with no
    number attached to it, so the quantity pattern would otherwise read it as one minute.
    """
    vague = _VAGUE.search(text)
    if vague:
        phrase = re.sub(r"\s+", " ", vague.group(1).lower())
        if "few minutes" in phrase:
            return DEFAULT_FEW_MINUTES
        if phrase in ("a moment", "a second"):
            return DEFAULT_MOMENT_SECONDS
        return DEFAULT_BACK_SECONDS

    match = _QUANTITY.search(text)
    if match:
        value = _to_number(match.group("value"))
        if value is None:
            value = 1.0
        if match.group("half"):
            value += 0.5
        unit = match.group("unit").lower()
        return value * (60.0 if unit.startswith(("minute", "min")) else 1.0)
    return None


def _figure_label(match: re.Match) -> str:
    """Normalised to match app.pdf._caption_label, which produces 'Figure 3' / 'Table 2'."""
    kind = "Figure" if match.group("kind").lower().startswith("fig") else "Table"
    number = _to_number(match.group("num"))
    return f"{kind} {int(number) if number and number.is_integer() else match.group('num')}"


def parse(transcript: str) -> Command:
    """Classify one utterance. Anything unclear becomes a question."""
    raw = (transcript or "").strip()
    if not raw:
        return Command(kind="question", transcript=raw)

    text = _TRAILING.sub("", _FILLER.sub("", raw)).strip()
    if not text:
        return Command(kind="question", transcript=raw)

    if _PAUSE.match(text):
        return Command(kind="pause", transcript=raw)
    if _RESUME.match(text):
        return Command(kind="resume", transcript=raw)

    # "show me figure 3" is imperative even though it can be phrased as a question, so it
    # is checked before the interrogative guard. "what does figure 3 look like" is caught
    # too, which is the intent either way.
    figure = _FIGURE.search(text)
    if figure:
        return Command(kind="show_figure", figure=_figure_label(figure), transcript=raw)

    asks_a_question = bool(_QUESTION_OPENER.match(text))
    seconds = _duration(text)

    if _PREV_SECTION.search(text) and not asks_a_question:
        return Command(kind="previous_section", transcript=raw)
    if _NEXT_SECTION.search(text) and not asks_a_question:
        return Command(kind="next_section", transcript=raw)
    if _REPEAT_SECTION.search(text) and not asks_a_question:
        return Command(kind="repeat_section", transcript=raw)

    # A time amount is what separates "go back two minutes" from "go back a section",
    # and its absence is what keeps "why did they go back to that" a question.
    if seconds is not None and not asks_a_question:
        if _FORWARD.search(text):
            return Command(kind="forward", seconds=seconds, transcript=raw)
        if _BACK.search(text):
            return Command(kind="back", seconds=seconds, transcript=raw)

    return Command(kind="question", transcript=raw)

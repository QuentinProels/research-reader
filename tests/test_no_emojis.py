"""No emojis anywhere in this repository. See ai.md.

Deliberately narrower than "no non-ASCII". Typographic characters are fine and are
already in use -- arrows in the README's pipeline diagram, em dashes in prose -- and
app/speech.py has to store the plus-minus sign, the multiplication sign and the Greek
alphabet as literal data, since replacing them is the entire point of that module.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Emoji and pictograph blocks. Excludes U+2190-21FF (arrows) and the mathematical
# operator blocks, which are typography rather than emoji.
EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),  # emoticons, pictographs, transport, symbols, flags
    (0x2600, 0x26FF),  # miscellaneous symbols: sun, umbrella, warning sign
    (0x2700, 0x27BF),  # dingbats: scissors, check marks, sparkles
    (0x2B00, 0x2BFF),  # star, filled squares, emoji-style arrows
    (0xFE00, 0xFE0F),  # variation selectors, which force emoji presentation
    (0x1F1E6, 0x1F1FF),  # regional indicators, which combine into flags
)

SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".wav", ".onnx", ".bin", ".lock"}


def _is_emoji(char: str) -> bool:
    point = ord(char)
    return any(low <= point <= high for low, high in EMOJI_RANGES)


def _tracked_text_files() -> list[Path]:
    listing = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return [
        REPO_ROOT / name
        for name in listing.stdout.splitlines()
        if name and Path(name).suffix.lower() not in SKIP_SUFFIXES
    ]


def test_no_emojis_in_tracked_files():
    offenders = []
    for path in _tracked_text_files():
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue  # binary, or listed but not present in the working tree
        for line_number, line in enumerate(content.splitlines(), start=1):
            found = {char for char in line if _is_emoji(char)}
            if found:
                relative = path.relative_to(REPO_ROOT)
                for char in sorted(found):
                    offenders.append(f"{relative}:{line_number}: U+{ord(char):04X}")
    assert not offenders, "emojis are not allowed in this repository (see ai.md):\n" + "\n".join(
        offenders
    )


def test_no_emojis_in_commit_messages():
    log = subprocess.run(
        ["git", "log", "--format=%H%x1f%s%x1f%b%x1e"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    offenders = []
    for entry in log.stdout.split("\x1e"):
        if not entry.strip():
            continue
        sha, subject, body = (entry.strip().split("\x1f") + ["", ""])[:3]
        found = {char for char in subject + body if _is_emoji(char)}
        if found:
            offenders.append(f"{sha[:8]} {subject[:50]}: " + " ".join(sorted(found)))
    assert not offenders, "emojis are not allowed in commit messages (see ai.md):\n" + "\n".join(
        offenders
    )


class TestTheCheckItself:
    """A rule nothing can violate is a rule nobody is following."""

    # Written as escapes, not literals: this file is itself scanned, so spelling the
    # examples out would make the checker fail on its own test data.
    @pytest.mark.parametrize(
        "char", ["\U0001f600", "\U0001f680", "\u2705", "\u26a0", "\u2b50"]
    )
    def test_emoji_are_detected(self, char):
        assert _is_emoji(char)

    @pytest.mark.parametrize("char", ["→", "←", "—", "±", "×", "α"])
    def test_typography_and_maths_are_allowed(self, char):
        """Arrows, em dashes, plus-minus, multiplication and Greek all appear legitimately
        in the README and in app/speech.py's replacement tables."""
        assert not _is_emoji(char)

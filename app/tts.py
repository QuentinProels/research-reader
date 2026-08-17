"""Kokoro-82M synthesis, CPU (ONNX Runtime).

The GPUs on this box are AMD/ROCm and already hold the 35B; Kokoro is small enough
that CPU on 24 cores keeps up comfortably with long-form rendering.
"""

import re
from pathlib import Path

import numpy as np
import soundfile as sf

from app import speech
from app.config import settings

SAMPLE_RATE = 24000
MAX_CHUNK_CHARS = 500

_session = None


class TTSUnavailable(RuntimeError):
    pass


def _kokoro():
    global _session
    if _session is None:
        if not settings.kokoro_model_path.exists() or not settings.kokoro_voices_path.exists():
            raise TTSUnavailable(
                f"Kokoro weights missing at {settings.kokoro_model_path} / "
                f"{settings.kokoro_voices_path}. Run scripts/fetch_models.sh."
            )
        from kokoro_onnx import Kokoro

        _session = Kokoro(str(settings.kokoro_model_path), str(settings.kokoro_voices_path))
    return _session


def available() -> bool:
    return settings.kokoro_model_path.exists() and settings.kokoro_voices_path.exists()


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split into TTS-sized segments on sentence boundaries, never mid-sentence.

    Normalising first is deliberate: the split below treats every ". " as a sentence
    end, so "e.g. attention" was cut in two and got a 0.35s gap inserted between the
    halves, on top of the one Kokoro already inserted for the period itself.
    """
    sentences = re.split(r"(?<=[.!?])\s+", speech.normalize(text))
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        while len(sentence) > max_chars:  # pathological run-on: hard-split on a space
            split_at = sentence.rfind(" ", 0, max_chars) or max_chars
            chunks.append(sentence[:split_at].strip())
            sentence = sentence[split_at:].strip()
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def synthesize(text: str) -> np.ndarray:
    # Idempotent, so it is safe that chunk_text has usually normalised this already.
    samples, _ = _kokoro().create(
        speech.normalize(text), voice=settings.kokoro_voice, speed=1.0, lang="en-us"
    )
    return np.asarray(samples, dtype=np.float32)


def render(segments: list[str], out_path: Path, on_progress=None) -> list[float]:
    """Render segments to one wav. Returns each segment's start offset in seconds."""
    audio: list[np.ndarray] = []
    offsets: list[float] = []
    cursor = 0.0
    gap = np.zeros(int(SAMPLE_RATE * 0.35), dtype=np.float32)
    for index, segment in enumerate(segments):
        samples = synthesize(segment)
        offsets.append(cursor)
        audio.append(samples)
        audio.append(gap)
        cursor += (len(samples) + len(gap)) / SAMPLE_RATE
        if on_progress:
            on_progress(index + 1, len(segments))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, np.concatenate(audio) if audio else np.zeros(1), SAMPLE_RATE)
    return offsets

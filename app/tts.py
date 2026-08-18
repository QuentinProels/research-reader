"""Kokoro-82M synthesis, CPU (ONNX Runtime).

The GPUs on this box are AMD/ROCm and already hold the 35B; Kokoro is small enough
that CPU on 24 cores keeps up comfortably with long-form rendering.
"""

import logging
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from app import speech
from app.config import settings

log = logging.getLogger(__name__)

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
    samples = np.concatenate(audio) if audio else np.zeros(1)
    sf.write(out_path, samples, SAMPLE_RATE)
    compress(out_path, samples)
    return offsets


def compress(wav_path: Path, samples: np.ndarray | None = None) -> Path | None:
    """Write an Opus copy next to the wav.

    A 30-minute paper is roughly 100MB as raw wav and 7MB as Opus. That is the
    difference between usable and unusable on mobile data, which is the whole point for
    listening in a car. libsndfile does this natively, so no ffmpeg is needed.

    Written to a temporary name and renamed on success. Encoding an hour of audio takes
    minutes, and a process killed part-way through leaves a file that is valid, playable,
    and silently short: one interrupted run produced 19.6 minutes of a 75.6 minute paper,
    and it would have been served as though it were the whole thing.
    """
    ogg_path = wav_path.with_suffix(".ogg")
    partial = ogg_path.with_suffix(".ogg.partial")
    try:
        if samples is None:
            samples, _rate = sf.read(wav_path, dtype="float32")
        sf.write(partial, samples, SAMPLE_RATE, format="OGG", subtype="OPUS")
        partial.replace(ogg_path)
    except BaseException:
        log.warning("could not write %s, falling back to wav", ogg_path.name, exc_info=True)
        partial.unlink(missing_ok=True)
        return None
    return ogg_path

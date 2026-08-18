"""Speech to text for spoken questions, via faster-whisper on CPU.

CPU is not a compromise here, it is the only option: faster-whisper runs on CTranslate2,
which supports CUDA and CPU but not ROCm, and both GPUs on this box are AMD and already
hold the 35B. Push-to-talk clips are a few seconds long, so a small model on 24 cores
keeps well inside a conversational latency budget. Measure before reaching for a larger
one.

The model loads once, on first use, and is reused. Loading it at import would make the
API slow to start for a feature most requests never touch.
"""

import logging
import threading
from pathlib import Path

from app.config import settings

log = logging.getLogger(__name__)

_model = None
_model_name: str | None = None
_lock = threading.Lock()


class STTUnavailable(RuntimeError):
    pass


def _get_model(name: str | None = None):
    """Load, or reload if the chosen model changed. Switching costs one load (~12s),
    which is why it is not done per request."""
    global _model, _model_name
    wanted = name or settings.stt_model
    with _lock:
        if _model is None or _model_name != wanted:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover
                raise STTUnavailable(f"faster-whisper is not installed: {exc}") from exc
            log.info("loading whisper model %s on cpu", wanted)
            _model = WhisperModel(
                wanted,
                device="cpu",
                compute_type="int8",
                download_root=str(settings.models_dir),
            )
            _model_name = wanted
    return _model


def available() -> bool:
    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError:
        return False
    return True


def transcribe(audio_path: Path, model: str | None = None) -> str:
    """Return what was said, or an empty string if the clip held no speech."""
    segments, _info = _get_model(model).transcribe(
        str(audio_path),
        language="en",
        beam_size=1,  # a spoken command is short and unambiguous; beam search buys little
        vad_filter=True,  # drop the silence around a push-to-talk press
        condition_on_previous_text=False,  # each question stands alone
    )
    return " ".join(segment.text.strip() for segment in segments).strip()

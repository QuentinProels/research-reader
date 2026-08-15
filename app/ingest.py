"""Getting a PDF onto local disk, from an upload or a URL.

Both paths converge on a validated local file; nothing downstream knows which
one it came from.
"""

import re
from pathlib import Path

import httpx

from app.config import settings

PDF_MAGIC = b"%PDF-"
ARXIV_ABS_RE = re.compile(r"arxiv\.org/abs/([\w.\-/]+)", re.IGNORECASE)
ARXIV_PDF_RE = re.compile(r"arxiv\.org/pdf/([\w.\-/]+?)(?:v\d+)?(?:\.pdf)?$", re.IGNORECASE)
_UA = {"User-Agent": "research-reader/0.1 (self-hosted; single user)"}


class IngestError(RuntimeError):
    """Carries a message meant to be shown to the user verbatim."""


def check_magic_bytes(head: bytes) -> None:
    """Reject anything that is not actually a PDF, whatever the extension says."""
    if not head.startswith(PDF_MAGIC):
        raise IngestError("That file is not a PDF (wrong magic bytes). Upload the actual PDF.")


def resolve_url(url: str) -> str:
    """Map a paper landing page to something that returns bytes."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise IngestError("Enter a full http(s) URL.")
    abs_match = ARXIV_ABS_RE.search(url)
    if abs_match:
        return f"https://arxiv.org/pdf/{abs_match.group(1)}"
    return url


def fetch(url: str, dest: Path) -> Path:
    """Download a PDF to dest. Raises IngestError with a message worth showing."""
    resolved = resolve_url(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.Client(follow_redirects=True, timeout=60.0, headers=_UA) as client:
            with client.stream("GET", resolved) as response:
                if response.status_code >= 400:
                    raise IngestError(
                        f"Couldn't fetch that link (HTTP {response.status_code}). "
                        "Many publishers gate PDFs behind a login -- please upload the PDF."
                    )
                content_type = response.headers.get("content-type", "")
                if "pdf" not in content_type.lower() and "octet-stream" not in content_type.lower():
                    raise IngestError(
                        f"That link returned {content_type or 'no content type'}, not a PDF. "
                        "It's probably a paywall or a JS landing page -- please upload the PDF."
                    )
                total = 0
                first = True
                with dest.open("wb") as handle:
                    for block in response.iter_bytes(64 * 1024):
                        if first:
                            check_magic_bytes(block[:5])
                            first = False
                        total += len(block)
                        if total > settings.max_upload_bytes:
                            raise IngestError(
                                f"That PDF is over the {settings.max_upload_bytes // 1024 // 1024}MB cap."
                            )
                        handle.write(block)
    except httpx.HTTPError as exc:
        dest.unlink(missing_ok=True)
        raise IngestError(f"Couldn't reach that URL: {exc}. Please upload the PDF instead.") from exc
    except IngestError:
        dest.unlink(missing_ok=True)
        raise
    return dest

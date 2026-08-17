"""Thin client for the llama-server OpenAI-compatible API (Qwen3.6-35B, multimodal)."""

import base64
from pathlib import Path

import httpx

from app.config import settings

_TIMEOUT = httpx.Timeout(300.0, connect=10.0)

REFLOW_SYSTEM = """You clean up text extracted from an academic PDF so it can be read aloud.

Rules:
- Preserve the author's meaning and technical content exactly. Do not summarise or editorialise.
- Fix reading order, join hyphenated line breaks, drop page headers, footers, page numbers,
  and running journal titles.
- Drop the reference list, acknowledgements, and author affiliation blocks.
- Replace inline citation markers ([12], (Smith et al., 2020)) with nothing, or with
  "prior work" where the sentence would otherwise not parse.
- NEVER remove or reword a reference to a figure, table, or equation. Keep "Figure 3",
  "Fig. 3", "Table 2" exactly as written, including when they appear in parentheses:
  "... attention (Figure 2)" must keep "Figure 2". These are not citations. Each one is
  the anchor where a spoken description of that figure gets inserted, so dropping one
  silently moves the description to the end of the recording, away from the text it
  explains. Rephrasing "Table 1" to "the corresponding table" breaks it just as badly.
- Describe equations qualitatively, in words a listener can follow. Never read symbols aloud
  one by one. "The loss is a weighted sum of a reconstruction error and a KL divergence term"
  is right; "L equals alpha times x sub i" is wrong.
- Return only the cleaned prose. No preamble, no markdown headings, no commentary."""

FIGURE_SYSTEM = """You describe a figure or table from an academic paper for a listener who
cannot see it. They are listening to the paper as audio.

Rules:
- Lead with what the figure shows and what it is for, then the trend or result that matters.
- Give concrete numbers only where they carry the point. Do not read off every data point.
- Describe axes and conditions in words, briefly, only if needed to make the trend meaningful.
- For tables, state what is being compared and which entry wins by how much.
- Never read equations symbol by symbol; describe them qualitatively.
- Two to five sentences. Plain spoken prose, no markdown, no bullet points, no preamble."""


class LLMError(RuntimeError):
    """The model server answered, but not usefully. Callers may degrade gracefully."""


class LLMUnavailable(LLMError):
    """The model server could not be reached at all.

    Distinct from LLMError because the right response is different: a single bad
    response can be worked around by falling back to raw text, but an unreachable
    server means every remaining call in the paper will fail too, and narrating a whole
    paper from unreflowed text is worse than waiting. The worker requeues these.
    """


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    return headers


def _chat(
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.3,
    thinking: bool = False,
) -> str:
    # llama-server here runs with --reasoning-budget -1, so Qwen thinks at length by
    # default. On reflow that measured 32.4s / 1955 tokens versus 1.0s / 43 tokens with
    # thinking off, for byte-identical output: these are mechanical rewriting tasks with
    # nothing to reason about. Left as a parameter because Q&A (step 6) will want it on.
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            response = client.post(
                f"{settings.llm_base_url}/chat/completions", json=payload, headers=_headers()
            )
    except httpx.HTTPError as exc:
        # Connection refused, disconnects mid-response, timeouts: the server is down or
        # restarting. Distinguish it so the job can be retried rather than failed.
        raise LLMUnavailable(f"cannot reach llama-server at {settings.llm_base_url}: {exc}") from exc
    if response.status_code in (502, 503, 504):
        raise LLMUnavailable(f"llama-server returned {response.status_code}")
    if response.status_code != 200:
        raise LLMError(f"llama-server returned {response.status_code}: {response.text[:300]}")
    return response.json()["choices"][0]["message"]["content"].strip()


def chat(
    messages: list[dict], max_tokens: int = 2048, temperature: float = 0.3, thinking: bool = False
) -> str:
    """Public entry point for callers that build their own message list."""
    return _chat(messages, max_tokens=max_tokens, temperature=temperature, thinking=thinking)


def health() -> tuple[bool, str]:
    """Is llama-server reachable and is our API key accepted?"""
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            response = client.get(f"{settings.llm_base_url}/models", headers=_headers())
        if response.status_code == 200:
            return True, settings.llm_model
        return False, f"HTTP {response.status_code}"
    except httpx.HTTPError as exc:
        return False, str(exc)


def reflow(text: str) -> str:
    """Clean one chunk of extracted text into listenable prose."""
    return _chat(
        [
            {"role": "system", "content": REFLOW_SYSTEM},
            {"role": "user", "content": text},
        ],
        max_tokens=4096,
    )


def caption_figure(image_path: Path, nearby_caption: str = "") -> str:
    """Describe a figure image qualitatively, for listening."""
    image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    hint = (
        f"The caption printed under it reads: {nearby_caption}\n\n"
        if nearby_caption.strip()
        else ""
    )
    return _chat(
        [
            {"role": "system", "content": FIGURE_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{hint}Describe this figure for a listener."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ],
        max_tokens=512,
    )

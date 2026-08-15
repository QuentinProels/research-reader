# Research Reader

Self-hosted. Feed it a research paper, get back narrated audio you can listen to like an
audiobook — including spoken descriptions of the figures and tables, dropped in at the
point in the text where the paper first refers to them.

Equations are described qualitatively ("the loss is a weighted sum of a reconstruction
error and a KL divergence term"), never read symbol by symbol. Read aloud, symbols are
unparseable by ear.

## Status

v0.1 — build-order steps 1–5 of 7. Ingest, parse, caption, reflow, render, play.
Voice Q&A (steps 6–7) is not built yet.

| Step | | |
|---|---|---|
| 1 | PDF parse + text extraction | done, needs corpus iteration |
| 2 | LLM reflow / cleanup pass | done |
| 3 | Figure detection + qualitative captioning | done |
| 4 | TTS render + chapter markers | done |
| 5 | Web player (seek, speed, resume, figure display) | done |
| 6 | Voice Q&A (STT → context → LLM → TTS) | not started |
| 7 | Q&A fallback: vector search over the full paper | not started |

## How it runs

The model server is not part of this repo. It expects an OpenAI-compatible endpoint
already serving a multimodal model — here that's `llama-server` with
`Qwen3.6-35B-A3B-UD-Q6_K_XL` plus `mmproj-F16.gguf` for vision, on `:8084`.

```
PDF upload or URL
  → PyMuPDF layout extraction (text blocks, reading order, figures, captions)
  → figure crops → vision LLM → spoken descriptions
  → text LLM → reading order, drop headers/footers/references, qualitative equations
  → captions inserted at their in-text reference points
  → ~500-char segments → Kokoro-82M → one wav + chapter and figure timestamps
```

## Setup

```bash
uv sync                       # python deps
./scripts/fetch_models.sh     # Kokoro weights, ~350MB, once
cp .env.example .env          # then fill in LLM_API_KEY and APP_PASSWORD_HASH
uv run uvicorn app.main:app --host 127.0.0.1 --port 8090
```

Generate the password hash:

```bash
uv run python -c "import bcrypt,getpass;print(bcrypt.hashpw(getpass.getpass().encode(),bcrypt.gensalt()).decode())"
```

Tests: `uv run pytest`

## Hardware note

This runs on AMD/ROCm (2× 34GB), not CUDA. That rules out some of the obvious picks —
notably `faster-whisper`, whose CTranslate2 backend is CUDA-or-CPU only. When step 6
lands, STT should be `whisper.cpp` with a HIP or Vulkan build, or faster-whisper pinned
to CPU. Kokoro runs on CPU by design; the GPUs are full of the 35B.

## Access

Exposed through a Cloudflare Tunnel — no inbound ports. Cloudflare Access (Google auth)
is configured in the Cloudflare dashboard, not here. The app additionally checks a single
shared password (bcrypt, `APP_PASSWORD_HASH` in `.env`) as a second layer in case the
hostname leaks or Access is misconfigured. That is insurance, not a replacement.

Upload hardening, since the tunnel is public-facing: magic-byte check before anything
touches the file, a size cap, a page-count cap to catch decompression bombs, and a parse
timeout.

## Deliberate v0 shortcuts

- **SQLite, not Postgres.** The schema is small and single-user, and nothing needs a
  vector column until step 7. `app/store.py` is the only file that touches SQL, so the
  swap to Postgres + pgvector stays a one-file change.
- **No job queue.** One paper at a time in a worker thread, as specced. Add Celery only
  if batch ingest ever becomes a real workflow.
- **App runs on the host, not in a container.** It calls a host-local model server and
  needs the host's CPU for TTS; containerising it buys nothing yet.

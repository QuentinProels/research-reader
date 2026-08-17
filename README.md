# Research Reader

Self-hosted. Feed it a research paper, get back narrated audio you can listen to like an
audiobook — including spoken descriptions of the figures and tables, dropped in at the
point in the text where the paper first refers to them.

Equations are described qualitatively ("the loss is a weighted sum of a reconstruction
error and a KL divergence term"), never read symbol by symbol. Read aloud, symbols are
unparseable by ear.

Working on this with an AI agent? Read [ai.md](ai.md) first: it covers the conventions
that are not obvious from the code, and the ways this machine differs from what the
original spec assumed.

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
cp .env.example .env          # then fill in LLM_API_KEY and DATABASE_URL

mkdir -p secrets && chmod 700 secrets
head -c 18 /dev/urandom | base64 | tr -d '=+/' > secrets/pg_password   # then paste it into DATABASE_URL
uv run python -c "import bcrypt,getpass;print(bcrypt.hashpw(getpass.getpass().encode(),bcrypt.gensalt()).decode())" > secrets/app_password.hash

docker compose up -d          # postgres + pgvector on 127.0.0.1:5433
uv run uvicorn app.main:app --host 127.0.0.1 --port 8090   # terminal 1: API
uv run python -m app.worker                                # terminal 2: queue worker
```

Both are needed. The API only enqueues; without a worker running, papers sit in
`queued` forever — the UI says so rather than spinning silently.

Tests: `uv run pytest`

### Why secrets sit in files, not `.env`

`docker compose` auto-loads `.env` and interpolates `$VAR` inside *every* value it finds
there, whether or not the compose file references it. A bcrypt hash is full of `$`, so
storing it in `.env` gets it silently mangled. Both secrets therefore live in `./secrets/`
(gitignored, `chmod 700`): compose mounts `pg_password` as a docker secret, and the app
reads `app_password.hash` directly. `.env` holds paths and non-`$` config only.

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

## The queue

Papers are processed by a separate worker process, not by the API. Queue state lives in
Postgres; there is no Redis and no Celery. Jobs are minutes long and rare, the database
was already here, and `SELECT … FOR UPDATE SKIP LOCKED` gives correct multi-worker
claiming in a single statement. Adding a broker would have meant two more services to
solve a problem that does not exist at single-user scale.

Why a separate process rather than a thread in the API: when the pipeline ran in an API
thread, restarting the API killed the job mid-flight and left the row frozen at its last
progress value with nothing working on it and no error — indistinguishable, from the UI,
from a job still running. Now the API can be restarted or crash while a paper keeps
rendering.

- **Crash recovery is lease-based.** Workers stamp `heartbeat_at` as they go. Anything
  non-terminal whose heartbeat is older than `LEASE_SECONDS` (5 min) is assumed dead and
  requeued, up to `MAX_ATTEMPTS` (3), after which it is failed with a real message.
- **Shutdown is two-stage**, as in most job runners. First signal: finish the current
  paper, then exit. Second: hand the job back to the queue and exit now.
- **Restarted jobs start over.** There is no mid-pipeline checkpoint, so a requeued paper
  re-parses and re-renders from the beginning. Worth adding only if interruptions become
  common.
- **Parallelism is by process.** Run more than one worker if you want it; claiming is
  atomic. On this box one is usually right — llama-server serves a single model and
  Kokoro is on CPU, so two workers mostly contend rather than go faster.

## Deliberate v0 shortcuts
- **App runs on the host, not in a container.** It calls a host-local model server and
  needs the host's CPU for TTS; containerising it buys nothing yet.

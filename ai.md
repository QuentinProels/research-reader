# Working on Research Reader

Instructions for AI agents. Read this before changing anything.

## What this is

Self-hosted. A research paper goes in as a PDF, narrated audio comes out, and the figures
and tables are described aloud by a vision model at the point in the text where the paper
first refers to them. Equations are described qualitatively, never read symbol by symbol,
because symbols read aloud are unparseable by ear.

The pipeline:

```
PDF upload or URL
  -> PyMuPDF layout extraction (text blocks, reading order, figures, captions)
  -> figure crops to the vision LLM, which returns spoken descriptions
  -> text LLM reflows prose, drops headers/footers/references, describes equations
  -> descriptions inserted at their in-text reference points
  -> normalise for speech, chunk to ~500 chars, Kokoro-82M
  -> one wav plus chapter and figure timestamps
```

Papers are queued in Postgres and processed by a separate worker process. The API only
enqueues. Both must be running.

## Build order, and where it stopped

Steps 1-5 are done: ingest, parse, caption, reflow, render, web player. Steps 6 and 7 are
not started: voice Q&A (STT to context to LLM to TTS, push-to-talk) and the pgvector
fallback search over the full paper.

## The machine this runs on

Its spec was written assuming NVIDIA and is wrong about this box. The gap only shows up at
runtime, so check rather than assume:

- **AMD/ROCm, two cards, 34GB each.** `nvidia-smi` fails here; use `rocm-smi`.
- **llama-server is on :8084**, serving `Qwen3.6-35B-A3B-UD-Q6_K_XL` with `mmproj-F16.gguf`
  for vision. It needs an API key. It gets restarted occasionally, which is why the queue
  has outage retry.
- It runs with `--reasoning-budget -1`, so Qwen reasons at length unless you pass
  `chat_template_kwargs: {enable_thinking: false}`. On mechanical rewriting that measured
  32x faster for byte-identical output. `app/llm.py` already does this; keep it that way
  for anything that is not genuinely a reasoning task.
- **Host Postgres already owns :5432.** This project's Postgres is on 5433.
- Kokoro runs on CPU by design. The GPUs are full of the 35B.

Before starting step 6: `faster-whisper` is named in the spec, but its CTranslate2 backend
is CUDA-or-CPU only and gets no acceleration here. Use `whisper.cpp` with a HIP or Vulkan
build, or pin faster-whisper to CPU. Decide before implementing, not halfway through.

## Branching and pull requests

- **Small, self-contained fix** -- one module, obvious, covered by existing tests: commit
  to `main` and say what you pushed.
- **Structural change** -- a new service, a schema migration, a cross-module refactor, or
  anything that changes the shape of the pipeline: branch, push, open a PR with `gh pr
  create`, hand over the link. Do not merge it yourself.
- **Genuinely ambiguous**: ask before starting, not after pushing.

Steps 6 and 7 are both structural. So is anything touching the queue schema.

## No emojis

No emojis anywhere in this repository. Not in code, comments, docstrings, log lines, UI
strings, test names, commit messages, PR descriptions, or documentation.

`tests/test_no_emojis.py` enforces this against every tracked file and will fail the
build. It permits typographic characters that are not emojis -- arrows, em dashes,
mathematical symbols -- and it has to, because `app/speech.py` stores `+/-`, the
multiplication sign and the Greek alphabet as data it exists to replace.

## Conventions that are not obvious

**Tests that touch Postgres use a throwaway database.** `tests/conftest.py` redirects
`DATABASE_URL` to `reader_test` before `app.config` is imported, and the queue fixtures
refuse to run against any database whose name does not end in `_test`. This is not
belt-and-braces: an earlier version truncated the live database and deleted queued papers
out from under a running worker.

**Secrets live in `./secrets/`, not `.env`.** `docker compose` auto-loads `.env` and
interpolates `$VAR` inside every value it finds there, whether or not the compose file
references it, so a bcrypt hash stored in `.env` gets silently mangled. Compose mounts
`secrets/pg_password` as a docker secret; the app reads `secrets/app_password.hash` by
path.

**All SQL lives in `app/store.py`.** Keep it that way.

**`LLMUnavailable` is not `LLMError`.** A bad response can be worked around by falling
back to raw text. An unreachable server means every remaining call will fail too, and
narrating a whole paper from unreflowed text is worse than waiting, so the worker requeues
with backoff instead. Do not collapse the two.

**Normalise text before chunking, not just before synthesis.** `chunk_text` splits on
`". "`, so an unnormalised `e.g.` gets cut into two chunks and picks up a 0.35s gap of our
own making on top of the one Kokoro inserts.

## Verify against real inputs, not just fixtures

Every significant bug in this project's history was found by running a real paper through
and looking at the output, and would have passed a green test suite:

- Figure descriptions were narrated once per section instead of once per paper, turning a
  15-page paper into 74 minutes of audio.
- The reflow prompt deleted every figure reference, sending all descriptions to the end of
  the recording.
- Caption-to-figure pairing was by index, so a multi-panel figure captured one panel and
  all four tables were dropped entirely.
- A worker busy rendering reported itself dead, so the UI warned the queue would not drain
  at exactly the moment it was draining.

Measure rather than reason about audio. Spelling numbers as words looked like an obvious
fix for a stutter and changed nothing measurable; the pauses that remained turned out to
be normal breath prosody that scales with sentence length.

## Commands

```bash
uv sync
./scripts/fetch_models.sh                                  # Kokoro weights, once
docker compose up -d                                       # Postgres + pgvector on 5433
uv run uvicorn app.main:app --host 127.0.0.1 --port 8090   # API
uv run python -m app.worker                                # queue worker
uv run pytest
uv run ruff check .
```

Exposed at `reader.quentinlab.co` through a Cloudflare Tunnel, whose config is root-owned
at `/etc/cloudflared/config.yml` and needs sudo the agent does not have. Prepare the diff
and hand over the commands rather than trying to apply them.

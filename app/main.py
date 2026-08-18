import json
import shutil
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import auth, commands, echo, ingest, llm, qa, store, stt, tts
from app.config import settings

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Research Reader")
store.init_db()

PUBLIC_PATHS = {"/login", "/healthz", "/static/login.css"}


@app.middleware("http")
async def require_password(request: Request, call_next):
    path = request.url.path
    if path in PUBLIC_PATHS or not auth.enabled() or auth.is_authenticated(request):
        return await call_next(request)
    if path.startswith("/api/"):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return RedirectResponse("/login", status_code=303)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/login")
def login_page():
    return FileResponse(WEB_DIR / "login.html")


@app.post("/login")
def login(request: Request, password: str = Form(...)):
    if not auth.verify_password(password):
        return RedirectResponse("/login?error=1", status_code=303)
    # Secure only over TLS: through the tunnel this is https, but a browser will
    # silently drop a Secure cookie on a plain-http localhost visit, which would
    # make login look broken for anyone testing on the box itself.
    forwarded = request.headers.get("x-forwarded-proto", request.url.scheme)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        auth.COOKIE_NAME,
        auth.issue_cookie(),
        max_age=auth.MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=forwarded == "https",
    )
    return response


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
def api_status():
    llm_ok, llm_detail = llm.health()
    workers = store.live_workers()
    return {
        "llm": {"ok": llm_ok, "detail": llm_detail},
        "tts": {"ok": tts.available(), "detail": "kokoro" if tts.available() else "weights missing"},
        # Without this the failure mode is silent: papers queue up and nothing ever
        # touches them, which from the UI is indistinguishable from "still working".
        "stt": {
            "ok": stt.available(),
            "detail": settings.stt_model if stt.available() else "faster-whisper missing",
        },
        "worker": {
            "ok": workers > 0,
            "detail": f"{workers} running" if workers else "none — queue will not drain",
        },
    }


@app.get("/api/settings")
def api_get_settings():
    return store.get_settings()


@app.post("/api/settings")
async def api_save_settings(request: Request):
    return store.save_settings(await request.json())


@app.get("/api/papers")
def api_papers():
    return store.list_papers()


@app.get("/api/papers/{paper_id}")
def api_paper(paper_id: str):
    paper = store.get_paper(paper_id)
    if not paper:
        raise HTTPException(404, "no such paper")
    return paper


@app.post("/api/papers/{paper_id}/resume")
def api_resume(paper_id: str, position: float = Form(...)):
    store.update_paper(paper_id, resume_s=max(position, 0.0))
    return {"ok": True}


@app.get("/api/papers/{paper_id}/audio")
def api_audio(paper_id: str):
    """Prefer the Opus version. A 30-minute paper is about 100MB as raw wav and about
    7MB as Opus, which is the difference between usable and not on mobile data."""
    paper = store.get_paper(paper_id)
    if not paper or not paper.get("audio_path"):
        raise HTTPException(404, "no audio yet")
    wav = settings.data_dir / paper["audio_path"]
    ogg = wav.with_suffix(".ogg")
    if ogg.exists():
        return FileResponse(ogg, media_type="audio/ogg")
    return FileResponse(wav, media_type="audio/wav")


@app.post("/api/upload")
async def api_upload(file: UploadFile):
    head = await file.read(5)
    try:
        ingest.check_magic_bytes(head)
    except ingest.IngestError as exc:
        raise HTTPException(400, str(exc)) from exc

    paper_id = store.create_paper(title=file.filename or "Untitled", source="upload")
    pdf_path = settings.papers_dir / paper_id / "source.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    written = len(head)
    with pdf_path.open("wb") as handle:
        handle.write(head)
        while block := await file.read(1024 * 1024):
            written += len(block)
            if written > settings.max_upload_bytes:
                handle.close()
                shutil.rmtree(pdf_path.parent, ignore_errors=True)
                raise HTTPException(
                    413, f"Over the {settings.max_upload_bytes // 1024 // 1024}MB upload cap."
                )
            handle.write(block)

    _enqueue(paper_id, pdf_path)
    return {"id": paper_id}


@app.post("/api/fetch")
def api_fetch(url: str = Form(...)):
    paper_id = store.create_paper(title=url, source=url)
    pdf_path = settings.papers_dir / paper_id / "source.pdf"
    try:
        ingest.fetch(url, pdf_path)
    except ingest.IngestError as exc:
        store.update_paper(paper_id, status="failed", error=str(exc))
        raise HTTPException(400, str(exc)) from exc
    _enqueue(paper_id, pdf_path)
    return {"id": paper_id}


@app.post("/api/papers/{paper_id}/listen")
async def api_listen(
    paper_id: str,
    audio: UploadFile,
    position: float = Form(0.0),
    history: str = Form("[]"),
    playing: bool = Form(False),
):
    """One push-to-talk utterance: transcribe it, then either act on it or answer it.

    Navigation is resolved here rather than in the browser because the chapter
    boundaries live in the database, and returning a concrete seek target keeps the
    player dumb.
    """
    paper = store.get_paper(paper_id)
    if not paper:
        raise HTTPException(404, "no such paper")

    clip = settings.papers_dir / paper_id / "question.wav"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(await audio.read())

    prefs = store.get_settings()
    try:
        transcript = stt.transcribe(clip, model=prefs["stt_model"])
    except stt.STTUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    finally:
        clip.unlink(missing_ok=True)

    if not transcript.strip():
        return {"transcript": "", "action": "none", "say": "I did not catch that."}

    # Hands-free listening hears the narration through the speakers. The transcript says
    # exactly what was being narrated at this moment, so an utterance that matches it is
    # echo however convincing it sounded. Silently ignored: announcing it would mean the
    # app talking back at every sentence it reads.
    if playing and prefs["echo_rejection"]:
        nearby = echo.narration_near(store.chunks_around(paper_id, position, 30.0), position)
        if echo.is_echo(transcript, nearby):
            return {"transcript": transcript, "kind": "echo", "action": "none", "say": ""}

    command = commands.parse(transcript)
    reply: dict = {"transcript": transcript, "kind": command.kind}

    if command.kind in ("back", "forward"):
        delta = command.seconds * (1 if command.kind == "forward" else -1)
        reply |= {"action": "seek", "position": max(position + delta, 0.0),
                  "say": f"{'Forward' if delta > 0 else 'Back'} {int(abs(delta))} seconds."}
    elif command.kind == "repeat_section":
        chapter = store.chapter_at(paper_id, position)
        reply |= {"action": "seek", "position": (chapter or {}).get("start_s", 0.0),
                  "say": f"Repeating {(chapter or {}).get('title', 'this section')}."}
    elif command.kind in ("next_section", "previous_section"):
        step = 1 if command.kind == "next_section" else -1
        chapter = store.adjacent_chapter(paper_id, position, step)
        if chapter is None:
            reply |= {"action": "none",
                      "say": "That is the last section." if step > 0 else "This is the first section."}
        else:
            reply |= {"action": "seek", "position": chapter["start_s"],
                      "say": chapter["title"]}
    elif command.kind == "show_figure":
        match = next(
            (f for f in paper["figures"] if f["label"].lower() == (command.figure or "").lower()),
            None,
        )
        if match is None:
            reply |= {"action": "none", "say": f"I could not find {command.figure}."}
        else:
            reply |= {"action": "show_figure", "figure": match,
                      "say": match.get("caption") or match["label"]}
    elif command.kind in ("pause", "resume"):
        reply |= {"action": command.kind, "say": ""}
    else:
        try:
            history_turns = json.loads(history)
        except json.JSONDecodeError:
            history_turns = []
        answered = qa.answer(
            paper_id, transcript, position, history_turns,
            thorough=prefs["answer_depth"] == "thorough",
        )
        reply |= {"action": "answer", "say": answered["text"],
                  "chapter": answered["chapter"]}
    return reply


@app.post("/api/speak")
def api_speak(text: str = Form(...)):
    """Render a short spoken reply. Kept separate so the browser can show the text the
    instant it arrives and start the audio only when it is ready."""
    if not tts.available():
        raise HTTPException(503, "TTS weights are missing")
    clip = settings.data_dir / "replies"
    clip.mkdir(parents=True, exist_ok=True)
    path = clip / f"{abs(hash(text)) % (10**12)}.wav"
    if not path.exists():
        tts.render(tts.chunk_text(text), path)
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/papers/{paper_id}/cancel")
def api_cancel(paper_id: str):
    if not store.cancel_paper(paper_id):
        raise HTTPException(409, "Only a paper still waiting in the queue can be cancelled.")
    return {"ok": True}


@app.delete("/api/papers/{paper_id}")
def api_delete(paper_id: str):
    if not store.delete_paper(paper_id):
        raise HTTPException(404, "no such paper")
    shutil.rmtree(settings.papers_dir / paper_id, ignore_errors=True)
    return {"ok": True}


def _enqueue(paper_id: str, pdf_path: Path) -> None:
    """Record where the PDF landed and leave it 'queued'.

    The API never runs the pipeline itself -- a separate worker process claims the row.
    That is the whole point: restarting the API no longer kills work in flight.
    """
    store.update_paper(paper_id, pdf_path=str(pdf_path.relative_to(settings.data_dir)))


app.mount("/figures", StaticFiles(directory=settings.papers_dir), name="figures")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

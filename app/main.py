import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import auth, ingest, llm, pipeline, store, tts
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
def login(password: str = Form(...)):
    if not auth.verify_password(password):
        return RedirectResponse("/login?error=1", status_code=303)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        auth.COOKIE_NAME,
        auth.issue_cookie(),
        max_age=auth.MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=True,
    )
    return response


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
def api_status():
    llm_ok, llm_detail = llm.health()
    return {
        "llm": {"ok": llm_ok, "detail": llm_detail},
        "tts": {"ok": tts.available(), "detail": "kokoro" if tts.available() else "weights missing"},
    }


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
    paper = store.get_paper(paper_id)
    if not paper or not paper.get("audio_path"):
        raise HTTPException(404, "no audio yet")
    return FileResponse(settings.data_dir / paper["audio_path"], media_type="audio/wav")


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

    _start(paper_id, pdf_path)
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
    _start(paper_id, pdf_path)
    return {"id": paper_id}


def _start(paper_id: str, pdf_path: Path) -> None:
    """One paper at a time, in a worker thread. No queue in v1 -- see spec section 3."""
    threading.Thread(
        target=pipeline.run, args=(paper_id, pdf_path), daemon=True, name=f"pipeline-{paper_id}"
    ).start()


app.mount("/figures", StaticFiles(directory=settings.papers_dir), name="figures")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

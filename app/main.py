"""ReSynth web server: one page, a small JSON API, and the job runner.

Start:  python -m app.main            (http://127.0.0.1:8000, this computer only)
        python -m app.main --lan      (listen on all interfaces so phones/laptops on the LAN can open it)
Settings (environment variables):
  RESYNTH_ANTHROPIC_KEY   Claude API key for research, plan and report prose (optional)
  RESYNTH_PORT            port (default 8000)
  RESYNTH_HOST            bind address (default 127.0.0.1; --lan sets 0.0.0.0)
  RESYNTH_MATCH_BUDGET    seconds the match search may take (default 180)
  RESYNTH_PASSWORD        if set, the page asks for this password (HTTP basic auth, any user name);
                          set it before exposing the app through a tunnel
  RESYNTH_JOBS            jobs folder (default ./jobs)
  RESYNTH_OPEN_BROWSER    1 = open the page in the default browser once the server is up
"""
from __future__ import annotations

import argparse
import base64
import hmac
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import jobs
from app.stages.claude import KEY_ENV, key_present
from app.stages.common import read_json

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
REGION_MIN_S, REGION_MAX_S = 1.0, 3.0
SERVED = re.compile(r"^(upload\.wav|slice\.wav|target\.wav|match_1\.wav|ab\.wav|stems/[a-z0-9_]+\.wav|"
                    r"patch(_alt[12])?\.vital|report\.(md|html|json))$")

runner = jobs.Runner()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    runner.stop_all()  # do not leave phase processes running after the server stops


app = FastAPI(title="ReSynth", docs_url=None, redoc_url=None, lifespan=lifespan)


def match_budget() -> float:
    try:
        return max(5.0, float(os.environ.get("RESYNTH_MATCH_BUDGET", "180")))
    except ValueError:
        return 180.0


@app.middleware("http")
async def password_gate(request: Request, call_next):
    pw = os.environ.get("RESYNTH_PASSWORD", "")
    if pw:
        ok = False
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("basic "):
            try:
                _, _, given = base64.b64decode(auth[6:]).decode("utf-8").partition(":")
                ok = hmac.compare_digest(given.encode(), pw.encode())
            except ValueError:
                ok = False
        if not ok:
            return Response("Password required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="ReSynth"'})
    return await call_next(request)


def _job_or_404(job_id: str) -> tuple[Path, dict[str, Any]]:
    try:
        d = jobs.job_dir(job_id)
    except (FileNotFoundError, jobs.JobError):
        raise HTTPException(404, "no such job")
    return d, jobs.load_job(d)


def job_state(d: Path, job: dict[str, Any]) -> dict[str, Any]:
    job = jobs.reconcile(d, job, runner)
    statuses = jobs.read_statuses(d)
    stems = None
    if (d / "isolate.json").exists():
        iso = read_json(d / "isolate.json")
        stems = {"status": iso.get("status"), "message": iso.get("message"), "suggested": iso.get("suggested"),
                 "names": ["raw", *iso.get("stems", {}).keys()], "rms": iso.get("rms", {}),
                 "chosen": iso.get("chosen")}
    research = None
    if (d / "research.json").exists() and statuses.get("research", {}).get("state") != "running":
        r = read_json(d / "research.json")
        research = {"status": r.get("status"), "reason": r.get("reason")}
    return {"job": job, "stages": statuses, "stems": stems, "research": research,
            "report": (d / "report.json").exists(), "busy_with": runner.busy()}


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {"key_present": key_present(), "key_env": KEY_ENV, "ffmpeg": bool(jobs.ffmpeg_path()),
            "max_clip_s": jobs.MAX_CLIP_S, "region_s": [REGION_MIN_S, REGION_MAX_S], "budget_s": match_budget()}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), song: str = Form(""), artist: str = Form(""),
                     description: str = Form(""), gear: str = Form("")) -> dict[str, Any]:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "the file is too large (limit 80 MB); upload a clip of at most 60 seconds")
    if not data:
        raise HTTPException(400, "the file is empty")
    try:
        job = jobs.create_job(file.filename or "clip", data, song[:200], artist[:200], description[:1000],
                              gear[:1000], match_budget())
    except jobs.JobError as e:
        raise HTTPException(400, str(e))
    d = jobs.job_dir(job["id"])
    runner.start(d, "research")
    return job_state(d, job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    d, job = _job_or_404(job_id)
    return job_state(d, job)


class Region(BaseModel):
    start: float
    end: float


@app.post("/api/jobs/{job_id}/region")
def set_region(job_id: str, region: Region) -> dict[str, Any]:
    d, job = _job_or_404(job_id)
    length = region.end - region.start
    if not (0 <= region.start < region.end <= job["duration_s"] + 1e-3):
        raise HTTPException(400, "the region is outside the clip")
    if not (REGION_MIN_S - 0.02 <= length <= REGION_MAX_S + 0.02):
        raise HTTPException(400, f"mark between {REGION_MIN_S:.0f} and {REGION_MAX_S:.0f} seconds (got {length:.1f} s)")
    busy = runner.busy()
    if busy:
        raise HTTPException(409, "another job is running; wait for it to finish")
    if job["phase"] not in ("uploaded", "choose_stem", "failed"):
        raise HTTPException(409, f"this job is already {job['phase']}")
    for name in ("isolate.json", "target.wav"):
        (d / name).unlink(missing_ok=True)
    for s in ("isolate", "analyse", "plan", "match", "report"):
        (d / "status" / f"{s}.json").unlink(missing_ok=True)
    job.update(region=[round(region.start, 3), round(region.end, 3)], stem=None, phase="isolating", error=None)
    jobs.save_job(d, job)
    runner.start(d, "isolate")
    return job_state(d, job)


class Stem(BaseModel):
    stem: str


@app.post("/api/jobs/{job_id}/stem")
def choose_stem(job_id: str, choice: Stem) -> dict[str, Any]:
    d, job = _job_or_404(job_id)
    if job["phase"] != "choose_stem":
        raise HTTPException(409, "stems are not ready to choose from")
    iso = read_json(d / "isolate.json")
    if choice.stem != "raw" and choice.stem not in iso.get("stems", {}):
        raise HTTPException(400, "unknown stem")
    if runner.busy():
        raise HTTPException(409, "another job is running; wait for it to finish")
    job.update(stem=choice.stem, phase="processing", error=None)
    jobs.save_job(d, job)
    runner.start(d, "process")
    return job_state(d, job)


@app.get("/api/jobs/{job_id}/report")
def get_report(job_id: str) -> JSONResponse:
    d, _ = _job_or_404(job_id)
    if not (d / "report.json").exists():
        raise HTTPException(404, "no report yet")
    return JSONResponse(read_json(d / "report.json"))


@app.get("/api/jobs/{job_id}/files/{name:path}")
def get_file(job_id: str, name: str) -> FileResponse:
    d, _ = _job_or_404(job_id)
    if not SERVED.match(name):
        raise HTTPException(404, "not available")
    path = (d / name).resolve()
    if d.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "not available")
    if path.suffix in (".vital", ".md"):
        stem = "resynth_" + (path.stem if path.suffix == ".vital" else "report")
        return FileResponse(path, filename=f"{stem}{path.suffix}", media_type="application/octet-stream")
    media = {".wav": "audio/wav", ".html": "text/html", ".json": "application/json"}[path.suffix]
    return FileResponse(path, media_type=media)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", media_type="text/html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="ReSynth web app")
    ap.add_argument("--lan", action="store_true", help="listen on all network interfaces (other devices can connect)")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    a = ap.parse_args()
    host = a.host or ("0.0.0.0" if a.lan else os.environ.get("RESYNTH_HOST", "127.0.0.1"))
    port = a.port or int(os.environ.get("RESYNTH_PORT", "8000"))
    print(f"ReSynth: open http://{'127.0.0.1' if host in ('0.0.0.0', '127.0.0.1') else host}:{port} in your browser")
    if host == "0.0.0.0" and not os.environ.get("RESYNTH_PASSWORD"):
        print("Warning: listening on the network without RESYNTH_PASSWORD; anyone on this network can use the app.")
    if not key_present():
        print(f"Note: {KEY_ENV} is not set; research is skipped and plans/reports are rule-based.")
    if os.environ.get("RESYNTH_OPEN_BROWSER") == "1":
        import threading
        import webbrowser

        threading.Timer(2.0, webbrowser.open, args=(f"http://127.0.0.1:{port}/",)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()

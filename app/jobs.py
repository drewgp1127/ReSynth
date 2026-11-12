"""Job folders and the stage runner.

A job lives in `<jobs root>/<id>/`. The web server never runs a heavy stage in its own process:
each phase is a separate Python process (`python -m app.jobs <phase> <job_dir>`), so stem
separation's memory is freed afterwards, the match stage's spawn-based worker pool starts from a
plain main module on every OS, and a crash cannot take the server down. Progress is read from
the per-stage status files the stages already write (`<job>/status/<stage>.json`).

Phases:
  research   stage 1 (started at upload, runs alongside the rest)
  isolate    stage 2 for the marked region -> stems for the user to pick from
  process    choose the stem, then analyse -> plan (waits for research) -> match -> report

job.json holds the user's inputs and the job's coarse state:
  {id, created, song, artist, description, gear, upload, duration_s, region, stem, budget_s,
   phase: uploaded|isolating|choose_stem|processing|done|failed, error}
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from app.stages.common import SR, read_json, write_audio, write_json, write_status

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = Path(os.environ.get("RESYNTH_JOBS", ROOT / "jobs")).resolve()  # absolute: phases run with cwd=ROOT
MAX_CLIP_S = 60.0
STAGES = ["research", "isolate", "analyse", "plan", "match", "report"]
RESEARCH_WAIT_S = 240.0
STALE_S = 900.0  # a running job whose status files have not changed for this long was interrupted  # plan waits this long for research before planning without it


class JobError(ValueError):
    """A problem the user can fix (bad upload, bad region...). Message is shown as-is."""


# ---- audio decoding ---------------------------------------------------------------------
def ffmpeg_path() -> str | None:
    return os.environ.get("RESYNTH_FFMPEG") or shutil.which("ffmpeg")


def decode_audio(src: Path, dst_wav: Path) -> float:
    """Decode an uploaded file to a 44.1 kHz stereo float WAV. Returns its duration in seconds.

    WAV/FLAC/OGG (and MP3 with libsndfile >= 1.1) decode with soundfile; anything it cannot read
    (M4A/AAC, older MP3 setups) goes through the ffmpeg executable when one is installed."""
    import soundfile as sf

    try:
        data, sr = sf.read(str(src), dtype="float32", always_2d=True)
        audio = data.T
        if sr != SR:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    except Exception as e:
        exe = ffmpeg_path()
        if not exe:
            raise JobError(f"could not read {src.suffix or 'this'} audio ({type(e).__name__}); M4A/AAC files "
                           "need ffmpeg installed (see the README), or convert the clip to WAV or FLAC") from e
        tmp = dst_wav.with_name("decoded_tmp.wav")
        proc = subprocess.run([exe, "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-t", str(MAX_CLIP_S + 5),
                               "-ac", "2", "-ar", str(SR), "-c:a", "pcm_f32le", str(tmp)],
                              capture_output=True, text=True, timeout=120)
        if proc.returncode != 0 or not tmp.exists():
            raise JobError("ffmpeg could not decode this file: " + (proc.stderr.strip().splitlines() or ["unknown error"])[-1])
        data, _ = sf.read(str(tmp), dtype="float32", always_2d=True)
        tmp.unlink(missing_ok=True)
        audio = data.T
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    audio = np.ascontiguousarray(audio[:2])
    dur = audio.shape[1] / SR
    if dur < 1.0:
        raise JobError("the clip is shorter than 1 second")
    if dur > MAX_CLIP_S + 0.5:
        raise JobError(f"the clip is {dur:.0f} s long; upload at most {MAX_CLIP_S:.0f} s")
    if float(np.abs(audio).max()) < 1e-5:
        raise JobError("the clip is silent")
    write_audio(dst_wav, audio)
    return dur


# ---- job folder ---------------------------------------------------------------------------
def job_dir(job_id: str) -> Path:
    if not job_id or not all(c.isalnum() or c == "-" for c in job_id):
        raise JobError("bad job id")
    d = JOBS_DIR / job_id
    if not d.is_dir():
        raise FileNotFoundError(job_id)
    return d


def load_job(d: Path) -> dict[str, Any]:
    return read_json(d / "job.json")


def save_job(d: Path, job: dict[str, Any]) -> None:
    write_json(d / "job.json", job)


def create_job(filename: str, data: bytes, song: str, artist: str, description: str, gear: str,
               budget_s: float) -> dict[str, Any]:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    d = JOBS_DIR / job_id
    d.mkdir()
    ext = Path(filename).suffix.lower()[:8] if Path(filename).suffix else ".bin"
    if not all(c.isalnum() or c == "." for c in ext):
        ext = ".bin"
    raw = d / f"upload_original{ext}"
    raw.write_bytes(data)
    try:
        dur = decode_audio(raw, d / "upload.wav")
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    job = {"id": job_id, "created": time.time(), "song": song.strip(), "artist": artist.strip(),
           "description": description.strip(), "gear": gear.strip(), "upload": "upload.wav",
           "original_name": Path(filename).name, "duration_s": round(dur, 3), "region": None, "stem": None,
           "budget_s": float(budget_s), "phase": "uploaded", "error": None}
    save_job(d, job)
    return job


def read_statuses(d: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for s in STAGES:
        p = d / "status" / f"{s}.json"
        if p.exists():
            try:
                out[s] = read_json(p)
            except ValueError:  # being replaced right now
                continue
    return out


# ---- phase processes ---------------------------------------------------------------------
class Runner:
    """Starts phase processes and remembers them so the server can notice crashes.

    One job's pipeline at a time: `busy()` is true while an isolate or process phase runs."""

    def __init__(self) -> None:
        self.procs: dict[tuple[str, str], subprocess.Popen] = {}

    def start(self, d: Path, phase: str) -> None:
        log = open(d / f"{phase}.log", "ab")
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.procs[(d.name, phase)] = subprocess.Popen(
            [sys.executable, "-m", "app.jobs", phase, str(d)], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
            env=env, creationflags=flags)
        log.close()

    def running(self, job_id: str, phase: str) -> bool:
        p = self.procs.get((job_id, phase))
        return p is not None and p.poll() is None

    def exit_code(self, job_id: str, phase: str) -> int | None:
        p = self.procs.get((job_id, phase))
        return None if p is None else p.poll()

    def busy(self) -> str | None:
        for (job_id, phase), p in self.procs.items():
            if phase in ("isolate", "process") and p.poll() is None:
                return job_id
        return None

    def stop_all(self) -> None:
        for p in self.procs.values():
            if p.poll() is None:
                p.terminate()


def reconcile(d: Path, job: dict[str, Any], runner: Runner) -> dict[str, Any]:
    """Mark stages failed when their process died without saying so (crash, killed)."""
    statuses = read_statuses(d)
    for phase, stages in (("research", ["research"]), ("isolate", ["isolate"]),
                          ("process", ["analyse", "plan", "match", "report"])):
        code = runner.exit_code(d.name, phase)
        if code is None or code == 0:
            continue
        if any((statuses.get(s) or {}).get("state") == "failed" for s in stages):
            continue  # the stage reported its own failure
        running = [s for s in stages if (statuses.get(s) or {}).get("state") == "running"]
        missing = [s for s in stages if s not in statuses]
        culprit = (running or missing or [None])[0]
        if culprit is None:
            continue
        write_status(d, culprit, "failed", 0.0, f"the worker process stopped unexpectedly (exit code {code}); "
                                               f"see {phase}.log in the job folder")
        if phase != "research" and job["phase"] not in ("failed", "done"):
            job.update(phase="failed", error=f"{culprit} stopped unexpectedly")
            save_job(d, job)
    phase = {"isolating": "isolate", "processing": "process"}.get(job["phase"])
    if phase and (d.name, phase) not in runner.procs:
        # the server restarted while this job ran; give up if nothing has moved for a while
        newest = max([float(st.get("updated", 0)) for st in statuses.values()] + [(d / "job.json").stat().st_mtime])
        if time.time() - newest > STALE_S:
            job.update(phase="failed", error="the job was interrupted (the server was restarted); mark the region again")
            save_job(d, job)
    return job


# ---- phase bodies (run in the child process) ---------------------------------------------
def phase_research(d: Path) -> None:
    from app.stages.research import run_research

    job = load_job(d)
    run_research(d, job["song"], job["artist"], job["description"])


def phase_isolate(d: Path) -> None:
    from app.stages.isolate import run_isolate

    job = load_job(d)
    try:
        a, b = job["region"]
        run_isolate(d, d / job["upload"], float(a), float(b))
        job = load_job(d)
        job["phase"] = "choose_stem"
    except Exception as e:
        job = load_job(d)
        job.update(phase="failed", error=f"isolate: {type(e).__name__}: {e}")
        save_job(d, job)
        raise
    save_job(d, job)


def _wait_for_research(d: Path, timeout_s: float) -> None:
    write_status(d, "plan", "running", 0.05, "waiting for gear research to finish")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        p = d / "status" / "research.json"
        if p.exists():
            try:
                if read_json(p).get("state") != "running":
                    return
            except ValueError:
                pass
        else:
            return  # research never started
        time.sleep(1.0)


def phase_process(d: Path) -> None:
    from app.stages.analyse import run_analyse
    from app.stages.isolate import choose_target
    from app.stages.match import run_match
    from app.stages.plan import run_plan
    from app.stages.report import run_report

    job = load_job(d)
    stage = "analyse"
    try:
        choose_target(d, job["stem"])
        run_analyse(d)
        stage = "plan"
        _wait_for_research(d, RESEARCH_WAIT_S)
        run_plan(d, job["description"])
        stage = "match"
        run_match(d, budget_s=job["budget_s"])
        stage = "report"
        run_report(d, job["gear"])
        job = load_job(d)
        job["phase"] = "done"
    except Exception as e:
        job = load_job(d)
        job.update(phase="failed", error=f"{stage}: {type(e).__name__}: {e}")
        save_job(d, job)
        raise
    save_job(d, job)


PHASES = {"research": phase_research, "isolate": phase_isolate, "process": phase_process}


def main() -> None:
    ap = argparse.ArgumentParser(description="ReSynth job phase runner (started by the web server)")
    ap.add_argument("phase", choices=sorted(PHASES))
    ap.add_argument("job_dir")
    a = ap.parse_args()
    PHASES[a.phase](Path(a.job_dir))


if __name__ == "__main__":
    main()

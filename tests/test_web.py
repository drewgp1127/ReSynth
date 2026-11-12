"""Web API checks with phase processes stubbed out (the real flow is tests/browser/flow.mjs)."""
import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import jobs
from app import main as web


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "jobs")
    started = []
    monkeypatch.setattr(web.runner, "start", lambda d, phase: started.append((d.name, phase)))
    monkeypatch.delenv("RESYNTH_PASSWORD", raising=False)
    c = TestClient(web.app)
    c.started = started
    return c


def wav_bytes(seconds: float, sr: int = 22050, fmt: str = "WAV") -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(buf, 0.3 * np.sin(2 * np.pi * 220 * t), sr, format=fmt)
    return buf.getvalue()


def upload(client, data: bytes, name: str = "clip.wav"):
    return client.post("/api/jobs", files={"file": (name, data, "audio/wav")},
                       data={"song": "S", "artist": "A", "description": "d", "gear": "g"})


def test_config(client):
    c = client.get("/api/config").json()
    assert c["max_clip_s"] == 60 and c["region_s"] == [1.0, 3.0] and "key_present" in c


def test_upload_region_and_files(client):
    r = upload(client, wav_bytes(5.0))
    assert r.status_code == 200, r.text
    state = r.json()
    job = state["job"]
    assert job["phase"] == "uploaded" and abs(job["duration_s"] - 5.0) < 0.01
    assert client.started == [(job["id"], "research")]
    d = jobs.JOBS_DIR / job["id"]
    info = sf.info(str(d / "upload.wav"))
    assert info.samplerate == 44100 and info.channels == 2

    assert client.get(f"/api/jobs/{job['id']}/files/upload.wav").status_code == 200
    assert client.get(f"/api/jobs/{job['id']}/files/job.json").status_code == 404
    assert client.get(f"/api/jobs/{job['id']}/files/../../etc/passwd").status_code == 404

    assert client.post(f"/api/jobs/{job['id']}/region", json={"start": 1.0, "end": 1.5}).status_code == 400
    assert client.post(f"/api/jobs/{job['id']}/region", json={"start": 4.0, "end": 6.0}).status_code == 400
    r = client.post(f"/api/jobs/{job['id']}/region", json={"start": 1.0, "end": 3.0})
    assert r.status_code == 200 and r.json()["job"]["phase"] == "isolating"
    assert client.started[-1] == (job["id"], "isolate")
    # stems are not ready yet
    assert client.post(f"/api/jobs/{job['id']}/stem", json={"stem": "raw"}).status_code == 409
    assert client.get(f"/api/jobs/{job['id']}/report").status_code == 404


def test_flac_upload(client):
    assert upload(client, wav_bytes(2.0, fmt="FLAC"), "clip.flac").status_code == 200


def test_rejects_long_bad_and_silent(client):
    r = upload(client, wav_bytes(61.5))
    assert r.status_code == 400 and "60" in r.json()["detail"]
    r = upload(client, b"this is not audio at all" * 100, "notes.wav")
    assert r.status_code == 400
    buf = io.BytesIO()
    sf.write(buf, np.zeros(44100 * 2), 44100, format="WAV")
    assert upload(client, buf.getvalue()).status_code == 400
    assert not any((jobs.JOBS_DIR).iterdir())  # failed uploads leave no job folders


def test_unknown_job(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/..%2F..").status_code == 404


def test_password_gate(client, monkeypatch):
    monkeypatch.setenv("RESYNTH_PASSWORD", "secret")
    assert client.get("/api/config").status_code == 401
    assert client.get("/api/config", auth=("any", "secret")).status_code == 200
    assert client.get("/api/config", auth=("any", "wrong")).status_code == 401


def test_crashed_phase_is_reported(client, tmp_path):
    job = upload(client, wav_bytes(3.0)).json()["job"]
    d = jobs.JOBS_DIR / job["id"]
    j = jobs.load_job(d)
    j["phase"] = "processing"
    jobs.save_job(d, j)
    (d / "status").mkdir(exist_ok=True)
    (d / "status" / "match.json").write_text(json.dumps({"stage": "match", "state": "running", "progress": 0.3}))

    class Dead:
        def poll(self):
            return 1

    web.runner.procs[(job["id"], "process")] = Dead()
    state = client.get(f"/api/jobs/{job['id']}").json()
    assert state["job"]["phase"] == "failed"
    assert state["stages"]["match"]["state"] == "failed"
    web.runner.procs.clear()

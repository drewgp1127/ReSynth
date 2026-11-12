import json

import numpy as np
import pytest

import app.stages.isolate as iso
from app.stages.common import read_audio, write_audio
from tests.audio_fixtures import drum_bass_bed

SR = 44100


def _song(tmp_path) -> np.ndarray:
    song = drum_bass_bed(6.0)
    t = np.arange(int(1.5 * SR)) / SR
    song[:, 2 * SR:2 * SR + len(t)] += 0.3 * np.sin(2 * np.pi * 440 * t)
    write_audio(tmp_path / "song.wav", song)
    return song


def test_crop_and_degrade_when_separator_unavailable(tmp_path, monkeypatch):
    song = _song(tmp_path)

    def blocked(*a, **k):
        raise ConnectionError("Failed to download https://github.com/TRvlvr/model_repo/releases/x.yaml: 403")

    monkeypatch.setattr(iso, "separate", blocked)
    info = iso.run_isolate(tmp_path, "song.wav", 2.0, 3.5, padding_s=0.05)
    assert info["status"] == "raw_slice_only" and info["blocked_host"] == "github.com"
    status = json.loads((tmp_path / "status" / "isolate.json").read_text())
    assert status["state"] == "degraded" and "github.com" in status["message"]
    sl = read_audio(tmp_path / "slice.wav")
    assert sl.shape == (2, round(1.6 * SR))
    assert np.allclose(sl, song[:, round(1.95 * SR):round(3.55 * SR)], atol=1e-4)
    iso.choose_target(tmp_path, "raw")
    assert read_audio(tmp_path / "target.wav").shape == (2, sl.shape[1])
    assert read_audio(tmp_path / "target_mono.wav").shape == (1, sl.shape[1])


def test_region_outside_clip_fails(tmp_path):
    _song(tmp_path)
    with pytest.raises(ValueError):
        iso.run_isolate(tmp_path, "song.wav", 5.0, 9.0)
    assert json.loads((tmp_path / "status" / "isolate.json").read_text())["state"] == "failed"


@pytest.mark.slow
def test_real_separation(tmp_path):
    pytest.importorskip("audio_separator")
    _song(tmp_path)
    info = iso.run_isolate(tmp_path, "song.wav", 2.0, 3.5)
    if info["status"] != "ok":
        pytest.skip(info["message"])
    assert {"drums", "bass", "other"} <= set(info["stems"])
    assert info["suggested"] not in ("drums", "vocals")
    for rel in info["stems"].values():
        assert read_audio(tmp_path / rel).shape[1] == read_audio(tmp_path / "slice.wav").shape[1]
    assert info["rms"]["drums"] > 0.01

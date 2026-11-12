"""Shared helpers for stages: job-folder IO and status files.

Each stage writes `<job_dir>/status/<stage>.json` (one file per stage, so stages running in
parallel never clobber each other). Fields: stage, state (running|done|failed|degraded),
progress (0..1), message, updated (unix time), plus stage-specific extras.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

SR = 44100


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def write_status(job_dir: Path, stage: str, state: str, progress: float = 0.0, message: str = "", **extra: Any) -> None:
    d = Path(job_dir) / "status"
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / f"{stage}.json", {
        "stage": stage, "state": state, "progress": round(float(progress), 3),
        "message": message, "updated": time.time(), **extra,
    })


def read_audio(path: Path, sr: int = SR) -> np.ndarray:
    """Read audio as float32 (channels, n) at `sr` (resampled with librosa if needed)."""
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.T
    if file_sr != sr:
        import librosa

        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
    return np.ascontiguousarray(data, dtype=np.float32)


def write_audio(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    sf.write(str(path), audio.T if audio.ndim == 2 else audio, sr, subtype="FLOAT")


def make_ab(target: np.ndarray, match: np.ndarray, gap_s: float = 0.5, sr: int = SR) -> np.ndarray:
    """Target, a gap, then the match RMS-matched to the target, as one stereo array."""
    def stereo(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return np.stack([x, x]) if x.ndim == 1 else (np.repeat(x, 2, axis=0) if x.shape[0] == 1 else x[:2])

    t, m = stereo(target), stereo(match)
    rt, rm = float(np.sqrt(np.mean(t ** 2))), float(np.sqrt(np.mean(m ** 2)))
    if rm > 1e-9:
        m = m * (rt / rm)
    out = np.concatenate([t, np.zeros((2, int(gap_s * sr)), np.float32), m], axis=1)
    peak = float(np.abs(out).max())
    return out / peak * 0.98 if peak > 0.98 else out

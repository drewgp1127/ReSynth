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

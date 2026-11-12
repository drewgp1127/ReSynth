"""Stage 2: crop the marked region from the uploaded clip and separate stems.

Input:  <job>/<upload file> (any format soundfile/librosa can read), region start/end in seconds
Output: <job>/slice.wav            raw marked region (+ `padding_s` each side), stereo 44.1 kHz
        <job>/stems/<name>.wav     one per stem (htdemucs_6s: bass, drums, guitar, other, piano, vocals),
                                   same span as slice.wav
        <job>/isolate.json         region, files, per-stem RMS, suggested stem, status
        <job>/status/isolate.json
Then `choose_target(job_dir, "piano" | "raw" | ...)` writes target.wav (stereo) and
target_mono.wav for the analyse/match stages.

Separation runs on the region plus `context_s` of audio either side (demucs needs context), and
the stems are trimmed back afterwards. If the separator or its model weights are unavailable,
the stage degrades to the raw slice only and says why in the status message.

Run: python -m app.stages.isolate <job_dir> <upload_file> <start_s> <end_s>
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from app.stages.common import SR, read_audio, read_json, write_audio, write_json, write_status

DEFAULT_MODEL = "htdemucs_6s.yaml"
MODELS_DIR = Path(os.environ.get("PATCH_FINDER_MODELS", Path(__file__).resolve().parents[2] / "models"))
NON_SYNTH_STEMS = {"drums", "vocals"}
# htdemucs_6s splits synths unpredictably between other/piano/guitar, so offer their sum too
MELODIC_STEMS = ("other", "piano", "guitar")
MELODIC_MIX = "melodic"


def _stem_name(filename: str) -> str:
    m = re.search(r"\(([^)]+)\)", filename)
    return (m.group(1) if m else Path(filename).stem).lower().replace(" ", "_")


def _blocked_host(err: BaseException) -> str | None:
    m = re.search(r"https?://([^/\s'\"]+)", str(err))
    return m.group(1) if m else None


def separate(audio: np.ndarray, model: str = DEFAULT_MODEL, models_dir: Path = MODELS_DIR) -> dict[str, np.ndarray]:
    """Run audio-separator on `audio` (channels, n) and return {stem: (channels, n)} at SR."""
    from audio_separator.separator import Separator  # heavy import (torch)

    models_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        src = tmp_dir / "input.wav"
        write_audio(src, audio)
        sep = Separator(model_file_dir=str(models_dir), output_dir=str(tmp_dir / "out"),
                        log_level=logging.WARNING, output_format="WAV")
        sep.load_model(model_filename=model)
        files = sep.separate(str(src))
        stems = {}
        for f in files:
            p = Path(f)
            if not p.is_absolute():
                p = tmp_dir / "out" / p
            stems[_stem_name(p.name)] = read_audio(p, SR)
    return stems


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    x = x[:, :n]
    return np.pad(x, ((0, 0), (0, n - x.shape[1]))) if x.shape[1] < n else x


def run_isolate(job_dir: str | Path, upload: str | Path, start_s: float, end_s: float, padding_s: float = 0.05,
                context_s: float = 1.0, model: str = DEFAULT_MODEL, models_dir: Path = MODELS_DIR) -> dict[str, Any]:
    job = Path(job_dir)
    upload = Path(upload) if Path(upload).is_absolute() else job / upload
    write_status(job, "isolate", "running", 0.05, "reading upload")
    try:
        audio = read_audio(upload, SR)
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        audio = audio[:2]
        total = audio.shape[1] / SR
        if not 0 <= start_s < end_s <= total + 1e-6:
            raise ValueError(f"region {start_s}-{end_s} s is outside the clip (0-{total:.2f} s)")
        a, b = max(start_s - padding_s, 0.0), min(end_s + padding_s, total)
        ia, ib = int(round(a * SR)), int(round(b * SR))
        raw = audio[:, ia:ib]
        write_audio(job / "slice.wav", raw)

        ca, cb = max(ia - int(context_s * SR), 0), min(ib + int(context_s * SR), audio.shape[1])
        write_status(job, "isolate", "running", 0.2, f"separating stems with {model} (CPU, may take a minute)")
        info: dict[str, Any] = {"region_s": [start_s, end_s], "slice_span_s": [round(a, 4), round(b, 4)],
                                "padding_s": padding_s, "slice": "slice.wav", "model": model, "stems": {}}
        try:
            stems = separate(audio[:, ca:cb], model, models_dir)
        except Exception as e:  # missing package, blocked weight download, model error
            host = _blocked_host(e)
            msg = (f"stem separation unavailable ({type(e).__name__}: {e}); "
                   + (f"model download host: {host}; " if host else "") + "using the raw slice only")
            info.update(status="raw_slice_only", message=msg, blocked_host=host, suggested="raw",
                        rms={"raw": float(np.sqrt(np.mean(raw ** 2)))})
            write_json(job / "isolate.json", info)
            write_status(job, "isolate", "degraded", 1.0, msg)
            return info

        (job / "stems").mkdir(exist_ok=True)
        stems = {k: _fit(v, cb - ca) for k, v in stems.items()}
        if all(k in stems for k in MELODIC_STEMS):
            stems[MELODIC_MIX] = sum(stems[k] for k in MELODIC_STEMS)
        rms = {"raw": float(np.sqrt(np.mean(raw ** 2)))}
        focus: dict[str, float] = {}
        ctx_mask = np.ones(cb - ca, dtype=bool)
        ctx_mask[ia - ca:ib - ca] = False
        for name, full in stems.items():
            stem = full[:, ia - ca:ib - ca]
            write_audio(job / "stems" / f"{name}.wav", stem)
            info["stems"][name] = f"stems/{name}.wav"
            rms[name] = float(np.sqrt(np.mean(stem ** 2)))
            ctx = float(np.sqrt(np.mean(full[:, ctx_mask] ** 2))) if ctx_mask.any() else 0.0
            focus[name] = rms[name] / (ctx + 1e-4)  # >1: louder in the marked region than around it
        # suggestion: the stem that is loud in the region AND concentrated in it (the user marked
        # where the sound is clearest); drums/vocals excluded
        scores = {k: rms[k] * min(focus[k], 10.0) ** 0.5 for k in stems if k not in NON_SYNTH_STEMS}
        info.update(status="ok", message=f"{len(stems)} stems", rms={k: round(v, 6) for k, v in rms.items()},
                    focus={k: round(v, 3) for k, v in focus.items()},
                    suggested=max(scores, key=scores.get) if scores else "raw")
        write_json(job / "isolate.json", info)
        write_status(job, "isolate", "done", 1.0, f"{len(stems)} stems written; suggested: {info['suggested']}")
        return info
    except Exception as e:
        write_status(job, "isolate", "failed", 0.0, f"{type(e).__name__}: {e}")
        raise


def choose_target(job_dir: str | Path, choice: str) -> Path:
    """Copy the chosen stem ("raw" = slice.wav) to target.wav (stereo) and target_mono.wav."""
    job = Path(job_dir)
    info = read_json(job / "isolate.json")
    src = job / (info["slice"] if choice == "raw" else info["stems"][choice])
    audio = read_audio(src, SR)
    write_audio(job / "target.wav", audio)
    write_audio(job / "target_mono.wav", audio.mean(axis=0))
    info["chosen"] = choice
    write_json(job / "isolate.json", info)
    return job / "target.wav"


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch Finder stage 2: crop + stem separation")
    ap.add_argument("job_dir")
    ap.add_argument("upload")
    ap.add_argument("start", type=float)
    ap.add_argument("end", type=float)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--choose", default=None, help="stem to copy to target.wav (default: suggested)")
    a = ap.parse_args()
    info = run_isolate(a.job_dir, a.upload, a.start, a.end, model=a.model)
    choose_target(a.job_dir, a.choose or info["suggested"])
    print(info["status"], info["message"], "rms:", info.get("rms"), "->", a.choose or info["suggested"])


if __name__ == "__main__":
    main()

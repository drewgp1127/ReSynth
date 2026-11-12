"""Measure render and loss throughput. Run: .venv/bin/python scripts/bench.py"""
from __future__ import annotations

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from app.synth.builder import base_patch  # noqa: E402
from app.synth.loss import LossFn  # noqa: E402
from app.synth.vital import Note, VitalRenderer  # noqa: E402


def main() -> None:
    p = base_patch("bench")
    p["settings"]["osc_2_on"] = 1.0
    p["settings"]["osc_1_unison_voices"] = 3.0
    r = VitalRenderer(p)
    n = 30
    t = time.perf_counter()
    for _ in range(n):
        r.synth.render(60, 0.8, 0.8, 1.0)
    dt = (time.perf_counter() - t) / n
    print(f"single note, 1.0 s render: {dt * 1000:.1f} ms -> {1 / dt:.1f} renders/s/process")

    notes = [Note(57, 0.0, 0.4), Note(61, 0.0, 0.4), Note(64, 0.0, 0.4),
             Note(55, 0.6, 0.4), Note(59, 0.6, 0.4), Note(62, 0.6, 0.4)]
    t = time.perf_counter()
    for _ in range(10):
        r.set_params({"filter_1_cutoff": 70.0})
        audio = r.render_notes(notes, 1.2)
    dt = (time.perf_counter() - t) / 10
    print(f"candidate (set params + 6 notes, 1.2 s clip): {dt * 1000:.1f} ms -> {1 / dt:.1f} candidates/s/process")

    loss = LossFn(audio, onsets=[0.0, 0.6])
    t = time.perf_counter()
    for _ in range(30):
        loss(audio * 0.9 + 0.001 * np.random.default_rng(0).standard_normal(audio.shape).astype(np.float32))
    dt = (time.perf_counter() - t) / 30
    print(f"loss eval (1.2 s clip): {dt * 1000:.1f} ms -> {1 / dt:.1f} loss evals/s/process")


if __name__ == "__main__":
    main()

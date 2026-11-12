"""Milestone 0 spike: prove the vita API does what ReSynth needs.

Run: .venv/bin/python scripts/spike_vita.py
Prints one line per check; exits non-zero if any check fails.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import vita

SR = 44100


def centroid(a: np.ndarray) -> float:
    return float(librosa.feature.spectral_centroid(y=a.mean(0), sr=SR).mean())


def midi_track(a: np.ndarray) -> np.ndarray:
    f0, _, _ = librosa.pyin(a.mean(0), fmin=60, fmax=4000, sr=SR, frame_length=2048, hop_length=441)
    return librosa.hz_to_midi(f0)


def settle(s: vita.Synth) -> None:
    """Warm-up render. Vital smooths control changes over ~0.2 s of processed audio, so the
    first render after loading a preset or setting controls differs from later ones."""
    s.render(60, 0.0, 0.02, 0.3)


def main() -> int:
    failures = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    s = vita.Synth()
    s.set_sample_rate(SR)
    c = s.get_controls()
    c["osc_1_random_phase"].set(0.0)  # random unison phase makes renders non-deterministic

    a = s.render(57, 0.8, 0.5, 1.0)
    check("render", a.shape == (2, SR) and np.abs(a).max() > 0.05, f"shape={a.shape} peak={np.abs(a).max():.3f}")

    c["filter_1_on"].set(1.0)
    c["filter_1_cutoff"].set(40.0)
    settle(s)
    b = s.render(57, 0.8, 0.5, 1.0)
    check("param change", centroid(b) < 0.6 * centroid(a), f"centroid {centroid(a):.0f} Hz -> {centroid(b):.0f} Hz with cutoff 40")
    c["filter_1_on"].set(0.0)

    ok = s.connect_modulation("env_3", "voice_transpose")
    c["env_3_attack"].set(0.01 ** 0.25)  # env times are stored as seconds ** (1/4)
    c["env_3_decay"].set(0.12 ** 0.25)
    c["env_3_sustain"].set(0.0)
    c["modulation_1_amount"].set(12 / 96)  # amount 1.0 == full transpose range (96 semitones)
    settle(s)
    m = midi_track(s.render(57, 0.8, 0.6, 0.6))
    peak = float(np.nanmax(m[:10]))
    check("env -> pitch modulation", ok and peak > 66 and abs(np.nanmedian(m[30:50]) - 57) < 0.3,
          f"peak midi {peak:.1f} in first 100 ms, settles at {np.nanmedian(m[30:50]):.2f}")

    js = s.to_json()
    settle(s)
    a1 = s.render(57, 0.8, 0.5, 1.0)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "spike.vital"
        p.write_text(js)
        json.loads(p.read_text())
        s2 = vita.Synth()
        s2.set_sample_rate(SR)
        loaded = s2.load_preset(str(p))
        settle(s2)
        a2 = s2.render(57, 0.8, 0.5, 1.0)
    check("save .vital + reload identical", loaded and np.abs(a1 - a2).max() < 1e-5, f"max abs diff {np.abs(a1 - a2).max():.2e}")

    # chords: render() takes one midi note, so render per note and sum.
    notes = [57, 61, 64]
    chord = sum(s.render(n, 0.8, 0.5, 1.0) for n in notes)
    chroma = librosa.feature.chroma_stft(y=chord.mean(0), sr=SR).mean(1)
    top = sorted(np.argsort(chroma)[-3:].tolist())
    check("chord by per-note sum", top == sorted(n % 12 for n in notes), f"top chroma bins {top}")

    print("FAILED: " + ", ".join(failures) if failures else "all checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

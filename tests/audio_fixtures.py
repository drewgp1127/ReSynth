"""Synthetic test audio: a simple drum + bass bed (no recorded or copyrighted audio)."""
from __future__ import annotations

import numpy as np

SR = 44100


def _env(n: int, decay_s: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (decay_s * SR))


def drum_bass_bed(duration: float, bpm: float = 100.0, bass_notes: tuple[int, ...] = (33, 33, 28, 31),
                  seed: int = 0) -> np.ndarray:
    """Kick on 1 and 3, snare on 2 and 4, eighth-note hats, one saw-ish bass note per beat. (2, n)."""
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    out = np.zeros(n)
    beat = 60.0 / bpm
    t = np.arange(int(0.4 * SR)) / SR
    kick = np.sin(2 * np.pi * (50 * t + (120 - 50) * 0.03 * (1 - np.exp(-t / 0.03)))) * _env(len(t), 0.12)
    snare = rng.standard_normal(int(0.2 * SR)) * _env(int(0.2 * SR), 0.05)
    snare = np.convolve(snare, np.ones(4) / 4, "same") + 0.5 * np.sin(2 * np.pi * 190 * t[:len(snare)]) * _env(len(snare), 0.04)
    hat = np.diff(rng.standard_normal(int(0.05 * SR) + 1)) * _env(int(0.05 * SR), 0.01)
    i = 0
    while i * beat < duration:
        s = int(i * beat * SR)
        for sample, gain, off in ((kick if i % 2 == 0 else snare, 0.9 if i % 2 == 0 else 0.5, 0),
                                  (hat, 0.15, 0), (hat, 0.12, int(beat / 2 * SR))):
            a = s + off
            m = min(len(sample), n - a)
            if m > 0:
                out[a:a + m] += gain * sample[:m]
        midi = bass_notes[i % len(bass_notes)]
        f = 440 * 2 ** ((midi - 69) / 12)
        bn = min(int(beat * 0.9 * SR), n - s)
        if bn > 0:
            tb = np.arange(bn) / SR
            saw = sum(np.sin(2 * np.pi * k * f * tb) / k for k in range(1, 8))
            out[s:s + bn] += 0.35 * saw * np.minimum(1, tb / 0.005) * _env(bn, 0.6)
        i += 1
    return np.stack([out, out]).astype(np.float32)

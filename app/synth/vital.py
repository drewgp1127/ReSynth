"""Thin wrapper over vita: load a template, set raw params, render a list of notes, save .vital.

vita renders one note per call, so a note list is rendered note by note and summed at each
onset (exact for per-voice processing; global effects must stay off while searching).
See docs/vita-notes.md for the API facts this relies on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import vita

SR = 44100
WARMUP = (60, 0.0, 0.02, 0.3)  # settles Vital's control smoothing after changes
TAIL_CAP_S = 2.0
DUR_QUANTUM_S = 0.02  # note lengths are rounded to this so repeated notes share one render


@dataclass(frozen=True)
class Note:
    midi: int
    onset: float  # seconds
    duration: float  # seconds
    velocity: float = 0.8  # 0..1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Note":
        return cls(int(round(d["midi"])), float(d["onset"]), float(d["duration"]), float(d.get("velocity", 0.8)))


def load_template(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def modulation_slots(patch: dict[str, Any]) -> dict[str, int]:
    """Map "source->destination" to its 1-based modulation slot."""
    slots = {}
    for i, m in enumerate(patch["settings"]["modulations"]):
        if m.get("source") and m.get("destination"):
            slots.setdefault(f"{m['source']}->{m['destination']}", i + 1)
    return slots


def resolve_controls(patch: dict[str, Any], raw: dict[str, float]) -> dict[str, float]:
    """Translate "src->dst" keys into modulation_N_amount control names."""
    slots = modulation_slots(patch)
    out = {}
    for key, value in raw.items():
        if "->" in key:
            if key not in slots:
                raise KeyError(f"template has no modulation {key}")
            key = f"modulation_{slots[key]}_amount"
        out[key] = float(value)
    return out


def apply_to_patch(patch: dict[str, Any], raw: dict[str, float], name: str | None = None) -> dict[str, Any]:
    """Return a copy of the patch dict with raw control values written into its settings."""
    out = json.loads(json.dumps(patch))
    out["settings"].update(resolve_controls(patch, raw))
    if name:
        out["preset_name"] = name
    return out


def save_vital(patch: dict[str, Any], path: str | Path, raw: dict[str, float] | None = None,
               name: str | None = None) -> Path:
    path = Path(path)
    data = apply_to_patch(patch, raw or {}, name)
    path.write_text(json.dumps(data))
    return path


def amp_release_seconds(controls: dict[str, Any]) -> float:
    return float(controls["env_1_release"].value()) ** 4


def mix_notes(render_one, notes: Iterable[Note], duration: float, tail: float, sr: int = SR) -> np.ndarray:
    """Place per-note renders at their onsets in a (2, n) buffer of `duration` seconds.

    Notes with the same pitch, velocity and length are rendered once and reused (render calls
    dominate the search's cost); each render is long enough for the earliest such note."""
    n_total = int(round(duration * sr))
    out = np.zeros((2, n_total), dtype=np.float32)
    groups: dict[tuple, list[Note]] = {}
    for note in notes:
        if int(round(note.onset * sr)) < n_total:
            dur = max(round(note.duration / DUR_QUANTUM_S), 1) * DUR_QUANTUM_S
            groups.setdefault((note.midi, round(note.velocity, 4), round(dur, 4)), []).append(note)
    for (midi, vel, dur), group in groups.items():
        render_dur = min(dur + tail, duration - min(n.onset for n in group))
        audio = render_one(midi, vel, dur, render_dur)
        for note in group:
            start = int(round(note.onset * sr))
            n = min(audio.shape[1], n_total - start)
            out[:, start:start + n] += audio[:, :n]
    return out


class VitalRenderer:
    """One Synth loaded with a template. Not thread-safe: use one per worker."""

    def __init__(self, patch: dict[str, Any], sr: int = SR):
        self.patch = patch
        self.sr = sr
        self.synth = vita.Synth()
        self.synth.set_sample_rate(sr)
        if not self.synth.load_json(json.dumps(patch)):
            raise ValueError("vita rejected the template JSON")
        self.controls = self.synth.get_controls()
        self._warm()

    def _warm(self) -> None:
        self.synth.render(*WARMUP)

    def set_params(self, raw: dict[str, float]) -> None:
        for name, value in resolve_controls(self.patch, raw).items():
            self.controls[name].set(value)
        self._warm()

    def render_notes(self, notes: Iterable[Note], duration: float) -> np.ndarray:
        """Render notes into a (2, duration*sr) float32 array."""
        tail = min(1.2 * amp_release_seconds(self.controls) + 0.05, TAIL_CAP_S)
        return mix_notes(self.synth.render, notes, duration, tail, self.sr)

    def to_json(self) -> str:
        return self.synth.to_json()


def render_patch(patch: dict[str, Any], notes: Iterable[Note], duration: float, raw: dict[str, float] | None = None,
                 sr: int = SR) -> np.ndarray:
    """Convenience: fresh synth, apply raw values via JSON, render."""
    r = VitalRenderer(apply_to_patch(patch, raw or {}), sr)
    return r.render_notes(notes, duration)

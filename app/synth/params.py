"""The searchable parameter list.

Each `Param` is expressed in natural units (Hz, seconds, semitones, cents, 0-1 levels) and maps
to one or more raw Vital control values. Modulation amounts are addressed as "source->destination"
and resolved to a `modulation_N_amount` control by `vital.py`, using the template's routing
(see `builder.STANDARD_MODULATIONS`).

The search works in a normalised unit cube: `to_unit` / `from_unit` map natural values to [0, 1]
using linear or log scaling within the param's range.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

# raw = forward(natural) * factor ; natural = inverse(raw / factor)
_TRANSFORMS: dict[str, tuple[Callable[[float], float], Callable[[float], float]]] = {
    "lin": (lambda v: v, lambda r: r),
    "sqrt": (lambda v: math.sqrt(max(v, 0.0)), lambda r: r * r),  # Vital "quadratic" controls
    "quartic": (lambda v: max(v, 0.0) ** 0.25, lambda r: max(r, 0.0) ** 4),  # envelope times (s)
    "hz_to_midi": (lambda v: 69.0 + 12.0 * math.log2(v / 440.0), lambda r: 440.0 * 2 ** ((r - 69.0) / 12.0)),
    "log2": (lambda v: math.log2(v), lambda r: 2.0 ** r),  # LFO frequency in Hz
}


@dataclass(frozen=True)
class Target:
    control: str  # Vital control name, or "source->destination" for a modulation amount
    transform: str = "lin"
    factor: float = 1.0


@dataclass(frozen=True)
class Param:
    name: str
    targets: tuple[Target, ...]
    lo: float
    hi: float
    log: bool = False
    integer: bool = False
    unit: str = ""
    description: str = ""

    def to_raw(self, value: float) -> dict[str, float]:
        value = self.clip(value)
        out = {}
        for t in self.targets:
            fwd, _ = _TRANSFORMS[t.transform]
            out[t.control] = fwd(value) * t.factor
        return out

    def from_raw(self, raw: float) -> float:
        t = self.targets[0]
        _, inv = _TRANSFORMS[t.transform]
        return inv(raw / t.factor)

    def clip(self, value: float) -> float:
        value = min(max(value, self.lo), self.hi)
        return float(round(value)) if self.integer else float(value)

    def with_range(self, lo: float, hi: float) -> "Param":
        return Param(self.name, self.targets, lo, hi, self.log, self.integer, self.unit, self.description)

    def to_unit(self, value: float) -> float:
        value = min(max(value, self.lo), self.hi)
        if self.log:
            return (math.log(value) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (value - self.lo) / (self.hi - self.lo)

    def from_unit(self, u: float) -> float:
        u = min(max(u, 0.0), 1.0)
        if self.log:
            v = math.exp(math.log(self.lo) + u * (math.log(self.hi) - math.log(self.lo)))
        else:
            v = self.lo + u * (self.hi - self.lo)
        return self.clip(v)


def _t(control: str, transform: str = "lin", factor: float = 1.0) -> Target:
    return Target(control, transform, factor)


def _env(prefix: str, env: int, what: str) -> list[Param]:
    return [
        Param(f"{prefix}_attack", (_t(f"env_{env}_attack", "quartic"),), 0.0005, 2.0, log=True, unit="s", description=f"{what} envelope attack"),
        Param(f"{prefix}_decay", (_t(f"env_{env}_decay", "quartic"),), 0.005, 4.0, log=True, unit="s", description=f"{what} envelope decay"),
        Param(f"{prefix}_sustain", (_t(f"env_{env}_sustain"),), 0.0, 1.0, description=f"{what} envelope sustain level"),
        Param(f"{prefix}_release", (_t(f"env_{env}_release", "quartic"),), 0.005, 3.0, log=True, unit="s", description=f"{what} envelope release"),
    ]


def _osc(i: int) -> list[Param]:
    return [
        Param(f"osc{i}_level", (_t(f"osc_{i}_level", "sqrt"),), 0.0, 1.0, description=f"oscillator {i} level"),
        Param(f"osc{i}_wave", (_t(f"osc_{i}_wave_frame", "lin", 255.0),), 0.0, 1.0,
              description=f"oscillator {i} shape: 0 sine, 1/3 triangle, 2/3 square, 1 saw (morphs between)"),
        Param(f"osc{i}_unison_voices", (_t(f"osc_{i}_unison_voices"),), 1, 8, integer=True, unit="voices",
              description=f"oscillator {i} unison voice count"),
        Param(f"osc{i}_unison_detune", (_t(f"osc_{i}_unison_detune", "sqrt"),), 0.0, 60.0, unit="%",
              description=f"oscillator {i} unison detune (Vital %)"),
    ]


PARAMS: dict[str, Param] = {p.name: p for p in [
    *_osc(1), *_osc(2), *_osc(3),
    Param("osc_detune_cents", (_t("osc_1_tune", "lin", -0.005), _t("osc_2_tune", "lin", 0.005)), 0.0, 50.0, unit="cents",
          description="detune between oscillators 1 and 2 (osc 1 down, osc 2 up by half each)"),
    Param("noise_level", (_t("sample_level", "sqrt"),), 0.0, 1.0, description="noise (sampler) level"),
    Param("fm_amount", (_t("osc_1_distortion_amount"),), 0.0, 1.0, description="oscillator 1 FM / distortion amount"),
    Param("filter_cutoff", (_t("filter_1_cutoff", "hz_to_midi"),), 40.0, 16000.0, log=True, unit="Hz", description="filter cutoff"),
    Param("filter_resonance", (_t("filter_1_resonance"),), 0.0, 0.95, description="filter resonance"),
    Param("filter_drive", (_t("filter_1_drive"),), 0.0, 20.0, unit="dB", description="filter drive (per-voice saturation)"),
    Param("filter_env_amount", (_t("env_2->filter_1_cutoff", "lin", 1 / 128),), 0.0, 72.0, unit="semitones",
          description="filter envelope depth on cutoff"),
    *_env("amp", 1, "amplitude"),
    *_env("filt", 2, "filter"),
    Param("pitch_env_amount", (_t("env_3->voice_transpose", "lin", 1 / 96),), -24.0, 24.0, unit="semitones",
          description="pitch envelope peak offset"),
    Param("pitch_env_attack", (_t("env_3_attack", "quartic"),), 0.0005, 0.2, log=True, unit="s", description="pitch envelope rise time"),
    Param("pitch_env_decay", (_t("env_3_decay", "quartic"),), 0.005, 1.0, log=True, unit="s", description="pitch envelope fall time"),
    Param("filter_lfo_rate", (_t("lfo_1_frequency", "log2"),), 0.05, 12.0, log=True, unit="Hz", description="filter LFO rate"),
    Param("filter_lfo_depth", (_t("lfo_1->filter_1_cutoff", "lin", 1 / 128),), 0.0, 36.0, unit="semitones", description="filter LFO depth"),
    Param("vibrato_rate", (_t("lfo_2_frequency", "log2"),), 1.0, 10.0, log=True, unit="Hz", description="vibrato rate"),
    Param("vibrato_depth", (_t("lfo_2->voice_transpose", "lin", 1 / 9600),), 0.0, 100.0, unit="cents", description="vibrato depth"),
    Param("tremolo_rate", (_t("lfo_3_frequency", "log2"),), 0.5, 12.0, log=True, unit="Hz", description="tremolo rate"),
    Param("tremolo_depth", (_t("lfo_3->voice_amplitude"),), 0.0, 1.0, description="tremolo depth"),
]}


@dataclass
class SearchSpace:
    """An ordered subset of PARAMS, optionally with narrowed ranges, plus fixed values."""

    params: list[Param]
    fixed: dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(cls, names: list[str], ranges: dict[str, tuple[float, float]] | None = None,
              fixed: dict[str, float] | None = None) -> "SearchSpace":
        ranges = ranges or {}
        ps = []
        for n in names:
            p = PARAMS[n]
            if n in ranges:
                p = p.with_range(*ranges[n])
            ps.append(p)
        return cls(ps, dict(fixed or {}))

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.params]

    def values_from_unit(self, u: list[float]) -> dict[str, float]:
        return {p.name: p.from_unit(x) for p, x in zip(self.params, u)}

    def unit_from_values(self, values: dict[str, float]) -> list[float]:
        return [p.to_unit(values[p.name]) for p in self.params]

    def raw(self, values: dict[str, float]) -> dict[str, float]:
        """Raw control values for the searched values plus the fixed ones."""
        out: dict[str, float] = {}
        for name, v in {**self.fixed, **values}.items():
            out.update(PARAMS[name].to_raw(v))
        return out

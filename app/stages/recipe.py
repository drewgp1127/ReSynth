"""The matched patch described in generic synth terms, from the FINAL parameter values.

`final_values` merges the template's own values with the plan's fixed values and the match's
best values, so every line of the recipe describes what is actually in patch.vital.
"""
from __future__ import annotations

from typing import Any

from app.stages.match import template_values
from app.synth.params import PARAMS


def final_values(patch: dict[str, Any], fixed: dict[str, float], found: dict[str, float]) -> dict[str, float]:
    vals = template_values(patch, list(PARAMS))
    vals.update({k: float(v) for k, v in fixed.items()})
    vals.update({k: float(v) for k, v in found.items()})
    return vals


# keyframe names per wavetable the templates use (scripts/make_templates.py, app/synth/builder.py)
TABLE_SHAPES = {
    "Basic Shapes": [(0.0, "sine"), (1 / 3, "triangle"), (2 / 3, "square"), (1.0, "saw")],
    "Tri-Saw": [(0.0, "triangle"), (128 / 255, "tri-saw (a triangle skewed towards a saw)"), (1.0, "saw")],
}


def shape_name(w: float, table: str = "Basic Shapes") -> str:
    """Wave-frame position 0..1 on the oscillator's wavetable, in words."""
    points = TABLE_SHAPES.get(table, TABLE_SHAPES["Basic Shapes"])
    for (a, na), (b, nb) in zip(points, points[1:]):
        if w <= b + 1e-9:
            f = (w - a) / (b - a)
            if f < 0.15:
                return na
            if f > 0.85:
                return nb
            na, nb = na.split(" (")[0], nb.split(" (")[0]
            return f"between {na} and {nb} ({int(round(100 * (1 - f)))}% {na}, {int(round(100 * f))}% {nb})"
    return "saw"


def _ms(s: float) -> str:
    return f"{s * 1000:.0f} ms" if s < 1 else f"{s:.2f} s"


def _hz(f: float) -> str:
    return f"{f:.0f} Hz" if f < 1000 else f"{f / 1000:.1f} kHz"


def recipe(patch: dict[str, Any], v: dict[str, float]) -> list[dict[str, Any]]:
    """Sections of {title, lines} describing the patch."""
    st = patch["settings"]
    out: list[dict[str, Any]] = []

    osc_lines = []
    for i in (1, 2, 3):
        if not st.get(f"osc_{i}_on"):
            continue
        level = v[f"osc{i}_level"]
        tables = st.get("wavetables") or []
        table = tables[i - 1].get("name", "Basic Shapes") if len(tables) >= i else "Basic Shapes"
        parts = [f"Oscillator {i}: {shape_name(v[f'osc{i}_wave'], table)} wave, level {level:.0%}"]
        tr = float(st.get(f"osc_{i}_transpose", 0.0))
        if abs(tr) >= 0.5:
            parts.append(f"transposed {tr:+.0f} semitones")
        voices = int(round(v[f"osc{i}_unison_voices"]))
        if voices > 1:
            parts.append(f"{voices}-voice unison, detune {v[f'osc{i}_unison_detune']:.0f}% (Vital scale)")
        osc_lines.append(", ".join(parts) + ".")
    if st.get("osc_1_on") and st.get("osc_2_on") and v["osc_detune_cents"] >= 0.5:
        osc_lines.append(f"Detune oscillators 1 and 2 against each other by {v['osc_detune_cents']:.0f} cents in total "
                         f"(one down {v['osc_detune_cents'] / 2:.0f}, one up {v['osc_detune_cents'] / 2:.0f}).")
    if v["fm_amount"] > 0.02 and st.get("osc_1_distortion_type"):
        osc_lines.append(f"Oscillator 1 is frequency-modulated by oscillator 2 at {v['fm_amount']:.0%} depth.")
    if st.get("sample_on"):
        osc_lines.append(f"White noise at level {v['noise_level']:.0%}.")
    out.append({"title": "Oscillators", "lines": osc_lines})

    if st.get("filter_1_on"):
        slope = "24 dB/oct" if float(st.get("filter_1_style", 0)) >= 1 else "12 dB/oct"
        fl = [f"Low-pass filter ({slope}), cutoff {_hz(v['filter_cutoff'])}, resonance {v['filter_resonance']:.0%}"
              + (f", drive {v['filter_drive']:.0f} dB" if v["filter_drive"] >= 0.5 else "") + "."]
        if v["filter_env_amount"] >= 0.5:
            fl.append(f"Filter envelope opens the cutoff by {v['filter_env_amount']:.0f} semitones: attack "
                      f"{_ms(v['filt_attack'])}, decay {_ms(v['filt_decay'])}, sustain {v['filt_sustain']:.0%}, "
                      f"release {_ms(v['filt_release'])}.")
        if v["filter_lfo_depth"] >= 0.5:
            fl.append(f"An LFO moves the cutoff by ±{v['filter_lfo_depth'] / 2:.0f} semitones at {v['filter_lfo_rate']:.2f} Hz.")
        out.append({"title": "Filter", "lines": fl})

    out.append({"title": "Amp envelope", "lines": [
        f"Attack {_ms(v['amp_attack'])}, decay {_ms(v['amp_decay'])}, sustain {v['amp_sustain']:.0%}, "
        f"release {_ms(v['amp_release'])}."]})

    mod = []
    if abs(v["pitch_env_amount"]) >= 0.3:
        mod.append(f"Pitch envelope on all oscillators: jumps {v['pitch_env_amount']:+.1f} semitones over "
                   f"{_ms(v['pitch_env_attack'])}, then falls back to the played pitch over {_ms(v['pitch_env_decay'])}"
                   " (sustain 0).")
    if v["vibrato_depth"] >= 1:
        mod.append(f"Vibrato: ±{v['vibrato_depth'] / 2:.0f} cents at {v['vibrato_rate']:.1f} Hz.")
    if v["tremolo_depth"] >= 0.02:
        mod.append(f"Tremolo: {v['tremolo_depth']:.0%} depth at {v['tremolo_rate']:.1f} Hz.")
    if mod:
        out.append({"title": "Modulation", "lines": mod})
    return out


def recipe_text(sections: list[dict[str, Any]]) -> str:
    return "\n".join(f"{s['title']}: " + " ".join(s["lines"]) for s in sections)

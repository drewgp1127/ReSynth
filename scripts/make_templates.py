"""Generate ReSynth's starting templates from Vital's init patch.

Run: .venv/bin/python scripts/make_templates.py
Writes app/templates/<name>.vital and app/templates/manifest.json. Deterministic: running it
again produces identical files. Uses no factory presets; every wave is generated here.

Values in `values` are natural units from app/synth/params.py; `settings` are raw Vital controls.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import vita

from app.synth.builder import SHAPE_FRAMES, base_patch, wavetable
from app.synth.params import PARAMS
from app.synth.vital import apply_to_patch

OUT = Path(__file__).resolve().parent.parent / "app" / "templates"
SAW, SQUARE, TRI, SINE = (SHAPE_FRAMES[k] / 255 for k in ("saw", "square", "triangle", "sine"))

COMMON_SEARCH = ["filter_cutoff", "filter_resonance", "amp_attack", "amp_decay", "amp_sustain", "amp_release"]


def tri_saw_table() -> dict[str, Any]:
    """Triangle (frame 0) -> skewed 'tri-saw' (128) -> saw (255)."""
    n = np.arange(2048) / 2048

    def skewed(peak: float) -> np.ndarray:  # rises over `peak` of the cycle, falls over the rest
        up = n < peak
        return np.where(up, -1 + 2 * n / peak, 1 - 2 * (n - peak) / (1 - peak))

    return wavetable("Tri-Saw", [(0, skewed(0.5)), (128, skewed(0.85)), (255, skewed(0.999))])


RECIPES: list[dict[str, Any]] = [
    {
        "name": "saw_lead", "description": "single saw, gently filtered, short filter envelope",
        "values": {"osc1_wave": SAW, "filter_cutoff": 3500.0, "filter_resonance": 0.2, "filter_env_amount": 12.0,
                   "filt_decay": 0.3, "amp_attack": 0.005, "amp_decay": 0.3, "amp_sustain": 0.8, "amp_release": 0.15},
        "search": COMMON_SEARCH + ["osc1_wave", "filter_env_amount", "filt_decay", "filter_drive", "vibrato_rate", "vibrato_depth"],
    },
    {
        "name": "supersaw", "description": "two 7-voice detuned saw stacks, open filter",
        "values": {"osc1_wave": SAW, "osc1_unison_voices": 7, "osc1_unison_detune": 35.0, "osc2_wave": SAW,
                   "osc2_level": 0.6, "osc2_unison_voices": 7, "osc2_unison_detune": 22.0, "filter_cutoff": 9000.0,
                   "amp_attack": 0.01, "amp_sustain": 1.0, "amp_release": 0.3},
        "settings": {"osc_2_on": 1.0},
        "search": COMMON_SEARCH + ["osc1_unison_detune", "osc2_unison_detune", "osc2_level", "osc_detune_cents"],
    },
    {
        "name": "square_lead", "description": "square plus sub square an octave down, light vibrato",
        "values": {"osc1_wave": SQUARE, "osc2_wave": SQUARE, "osc2_level": 0.4, "filter_cutoff": 3000.0,
                   "amp_attack": 0.005, "amp_sustain": 0.9, "amp_release": 0.12, "vibrato_rate": 5.5, "vibrato_depth": 15.0},
        "settings": {"osc_2_on": 1.0, "osc_2_transpose": -12.0},
        "search": COMMON_SEARCH + ["osc1_wave", "osc2_level", "vibrato_rate", "vibrato_depth", "filter_drive"],
    },
    {
        "name": "pluck", "description": "saw + square through a closed filter opened by a fast envelope",
        "values": {"osc1_wave": SAW, "osc2_wave": SQUARE, "osc2_level": 0.5, "filter_cutoff": 400.0,
                   "filter_resonance": 0.25, "filter_env_amount": 50.0, "filt_decay": 0.25,
                   "amp_attack": 0.001, "amp_decay": 0.6, "amp_sustain": 0.0, "amp_release": 0.3},
        "settings": {"osc_2_on": 1.0},
        "search": COMMON_SEARCH + ["filter_env_amount", "filt_decay", "osc2_level", "osc_detune_cents"],
    },
    {
        "name": "pad", "description": "slow saw + triangle unison stacks, filter LFO",
        "values": {"osc1_wave": SAW, "osc1_unison_voices": 4, "osc1_unison_detune": 20.0, "osc2_wave": TRI,
                   "osc2_level": 0.7, "osc2_unison_voices": 4, "osc2_unison_detune": 15.0, "filter_cutoff": 1500.0,
                   "filter_resonance": 0.15, "amp_attack": 0.6, "amp_decay": 1.0, "amp_sustain": 0.8,
                   "amp_release": 1.2, "filter_lfo_rate": 0.3, "filter_lfo_depth": 6.0},
        "settings": {"osc_2_on": 1.0},
        "search": COMMON_SEARCH + ["osc1_unison_detune", "osc2_level", "filter_lfo_rate", "filter_lfo_depth"],
    },
    {
        "name": "organ", "description": "three sines at 8', 4' and a fifth above, fast on/off",
        "values": {"osc1_wave": SINE, "osc2_wave": SINE, "osc2_level": 0.6, "osc3_wave": SINE, "osc3_level": 0.4,
                   "filter_cutoff": 16000.0, "amp_attack": 0.004, "amp_sustain": 1.0, "amp_release": 0.05,
                   "vibrato_rate": 6.5, "vibrato_depth": 8.0},
        "settings": {"osc_2_on": 1.0, "osc_2_transpose": 12.0, "osc_3_on": 1.0, "osc_3_transpose": 19.0},
        "search": ["osc2_level", "osc3_level", "osc1_wave", "amp_attack", "amp_release", "vibrato_rate",
                   "vibrato_depth", "filter_cutoff", "filter_drive"],
    },
    {
        "name": "fm_bell", "description": "sine carrier frequency-modulated by a silent sine at ratio 3.5",
        "values": {"osc1_wave": SINE, "osc2_wave": SINE, "osc2_level": 0.0, "fm_amount": 0.4, "filter_cutoff": 16000.0,
                   "amp_attack": 0.001, "amp_decay": 2.0, "amp_sustain": 0.0, "amp_release": 1.5},
        "settings": {"osc_2_on": 1.0, "osc_2_transpose": 21.0, "osc_2_tune": 0.69, "osc_1_distortion_type": 7.0},
        "search": ["fm_amount", "amp_attack", "amp_decay", "amp_sustain", "amp_release", "filter_cutoff"],
    },
    {
        "name": "bass", "description": "saw + sub square, resonant low-pass with envelope, some drive",
        "values": {"osc1_wave": SAW, "osc2_wave": SQUARE, "osc2_level": 0.6, "filter_cutoff": 300.0,
                   "filter_resonance": 0.3, "filter_env_amount": 36.0, "filt_decay": 0.2, "filter_drive": 6.0,
                   "amp_attack": 0.002, "amp_decay": 0.3, "amp_sustain": 0.8, "amp_release": 0.08},
        "settings": {"osc_2_on": 1.0, "osc_2_transpose": -12.0},
        "search": COMMON_SEARCH + ["filter_env_amount", "filt_decay", "filter_drive", "osc2_level", "osc1_wave"],
    },
    {
        "name": "pitch_blip_keys",
        "description": "two slightly detuned tri-saw oscillators with a fast up-then-down pitch blip on each note "
                       "(the 'Lemon Glow' type); osc wave here sweeps triangle -> tri-saw -> saw",
        "values": {"osc1_wave": 128 / 255, "osc2_wave": 128 / 255, "osc2_level": 0.7, "osc1_level": 0.7,
                   "osc_detune_cents": 12.0, "pitch_env_amount": 5.0, "pitch_env_attack": 0.008,
                   "pitch_env_decay": 0.06, "filter_cutoff": 12000.0, "filter_drive": 4.0,
                   "amp_attack": 0.003, "amp_decay": 0.8, "amp_sustain": 0.6, "amp_release": 0.4},
        "settings": {"osc_2_on": 1.0},
        "tables": {0: "tri_saw", 1: "tri_saw"},
        "search": COMMON_SEARCH + ["osc1_wave", "osc_detune_cents", "pitch_env_amount", "pitch_env_attack",
                                   "pitch_env_decay", "filter_drive"],
    },
    {
        "name": "noise_sweep", "description": "white noise through a 24 dB low-pass swept by a slow filter envelope",
        "values": {"osc1_level": 0.0, "noise_level": 0.7, "filter_cutoff": 200.0, "filter_resonance": 0.4,
                   "filter_env_amount": 72.0, "filt_attack": 0.8, "filt_decay": 1.5, "filt_sustain": 0.0,
                   "amp_attack": 0.2, "amp_sustain": 1.0, "amp_release": 1.0},
        "settings": {"sample_on": 1.0, "filter_1_style": 1.0},
        "search": COMMON_SEARCH + ["filter_env_amount", "filt_attack", "filt_decay", "filter_lfo_rate", "filter_lfo_depth"],
    },
    {
        "name": "brass", "description": "two saws, filter envelope with slow attack, light vibrato",
        "values": {"osc1_wave": SAW, "osc1_unison_voices": 2, "osc1_unison_detune": 10.0, "osc2_wave": SAW,
                   "osc2_level": 0.8, "osc_detune_cents": 8.0, "filter_cutoff": 600.0, "filter_env_amount": 40.0,
                   "filt_attack": 0.08, "filt_decay": 0.4, "filt_sustain": 0.5, "amp_attack": 0.05,
                   "amp_decay": 0.2, "amp_sustain": 0.85, "amp_release": 0.2, "vibrato_rate": 5.0, "vibrato_depth": 10.0},
        "settings": {"osc_2_on": 1.0},
        "search": COMMON_SEARCH + ["filter_env_amount", "filt_attack", "filt_decay", "filt_sustain", "osc_detune_cents"],
    },
    {
        "name": "strings", "description": "two 5-voice saw stacks, slow attack and release, vibrato",
        "values": {"osc1_wave": SAW, "osc1_unison_voices": 5, "osc1_unison_detune": 25.0, "osc2_wave": SAW,
                   "osc2_level": 0.7, "osc2_unison_voices": 5, "osc2_unison_detune": 18.0, "filter_cutoff": 3000.0,
                   "filter_resonance": 0.1, "amp_attack": 0.35, "amp_decay": 0.5, "amp_sustain": 0.9,
                   "amp_release": 0.6, "vibrato_rate": 5.5, "vibrato_depth": 12.0},
        "settings": {"osc_2_on": 1.0},
        "search": COMMON_SEARCH + ["osc1_unison_detune", "osc2_level", "vibrato_rate", "vibrato_depth"],
    },
]

TABLES = {"tri_saw": tri_saw_table}


def build(recipe: dict[str, Any]) -> dict[str, Any]:
    patch = base_patch(recipe["name"])
    patch["comments"] = f"ReSynth template: {recipe['description']}. Generated from Vital's init patch."
    for idx, table in recipe.get("tables", {}).items():
        patch["settings"]["wavetables"][idx] = TABLES[table]()
    patch["settings"].update(recipe.get("settings", {}))
    raw: dict[str, float] = {}
    for name, value in recipe["values"].items():
        raw.update(PARAMS[name].to_raw(value))
    patch = apply_to_patch(patch, raw)
    synth = vita.Synth()  # canonicalise through Vital's own serializer
    if not synth.load_json(json.dumps(patch)):
        raise RuntimeError(f"vita rejected template {recipe['name']}")
    return json.loads(synth.to_json())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for recipe in RECIPES:
        patch = build(recipe)
        (OUT / f"{recipe['name']}.vital").write_text(json.dumps(patch, indent=1))
        manifest[recipe["name"]] = {"description": recipe["description"], "search": recipe["search"]}
        print(f"wrote {recipe['name']}.vital")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

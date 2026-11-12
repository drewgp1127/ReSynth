"""End-to-end check without the Claude stages.

Renders the pitch-blip template with hidden parameter values playing chords, mixes it under a
synthetic drum + bass bed, then runs isolate -> analyse -> plan (rule-based, as without an API
key) -> match, and writes target, match and a loudness-matched A/B to an output folder, plus
e2e_summary.json with the numbers.

Run: .venv/bin/python -m tests.e2e_pipeline [out_dir] [--budget SECONDS]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from app.stages.analyse import run_analyse
from app.stages.common import SR, make_ab, read_audio, read_json, write_audio, write_json
from app.stages.isolate import choose_target, run_isolate
from app.stages.match import TEMPLATES_DIR, run_match, template_values
from app.stages.plan_rules import rule_plan
from app.synth.loss import LossFn
from app.synth.params import SearchSpace
from app.synth.vital import Note, load_template, render_patch
from tests.audio_fixtures import drum_bass_bed

TEMPLATE = "pitch_blip_keys"
TRUTH = {"osc_detune_cents": 18.0, "pitch_env_amount": 7.0, "pitch_env_decay": 0.05, "filter_cutoff": 5000.0,
         "amp_attack": 0.004, "amp_decay": 0.5, "amp_sustain": 0.45, "amp_release": 0.25, "osc1_wave": 0.6,
         "filter_drive": 8.0}
CHORDS = [(0.0, (53, 57, 60, 64)), (0.6, (55, 59, 62, 65)), (1.2, (52, 55, 59, 64)), (1.8, (53, 57, 60, 65))]
NOTE_LEN = 0.45
SYNTH_AT, REGION = 2.0, (2.0, 4.6)
PAD = 0.05


def truth_notes(offset: float = 0.0) -> list[Note]:
    return [Note(m, t + offset, NOTE_LEN) for t, chord in CHORDS for m in chord]


def build_song(out: Path) -> tuple[np.ndarray, np.ndarray]:
    patch = load_template(TEMPLATES_DIR / f"{TEMPLATE}.vital")
    synth = render_patch(patch, truth_notes(), 3.0, SearchSpace.build(list(TRUTH)).raw(TRUTH))
    bed = drum_bass_bed(8.0)
    gain = 0.8 * np.sqrt(np.mean(bed ** 2)) / np.sqrt(np.mean(synth ** 2))  # synth a bit under the bed
    song = bed.copy()
    a = int(SYNTH_AT * SR)
    song[:, a:a + synth.shape[1]] += gain * synth
    song /= max(1.0, float(np.abs(song).max()) / 0.98)
    write_audio(out / "song.wav", song)
    return song, synth


def run_e2e(out: Path, budget_s: float = 180.0) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    _, synth = build_song(out)
    region_len = REGION[1] - REGION[0] + 2 * PAD
    clean = synth[:, :int((REGION[1] - SYNTH_AT + PAD) * SR)]
    clean = np.pad(clean, ((0, 0), (int(PAD * SR), 0)))[:, :int(region_len * SR)]
    write_audio(out / "clean_truth.wav", clean)

    iso = run_isolate(out, "song.wav", *REGION, padding_s=PAD)
    chosen = iso["suggested"]
    choose_target(out, chosen)
    analysis = run_analyse(out)

    # the plan stage as the app runs it without an API key (rule-based, from the analysis)
    manifest = json.loads((TEMPLATES_DIR / "manifest.json").read_text())
    plan = rule_plan(analysis, "the pulsing main synth chords with a quick pitch blip on each chord", manifest)
    write_json(out / "plan.json", plan)
    params = plan["params"]
    match = run_match(out, budget_s=budget_s)

    target = read_audio(out / "target.wav")
    best = read_audio(out / "match_1.wav")
    write_audio(out / "ab.wav", make_ab(target, best))

    patch = load_template(TEMPLATES_DIR / f"{TEMPLATE}.vital")
    notes = [Note.from_dict(n) for n in analysis["notes"]]
    default_render = render_patch(patch, notes, target.shape[1] / SR)
    onsets = sorted({n.onset for n in notes})
    vs_clean = LossFn(clean, onsets=onsets)
    found = match["best"][0]["values"]
    start = template_values(patch, params)
    truth_found = sorted((n.midi, round(n.onset, 2)) for n in truth_notes(PAD))
    detected = sorted((n["midi"], round(n["onset"], 2)) for n in analysis["notes"])
    summary = {
        "isolate": {"status": iso["status"], "chosen_stem": chosen, "rms": iso.get("rms")},
        "analysis": {
            "note_method": analysis["note_method"],
            "notes_truth": len(truth_found), "notes_detected": len(detected),
            "pitch_classes_match": sorted({m % 12 for m, _ in detected}) == sorted({m % 12 for m, _ in truth_found}),
            "detected_notes": detected,
            "pitch_blip_detected": analysis["pitch_movement"]["detected"],
            "pitch_blip_peak_semitones": analysis["pitch_movement"]["peak_semitones"],
            "beating": analysis["beating"],
            "attack_s": analysis["envelope"]["attack_s"],
        },
        "match": {
            "loss_vs_target_initial": round(match["initial_loss"], 4),
            "loss_vs_target_final": round(match["best"][0]["loss"], 4),
            "evals": match["evals"], "evals_per_s": match["evals_per_s"], "stop_reason": match["stop_reason"],
            "loss_vs_clean_synth_template_default": round(vs_clean(default_render)[0], 4),
            "loss_vs_clean_synth_match": round(vs_clean(best)[0], 4),
            "loss_vs_clean_synth_truth_params_with_detected_notes": round(vs_clean(render_patch(
                patch, notes, target.shape[1] / SR, SearchSpace.build(list(TRUTH)).raw(TRUTH)))[0], 4),
        },
        "plan": {"template": plan["template"], "params": params, "ranges": plan["ranges"], "initial": plan["initial"]},
        "params": {k: {"start": round(start.get(k, math.nan), 4), "truth": TRUTH.get(k),
                       "found": round(v, 4)} for k, v in found.items()},
        "files": ["song.wav", "clean_truth.wav", "target.wav", "match_1.wav", "ab.wav", "match_1.vital"],
    }
    write_json(out / "e2e_summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=str(Path(__file__).resolve().parent.parent / "scratch" / "e2e"))
    ap.add_argument("--budget", type=float, default=180.0)
    a = ap.parse_args()
    print(json.dumps(run_e2e(Path(a.out), a.budget), indent=1))


if __name__ == "__main__":
    main()

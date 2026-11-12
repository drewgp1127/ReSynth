"""Synthetic recovery: render with hidden values, search from template defaults, check recovery."""
import json
import math

import numpy as np
import pytest

from app.stages.common import write_audio
from app.stages.match import MatchConfig, run_match, search
from app.synth.params import SearchSpace
from app.synth.vital import render_patch, save_vital

DUR = 1.2


def semitones(a: float, b: float) -> float:
    return abs(12 * math.log2(a / b))


def test_recovery_short(patch, chords):
    names = ["filter_cutoff", "amp_attack", "amp_release", "osc_detune_cents"]
    truth = {"filter_cutoff": 900.0, "amp_attack": 0.04, "amp_release": 0.25, "osc_detune_cents": 14.0}
    target = render_patch(patch, chords, DUR, SearchSpace.build(names).raw(truth))
    res = search(patch, target, chords, DUR, MatchConfig(template="test", params=names, budget_s=35))
    best = res["best"][0]
    print("initial", res["initial_loss"], "best", best["loss"], "evals", res["evals"], "evals/s", res["evals_per_s"],
          "values", best["values"])
    assert best["loss"] < 0.3 * res["initial_loss"]
    v = best["values"]
    assert semitones(v["filter_cutoff"], truth["filter_cutoff"]) < 3
    assert 0.5 < v["amp_attack"] / truth["amp_attack"] < 2
    assert abs(v["osc_detune_cents"] - truth["osc_detune_cents"]) < 4


@pytest.mark.slow
def test_recovery_eight_params(patch, chords):
    names = ["filter_cutoff", "filter_resonance", "amp_attack", "amp_release", "osc1_unison_detune",
             "osc_detune_cents", "filter_env_amount", "filt_decay"]
    truth = {"filter_cutoff": 900.0, "filter_resonance": 0.45, "amp_attack": 0.04, "amp_release": 0.25,
             "osc1_unison_detune": 25.0, "osc_detune_cents": 14.0, "filter_env_amount": 30.0, "filt_decay": 0.15}
    target = render_patch(patch, chords, DUR, SearchSpace.build(names).raw(truth))
    res = search(patch, target, chords, DUR, MatchConfig(template="test", params=names, budget_s=120))
    best = res["best"][0]
    print("initial", res["initial_loss"], "best", best["loss"], "evals", res["evals"], "evals/s", res["evals_per_s"],
          "values", best["values"])
    assert best["loss"] < 0.3 * res["initial_loss"]
    v = best["values"]
    assert semitones(v["filter_cutoff"], truth["filter_cutoff"]) < 4
    assert 0.5 < v["amp_attack"] / truth["amp_attack"] < 2
    assert abs(v["osc_detune_cents"] - truth["osc_detune_cents"]) < 4
    assert abs(v["osc1_unison_detune"] - truth["osc1_unison_detune"]) < 8


def test_match_stage_job_folder(patch, chords, tmp_path):
    tpl = save_vital(patch, tmp_path / "tpl.vital")
    names = ["filter_cutoff", "amp_release"]
    truth = {"filter_cutoff": 1500.0, "amp_release": 0.2}
    write_audio(tmp_path / "target.wav", render_patch(patch, chords, DUR, SearchSpace.build(names).raw(truth)))
    (tmp_path / "analysis.json").write_text(json.dumps({
        "duration": DUR,
        "notes": [{"midi": n.midi, "onset": n.onset, "duration": n.duration, "velocity": n.velocity} for n in chords],
    }))
    (tmp_path / "plan.json").write_text(json.dumps({"template": str(tpl), "params": names}))
    res = run_match(tmp_path, budget_s=8, workers=2)
    out = json.loads((tmp_path / "match.json").read_text())
    assert out["best"][0]["loss"] <= out["initial_loss"]
    assert len(out["best"]) >= 1 and (tmp_path / "match_1.vital").exists() and (tmp_path / "match_1.wav").exists()
    json.loads((tmp_path / "match_1.vital").read_text())
    status = json.loads((tmp_path / "status" / "match.json").read_text())
    assert status["state"] == "done" and status["progress"] == 1.0
    assert res["evals"] > 0


TEMPLATE_CASES = {
    "pluck": {"filter_cutoff": 700.0, "filter_env_amount": 35.0, "filt_decay": 0.12, "amp_decay": 0.35, "amp_attack": 0.01},
    "pitch_blip_keys": {"pitch_env_amount": 8.0, "pitch_env_decay": 0.04, "osc_detune_cents": 20.0,
                        "filter_cutoff": 5000.0, "amp_attack": 0.01},
    "pad": {"amp_attack": 0.25, "filter_cutoff": 900.0, "osc1_unison_detune": 30.0, "filter_lfo_rate": 0.6,
            "amp_release": 0.5},
}


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(TEMPLATE_CASES))
def test_recovery_on_templates(name, chords):
    from app.stages.match import TEMPLATES_DIR
    from app.synth.vital import load_template

    patch = load_template(TEMPLATES_DIR / f"{name}.vital")
    truth = TEMPLATE_CASES[name]
    names = list(truth)
    dur = 1.6
    target = render_patch(patch, chords, dur, SearchSpace.build(names).raw(truth))
    res = search(patch, target, chords, dur, MatchConfig(template=name, params=names, budget_s=60))
    best = res["best"][0]
    print(name, "initial", res["initial_loss"], "best", best["loss"], "evals", res["evals"],
          "values", {k: round(v, 4) for k, v in best["values"].items()})
    assert best["loss"] < 0.3 * res["initial_loss"]
    v = best["values"]
    assert semitones(v["filter_cutoff"], truth["filter_cutoff"]) < 4
    assert 0.5 < v["amp_attack"] / truth["amp_attack"] < 2
    if "osc_detune_cents" in truth:
        assert abs(v["osc_detune_cents"] - truth["osc_detune_cents"]) < 4
    if "pitch_env_amount" in truth:
        assert abs(v["pitch_env_amount"] - truth["pitch_env_amount"]) < 2


def test_select_window_picks_whole_chords():
    from app.stages.match import select_window
    from app.synth.vital import Note

    chords = [Note(m, t, 0.45) for t in (0.05, 0.65, 1.25, 1.85) for m in (53, 57, 60, 64)]
    span = select_window(chords, 2.7, 1.3)
    assert span is not None
    a, b = span
    assert a <= 0.05 and abs(b - 1.25) < 1e-9  # two whole chords, ends at the third onset
    assert select_window(chords, 1.4, 1.3) is None  # short clip: search on all of it
    # a partly detected first chord is skipped in favour of a complete one
    partial = [n for n in chords if not (n.onset == 0.05 and n.midi != 53)]
    a, _ = select_window(partial, 2.7, 1.3)
    assert abs(a - 0.62) < 1e-9


def test_crop_context_shifts_notes(chords):
    from app.stages.match import crop_context
    from app.synth.loss import LossWeights

    target = np.zeros((2, int(1.2 * 44100)), np.float32)
    target[:, 1000:] = 0.1
    ctx = crop_context(target, chords, (0.55, 1.2), LossWeights())
    assert [n.onset for n in ctx.notes] == pytest.approx([0.05, 0.05, 0.05])
    assert abs(ctx.duration - 0.65) < 1e-3

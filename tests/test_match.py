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

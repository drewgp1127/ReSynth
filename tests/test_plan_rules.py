"""Rule-based plan from analysis.json (no API)."""
import json
from pathlib import Path

from app.stages.plan_rules import analysis_priors, rule_plan

MANIFEST = json.loads((Path(__file__).resolve().parent.parent / "app" / "templates" / "manifest.json").read_text())


def blip_analysis() -> dict:
    return {
        "duration": 2.6,
        "notes": [{"midi": m, "onset": t, "duration": 0.45, "velocity": 0.8} for t in (0.05, 0.65)
                  for m in (53, 57, 60, 64)],
        "envelope": {"attack_s": 0.03, "release_s": 0.6},
        "spectral": {"centroid_hz_mean": 900, "rolloff_hz_mean": 1500, "flatness_mean": 0.001, "odd_even_ratio": 0.3},
        "pitch_movement": {"detected": True, "peak_semitones": 2.0, "settle_time_s": 0.02, "blip_count": 2,
                           "group_count": 2},
        "beating": {"rate_hz": None, "detune_cents_estimate": None, "confirmed": False},
        "stereo": {"width": 0.0},
    }


def test_rule_plan_uses_analysis():
    plan = rule_plan(blip_analysis(), "", MANIFEST)
    assert plan["template"] == "pitch_blip_keys"
    assert len(plan["params"]) <= 10 and "pitch_env_amount" in plan["params"] and "filter_drive" in plan["params"]
    lo, hi = plan["ranges"]["pitch_env_amount"]
    assert lo == 2.0 and hi == 24.0 and plan["initial"]["pitch_env_amount"] == 6.0
    assert plan["ranges"]["amp_attack"][1] <= 0.08 and "amp_release" not in plan["ranges"]


def test_rule_plan_without_blip_fixes_pitch_env_off():
    a = blip_analysis()
    a["pitch_movement"] = {"detected": False, "peak_semitones": 0.0}
    plan = rule_plan(a, "a warm pad", MANIFEST)
    assert plan["template"] == "pad"
    assert "pitch_env_amount" not in plan["params"] and plan["fixed"]["pitch_env_amount"] == 0.0


def test_priors_detune_from_beating():
    a = blip_analysis()
    a["beating"] = {"rate_hz": 1.5, "detune_cents_estimate": 12.0, "confirmed": True}
    p = analysis_priors(a, ["osc_detune_cents"])
    assert p["initial"]["osc_detune_cents"] == 12.0 and p["ranges"]["osc_detune_cents"] == [6.0, 24.0]



"""Stage 4: choose a template and the search space for the match stage.

Inputs:  <job>/analysis.json (required), <job>/research.json (optional), the user's description,
         app/templates/manifest.json
Output:  <job>/plan.json, <job>/status/plan.json

Claude picks one of the shipped templates and edits it: which parameters to search (at most
MAX_PARAMS), their ranges and starting values. It never writes a preset from scratch. Its answer
is validated against `app/synth/params.py` and the manifest; anything invalid (unknown template
or parameter, empty or out-of-bounds ranges) falls back to the rule-based plan in
`plan_rules.py`, which also runs when there is no API key. Ranges and starting values Claude did
not give are filled from the analysis (`analysis_priors`).

Run: python -m app.stages.plan <job_dir> [--description ...] [--offline]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.stages.claude import ClaudeUnavailable, ask_json, get_client
from app.stages.common import read_json, write_json, write_status
from app.stages.plan_rules import DEFAULT_WEIGHTS, MAX_PARAMS, analysis_priors, rule_plan
from app.synth.params import PARAMS

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def load_manifest() -> dict[str, Any]:
    return json.loads((TEMPLATES_DIR / "manifest.json").read_text())


class InvalidPlan(ValueError):
    pass


# Measured under a mix, these read low: stem separation and masking remove highs, and a pitch
# blip's measured depth is a lower bound. Claude may narrow the near side; the far side stays open.
OPEN_FAR_SIDE = ("filter_cutoff", "pitch_env_amount")


def open_far_side(name: str, lo: float, hi: float) -> tuple[float, float]:
    """Widen a range of an under-measured parameter to the parameter's limit away from zero."""
    p = PARAMS[name]
    if name not in OPEN_FAR_SIDE:
        return lo, hi
    if hi > 0:
        hi = p.hi
    if lo < 0:
        lo = p.lo
    return lo, hi


def analysis_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    """The numbers Claude needs (drops the long per-frame tracks)."""
    pm = analysis.get("pitch_movement") or {}
    return {
        "duration_s": analysis.get("duration"),
        "notes": [{k: n[k] for k in ("midi", "onset", "duration")} for n in analysis.get("notes", [])][:24],
        "envelope": {k: (analysis.get("envelope") or {}).get(k) for k in ("attack_s", "release_s")},
        "spectral": {k: v for k, v in (analysis.get("spectral") or {}).items() if not k.endswith("_10ms") and k != "hop_s"},
        "pitch_movement": {k: pm.get(k) for k in ("detected", "blip_count", "group_count", "peak_semitones",
                                                  "settle_time_s")},
        "pitch_movement_note": "peak_semitones under-reads fast blips (roughly a third of the true peak); treat it as a lower bound",
        "vibrato": analysis.get("vibrato"), "tremolo": analysis.get("tremolo"),
        "filter_lfo": analysis.get("filter_lfo"), "beating": analysis.get("beating"), "stereo": analysis.get("stereo"),
    }


def plan_schema(manifest: dict[str, Any]) -> dict[str, Any]:
    names = sorted(PARAMS)
    return {
        "type": "object", "additionalProperties": False,
        "required": ["template", "params", "ranges", "initial", "reasons"],
        "properties": {
            "template": {"type": "string", "enum": sorted(manifest)},
            "params": {"type": "array", "items": {"type": "string", "enum": names}},
            "ranges": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["param", "lo", "hi"],
                "properties": {"param": {"type": "string", "enum": names}, "lo": {"type": "number"},
                               "hi": {"type": "number"}}}},
            "initial": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["param", "value"],
                "properties": {"param": {"type": "string", "enum": names}, "value": {"type": "number"}}}},
            "reasons": {"type": "array", "items": {"type": "string"}},
        },
    }


SYSTEM = f"""You plan a render-and-compare search that matches a synth sound with a Vital preset.
You choose ONE of the provided templates (each is a complete starting patch) and decide which of
its parameters the search should adjust, with ranges and starting values in the parameters' own
units. You never design a patch from scratch: anything you do not search keeps the template's value.

Rules:
- Search at most {MAX_PARAMS} parameters: the ones that most change how this sound differs from the
  template. Ranges must lie inside each parameter's bounds; narrow them when the analysis or
  research justifies it, otherwise leave the full bounds.
- Filter cutoff and pitch-envelope depth measured from a separated stem read LOW (separation and
  the mix remove highs; a blip's measured depth is a lower bound). For these two, set only the
  lower bound from the analysis; the upper bound is always kept at the parameter's limit.
- Use the analysis numbers as evidence. A recreation from the research describes the type of
  sound, not the original gear; use it only as a hint for the template and ranges.
- Give a short reason for each choice."""


def build_prompt(analysis: dict[str, Any], research: dict[str, Any] | None, description: str,
                 manifest: dict[str, Any]) -> str:
    params = {n: {"lo": p.lo, "hi": p.hi, "unit": p.unit, "what": p.description} for n, p in PARAMS.items()}
    templates = {k: {"description": v["description"], "default_search": v["search"]} for k, v in manifest.items()}
    res = None
    if research and research.get("status") == "done":
        res = {k: research.get(k) for k in ("model_statement", "instruments", "synthesis", "recreations", "summary")}
    return json.dumps({"user_description": description, "analysis": analysis_summary(analysis), "research": res,
                       "templates": templates, "parameters": params}, indent=1)


def validate(raw: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Check Claude's plan against the manifest and params.py; return a normalised plan."""
    tpl = raw.get("template")
    if tpl not in manifest:
        raise InvalidPlan(f"unknown template {tpl!r}")
    params = [p for p in dict.fromkeys(raw.get("params") or [])]
    unknown = [p for p in params if p not in PARAMS]
    if unknown:
        raise InvalidPlan(f"unknown parameters {unknown}")
    if not params:
        raise InvalidPlan("no parameters to search")
    if len(params) > MAX_PARAMS + 2:
        raise InvalidPlan(f"{len(params)} parameters (limit {MAX_PARAMS})")
    params = params[:MAX_PARAMS]
    ranges: dict[str, list[float]] = {}
    for r in raw.get("ranges") or []:
        name = r.get("param")
        if name not in params:
            continue
        p = PARAMS[name]
        lo, hi = float(r["lo"]), float(r["hi"])
        lo, hi = max(min(lo, hi), p.lo), min(max(lo, hi), p.hi)
        if hi <= lo or (p.log and lo <= 0):
            raise InvalidPlan(f"range for {name} is empty or outside {p.lo}..{p.hi}")
        ranges[name] = list(open_far_side(name, lo, hi))
    initial: dict[str, float] = {}
    for i in raw.get("initial") or []:
        name = i.get("param")
        if name not in params:
            continue
        lo, hi = ranges.get(name, [PARAMS[name].lo, PARAMS[name].hi])
        initial[name] = float(min(max(float(i["value"]), lo), hi))
    return {"template": tpl, "params": params, "ranges": ranges, "initial": initial,
            "reasons": [str(x) for x in raw.get("reasons") or []]}


def merge_priors(plan: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    """Fill ranges / starting values Claude left out with the analysis-based ones."""
    pri = analysis_priors(analysis, plan["params"])
    for name, rng in pri["ranges"].items():
        plan["ranges"].setdefault(name, rng)
    for name, v in pri["initial"].items():
        lo, hi = plan["ranges"].get(name, [PARAMS[name].lo, PARAMS[name].hi])
        if name not in plan["initial"] and lo <= v <= hi:
            plan["initial"][name] = v
    plan.setdefault("fixed", {})
    if "pitch_env_amount" not in plan["params"] and not (analysis.get("pitch_movement") or {}).get("detected"):
        plan["fixed"]["pitch_env_amount"] = 0.0
    plan.setdefault("weights", dict(DEFAULT_WEIGHTS))
    return plan


def make_plan(analysis: dict[str, Any], research: dict[str, Any] | None, description: str,
              manifest: dict[str, Any] | None = None, client: Any = None, offline: bool = False) -> dict[str, Any]:
    manifest = manifest or load_manifest()
    if offline:
        return {**rule_plan(analysis, description, manifest), "fallback_reason": "offline"}
    try:
        client = client or get_client()
        raw = ask_json(client, SYSTEM, build_prompt(analysis, research, description, manifest), plan_schema(manifest))
        plan = merge_priors(validate(raw, manifest), analysis)
        plan["source"] = "claude"
        return plan
    except (ClaudeUnavailable, InvalidPlan, KeyError, TypeError, ValueError) as e:
        return {**rule_plan(analysis, description, manifest), "fallback_reason": str(e)}


def run_plan(job_dir: str | Path, description: str = "", client: Any = None, offline: bool = False) -> dict[str, Any]:
    job = Path(job_dir)
    write_status(job, "plan", "running", 0.2, "choosing a template")
    try:
        analysis = read_json(job / "analysis.json")
        research = read_json(job / "research.json") if (job / "research.json").exists() else None
        plan = make_plan(analysis, research, description, client=client, offline=offline)
        write_json(job / "plan.json", plan)
        msg = f"template {plan['template']}, {len(plan['params'])} parameters ({plan['source']})"
        if plan.get("fallback_reason"):
            msg += f"; rule-based because: {plan['fallback_reason']}"
        write_status(job, "plan", "done", 1.0, msg)
        return plan
    except Exception as e:
        write_status(job, "plan", "failed", 0.0, f"{type(e).__name__}: {e}")
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description="ReSynth stage 4: plan the search")
    ap.add_argument("job_dir")
    ap.add_argument("--description", default="")
    ap.add_argument("--offline", action="store_true", help="rule-based plan, no API call")
    a = ap.parse_args()
    print(json.dumps(run_plan(a.job_dir, a.description, offline=a.offline), indent=1))


if __name__ == "__main__":
    main()

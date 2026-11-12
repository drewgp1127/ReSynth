"""Deterministic search plan from analysis.json alone (no API key needed).

Used as the plan stage's fallback and to fill in analysis-based starting values and ranges for
a plan Claude wrote. Everything here is a rule over the analysis numbers; nothing is learned.

`analysis_priors(analysis, params)` gives starting values / narrowed ranges / fixed values:
- amp_attack: a measured attack >= 50 ms is trusted (range x/4..x*4); shorter readings are
  dominated by the analyser's 23 ms RMS frames, so the range is just capped at 80 ms.
- amp_release: not used. The analyser's fall time depends on the sustain level and the clip
  length (on test renders a 1.5 s release read 0.7 s, a 0.05 s release with low sustain read
  0.3 s), so it is not a safe range.
- pitch_env_amount: when a blip is detected, its sign is known and its measured peak is a lower
  bound (the 46 ms analysis frames smear a 30-60 ms blip; on synthetic renders the reading is
  roughly a third of the true peak), so the range is |peak|..24 on that side and the start is
  3 x |peak|. Without a blip the pitch envelope is fixed at 0 and not searched.
- osc_detune_cents: when beating was confirmed, start at the estimate, range est/2..est*2.
"""
from __future__ import annotations

import re
from typing import Any

from app.synth.params import PARAMS

MAX_PARAMS = 10  # 8 was measurably worse on the e2e case (see README "Match search")
PITCH_PARAMS = ("pitch_env_amount", "pitch_env_decay")
# when a template's manifest list is longer than MAX_PARAMS, keep these first (most audible)
PRIORITY = ("filter_cutoff", "pitch_env_amount", "osc1_wave", "osc_detune_cents", "amp_attack", "amp_decay",
            "amp_sustain", "amp_release", "pitch_env_decay", "filter_drive", "filter_env_amount", "filt_decay",
            "osc2_level", "osc1_unison_detune", "filter_resonance", "fm_amount", "filter_lfo_rate",
            "filter_lfo_depth", "vibrato_rate", "vibrato_depth")
DEFAULT_WEIGHTS = {"stft": 0.5, "ltas": 3.0, "mfcc": 0.5, "env": 1.0, "attack": 2.0, "pitch": 0.0}


def _clip_range(name: str, lo: float, hi: float) -> tuple[float, float]:
    p = PARAMS[name]
    lo, hi = max(min(lo, hi), p.lo), min(max(lo, hi), p.hi)
    if hi <= lo:
        lo, hi = p.lo, p.hi
    return float(lo), float(hi)


def blip(analysis: dict[str, Any]) -> tuple[bool, float]:
    pm = analysis.get("pitch_movement") or {}
    return bool(pm.get("detected")), float(pm.get("peak_semitones") or 0.0)


def analysis_priors(analysis: dict[str, Any], params: list[str]) -> dict[str, Any]:
    """{"initial": {...}, "ranges": {...}, "fixed": {...}, "notes": [why...]} for `params`."""
    initial: dict[str, float] = {}
    ranges: dict[str, tuple[float, float]] = {}
    fixed: dict[str, float] = {}
    why: list[str] = []
    env = analysis.get("envelope") or {}

    att = env.get("attack_s")
    if "amp_attack" in params and att is not None:
        if att >= 0.05:
            initial["amp_attack"] = float(att)
            ranges["amp_attack"] = _clip_range("amp_attack", att / 4, att * 4)
            why.append(f"attack measured {att:.3f} s")
        else:
            ranges["amp_attack"] = _clip_range("amp_attack", PARAMS["amp_attack"].lo, 0.08)
            why.append("attack is fast (below the analyser's resolution)")

    found, peak = blip(analysis)
    if "pitch_env_amount" in params:
        if found and abs(peak) >= 0.5:
            sign = 1.0 if peak > 0 else -1.0
            lo, hi = abs(peak), 24.0
            ranges["pitch_env_amount"] = _clip_range("pitch_env_amount", sign * lo, sign * hi)
            initial["pitch_env_amount"] = sign * min(3 * abs(peak), 24.0)
            why.append(f"pitch blip measured {peak:+.1f} st (a lower bound)")
        else:
            fixed["pitch_env_amount"] = 0.0
            why.append("no pitch blip: pitch envelope off")
    if "pitch_env_decay" in params and found:
        settle = (analysis.get("pitch_movement") or {}).get("settle_time_s")
        ranges["pitch_env_decay"] = _clip_range("pitch_env_decay", 0.005, max(0.3, 3 * (settle or 0.1)))

    beat = analysis.get("beating") or {}
    est = beat.get("detune_cents_estimate")
    if "osc_detune_cents" in params and est and beat.get("confirmed"):
        initial["osc_detune_cents"] = float(min(est, PARAMS["osc_detune_cents"].hi))
        ranges["osc_detune_cents"] = _clip_range("osc_detune_cents", est / 2, est * 2)
        why.append(f"oscillator beating suggests ~{est:.0f} cents detune")

    return {"initial": initial, "ranges": {k: list(v) for k, v in ranges.items()}, "fixed": fixed, "notes": why}


def choose_params(candidates: list[str], analysis: dict[str, Any], limit: int = MAX_PARAMS) -> list[str]:
    """Up to `limit` of `candidates`, most audible first; pitch-envelope params only with a blip."""
    found, _ = blip(analysis)
    cands = list(dict.fromkeys(candidates))
    if found:
        cands += [p for p in PITCH_PARAMS if p not in cands]
    else:
        cands = [p for p in cands if p not in PITCH_PARAMS]
    rank = {n: i for i, n in enumerate(PRIORITY)}
    return sorted(cands, key=lambda n: rank.get(n, len(rank)))[:limit]


# ---- template choice ------------------------------------------------------------------
KEYWORDS = {
    "pitch_blip_keys": ("blip", "pitch bend", "pitch drop", "chirp", "zap"),
    "pad": ("pad", "ambient", "swell", "wash"),
    "strings": ("string", "violin", "orchestral"),
    "brass": ("brass", "horn", "trumpet"),
    "organ": ("organ", "hammond"),
    "fm_bell": ("bell", "fm", "chime", "glass", "dx7", "electric piano", "rhodes"),
    "bass": ("bass", "sub"),
    "pluck": ("pluck", "stab", "staccato"),
    "supersaw": ("supersaw", "trance", "hoover", "wide saw"),
    "square_lead": ("square", "chiptune", "hollow", "nasal"),
    "noise_sweep": ("noise", "riser", "sweep", "whoosh"),
    "saw_lead": ("lead",),
}


def choose_template(analysis: dict[str, Any], description: str, manifest: dict[str, Any]) -> tuple[str, str]:
    """(template, reason). Analysis evidence first, then description keywords, then a default."""
    notes = analysis.get("notes") or []
    env = analysis.get("envelope") or {}
    spec = analysis.get("spectral") or {}
    found, peak = blip(analysis)
    chord = len(notes) > 0 and max((sum(1 for m in notes if abs(m["onset"] - n["onset"]) < 0.04) for n in notes),
                                    default=1) >= 2
    if found:
        return "pitch_blip_keys", f"pitch blip of {peak:+.1f} st on {'chords' if chord else 'notes'}"
    if (spec.get("flatness_mean") or 0) > 0.2:
        return "noise_sweep", "noisy spectrum (high flatness)"
    desc = description.lower()
    for name, words in KEYWORDS.items():
        hit = next((w for w in words if re.search(rf"\b{re.escape(w)}", desc)), None)
        if name in manifest and hit:
            return name, f"description mentions '{hit}'"
    attack = env.get("attack_s") or 0.0
    if attack >= 0.15:
        return ("pad" if chord else "strings"), f"slow attack ({attack:.2f} s)"
    if notes and min(n["midi"] for n in notes) < 43 and not chord:
        return "bass", "low monophonic notes"
    if (spec.get("odd_even_ratio") or 0) > 3.0:
        return "square_lead", "odd harmonics dominate (square-like)"
    if chord and (analysis.get("beating") or {}).get("confirmed"):
        return "supersaw", "detuned chords"
    if chord:
        return "pluck" if (env.get("release_s") or 1.0) < 0.3 else "supersaw", "chords"
    return "saw_lead", "default for single notes"


def rule_plan(analysis: dict[str, Any], description: str, manifest: dict[str, Any],
              template: str | None = None) -> dict[str, Any]:
    """A complete plan.json from analysis alone (template may be forced)."""
    reason = "given"
    if template is None:
        template, reason = choose_template(analysis, description, manifest)
    params = choose_params(manifest[template]["search"], analysis)
    pri = analysis_priors(analysis, params)
    params = [p for p in params if p not in pri["fixed"]]
    if not blip(analysis)[0]:
        pri["fixed"]["pitch_env_amount"] = 0.0  # a template's own pitch blip would contradict the analysis
    return {"template": template, "params": params, "ranges": pri["ranges"], "initial": pri["initial"],
            "fixed": pri["fixed"], "weights": dict(DEFAULT_WEIGHTS), "source": "rules",
            "reasons": [f"template {template}: {reason}", *pri["notes"]]}

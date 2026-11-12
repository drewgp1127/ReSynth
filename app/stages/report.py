"""Stage 6: final files and the written report.

Inputs:  <job>/match.json, match_{1,2,3}.vital, match_1.wav, target.wav, plan.json, analysis.json,
         research.json (optional), the user's gear text
Outputs: patch.vital, patch_alt1.vital, patch_alt2.vital   (best match + 2 runner-ups)
         ab.wav            target, a gap, then the match (loudness-matched)
         report.json       everything below as data
         report.md, report.html
         status/report.json

The recipe is generated from the final matched parameter values (recipe.py), not from the plan.
Claude, when available, writes the "on your gear" section and effect suggestions from that
recipe; without a key those come from simple rules and the report says so.

Similarity label from the best match loss against the target (the same weighted loss the search
minimises). Thresholds were calibrated on synthetic runs (see SIMILARITY_THRESHOLDS); with a
mixed target even the true parameter values do not score "close" every time.

Run: python -m app.stages.report <job_dir> [--gear "Serum, Logic"] [--offline]
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any

from app.stages.claude import ClaudeUnavailable, ask_json, get_client
from app.stages.common import make_ab, read_audio, read_json, write_audio, write_json, write_status
from app.stages.match import template_path
from app.stages.recipe import final_values, recipe, recipe_text
from app.synth.vital import load_template

# best-match loss (default weights) -> label. Calibrated on synthetic runs (pitch_blip_keys chords
# mixed under drums + bass, separated with htdemucs_6s):
#   same template, clean target, values recovered:                 0.05 - 0.3
#   right template under the mix (even the true values score this): 0.85 - 1.3
#   wrong templates after a 60 s search (organ, pad):              1.66 - 1.73
#   fm_bell, a wrong family, still reached 1.39, so the bands overlap: the label is a rough guide.
SIMILARITY_THRESHOLDS = (("close", 1.0), ("same family", 1.4))
SIMILARITY_TEXT = {
    "close": "Close: the match should sound like the same patch; fine-tune by ear.",
    "same family": "Same family: the right kind of sound, but expect to adjust brightness, envelope or detune by ear.",
    "weak": "Weak: the search did not get near the target. Try a cleaner region or a different stem.",
}


def similarity(loss: float) -> str:
    for label, limit in SIMILARITY_THRESHOLDS:
        if loss < limit:
            return label
    return "weak"


def rule_effects(analysis: dict[str, Any], research: dict[str, Any] | None) -> list[dict[str, str]]:
    fx = []
    st = analysis.get("stereo") or {}
    if (st.get("width") or 0) > 0.25:
        fx.append({"effect": "Chorus or stereo ensemble", "settings": "slow rate, moderate depth, wide mix",
                   "why": f"the target is wide (side/mid ratio {st['width']:.2f}) and the search renders in mono-ish"})
    if (analysis.get("envelope") or {}).get("release_s", 0) and analysis["envelope"]["release_s"] > 0.8:
        fx.append({"effect": "Reverb", "settings": "medium room or plate, 1-2 s decay, 15-25% mix",
                   "why": "the tail rings longer than a dry synth release usually does"})
    for r in (research or {}).get("recreations", []) if research and research.get("status") == "done" else []:
        if r.get("settings"):
            fx.append({"effect": f"From the recreation '{r.get('title', '')}'", "settings": r["settings"],
                       "why": "a published recreation of this sound uses this (not the original gear)"})
    fx.append({"effect": "EQ and saturation to taste", "settings": "compare against the song in your DAW",
               "why": "mix effects and tape character are not matched by the search"})
    return fx


def caveats(analysis: dict[str, Any], research: dict[str, Any] | None, match: dict[str, Any]) -> list[str]:
    out = ["The preset matches the marked notes only; check it across the keyboard and at other velocities.",
           "Reverb, delay, chorus and other effects are not matched by the search; add them by ear.",
           "Stem separation can smear or leak other instruments into the target, and the search then matches the smear."]
    if (analysis.get("pitch_movement") or {}).get("detected"):
        out.append("The measured pitch-blip size is a lower bound; if the blip sounds too small, raise the pitch "
                   "envelope amount.")
    if research is None or research.get("status") != "done":
        reason = (research or {}).get("reason", "not run")
        out.append(f"Gear research was skipped ({reason}).")
    elif not research.get("model_known"):
        out.append("No source names the synth model; the gear section gives the type of sound only.")
    if match.get("window_s"):
        out.append("The search ran mostly on a short window of the region and was checked on the whole region.")
    return out


REPORT_SYSTEM = """You help a music producer recreate a synth sound on their own gear.
You get the matched patch as a recipe in generic synth terms (these numbers are final and come from
the matched preset: do not change them), the research about the original gear, and the producer's
gear list. Write:
- on_your_gear: how to set this recipe up on each piece of gear they listed (text only; name the
  specific controls where you know them, say when a control has no direct equivalent). If they
  listed nothing, explain it for a generic subtractive synth.
- effects: effects to add after the synth that the search does not cover, with settings and why.
- notes: up to three short practical tips for finishing the sound by ear.
Never present a published recreation as the original gear."""

REPORT_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["on_your_gear", "effects", "notes"],
    "properties": {
        "on_your_gear": {"type": "string"},
        "effects": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["effect", "settings", "why"],
            "properties": {"effect": {"type": "string"}, "settings": {"type": "string"}, "why": {"type": "string"}}}},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
}


def claude_prose(recipe_str: str, research: dict[str, Any] | None, gear: str, label: str, client: Any) -> dict[str, Any]:
    res = None
    if research and research.get("status") == "done":
        res = {k: research.get(k) for k in ("model_statement", "instruments", "synthesis", "recreations", "summary")}
    prompt = json.dumps({"recipe": recipe_str, "similarity": label, "research": res, "gear": gear or "(none given)"},
                        indent=1)
    return ask_json(client, REPORT_SYSTEM, prompt, REPORT_SCHEMA, max_tokens=8000)


def generic_gear_text(gear: str) -> str:
    base = ("Set up the recipe on any subtractive synth: match the oscillator shapes and detune first, then the "
            "filter cutoff and envelopes, then any pitch envelope. Use the A/B audio to compare.")
    if gear.strip():
        base = f"Your gear: {gear.strip()}. " + base + " (Gear-specific instructions need the Claude API.)"
    return base


def build_report(job: Path, gear: str = "", client: Any = None, offline: bool = False) -> dict[str, Any]:
    match = read_json(job / "match.json")
    plan = read_json(job / "plan.json")
    analysis = read_json(job / "analysis.json")
    research = read_json(job / "research.json") if (job / "research.json").exists() else None

    files = {"patch": "patch.vital"}
    shutil.copyfile(job / "match_1.vital", job / "patch.vital")
    for i in (2, 3):
        if (job / f"match_{i}.vital").exists():
            shutil.copyfile(job / f"match_{i}.vital", job / f"patch_alt{i - 1}.vital")
            files[f"alt{i - 1}"] = f"patch_alt{i - 1}.vital"
    write_audio(job / "ab.wav", make_ab(read_audio(job / "target.wav"), read_audio(job / "match_1.wav")))
    files["ab"] = "ab.wav"

    patch = load_template(template_path(plan["template"]))
    best = match["best"][0]
    values = final_values(patch, plan.get("fixed") or match.get("fixed") or {}, best["values"])
    sections = recipe(patch, values)
    loss = float(best["loss"])
    label = similarity(loss)

    prose_source, prose_reason = "rules", "offline"
    prose: dict[str, Any] = {"on_your_gear": generic_gear_text(gear), "effects": rule_effects(analysis, research),
                             "notes": []}
    if not offline:
        try:
            prose = claude_prose(recipe_text(sections), research, gear, label, client or get_client())
            prose_source, prose_reason = "claude", ""
        except ClaudeUnavailable as e:
            prose_reason = str(e)

    report = {
        "template": plan["template"],
        "similarity": {"label": label, "loss": round(loss, 4), "initial_loss": round(float(match["initial_loss"]), 4),
                       "explanation": SIMILARITY_TEXT[label]},
        "recipe": sections,
        "values": {k: round(v, 4) for k, v in best["values"].items()},
        "runner_ups": [{"file": files.get(f"alt{i}"), "loss": round(float(c["loss"]), 4)}
                       for i, c in enumerate(match["best"][1:3], 1)],
        "on_your_gear": prose["on_your_gear"],
        "effects": prose["effects"],
        "notes": prose.get("notes", []),
        "prose_source": prose_source, "prose_skipped_reason": prose_reason,
        "gear_text": gear,
        "research": research,
        "caveats": caveats(analysis, research, match),
        "plan_source": plan.get("source"), "plan_reasons": plan.get("reasons", []),
        "files": files,
    }
    return report


# ---- rendering ------------------------------------------------------------------------
CONF_LABEL = {"confirmed": "Confirmed", "community-sourced": "Community-sourced", "speculated": "Speculated"}


def _gear_lines(research: dict[str, Any] | None) -> list[str]:
    if not research or research.get("status") != "done":
        return [f"Research skipped: {(research or {}).get('reason', 'not run')}."]
    lines = [research.get("model_statement", "")]
    for i in research.get("instruments", []):
        url = i.get("source_url") or ""
        lines.append(f"[{CONF_LABEL.get(i['confidence'], i['confidence'])}] {i['name']} (model: {i['model']}) - "
                     f"{i.get('evidence', '')}" + (f" <{url}>" if url else "")
                     + ("" if i.get("url_verified", True) else " (link not verified)"))
    syn = research.get("synthesis") or {}
    if syn:
        lines.append(f"Synthesis type [{CONF_LABEL.get(syn.get('confidence'), '')}]: {syn.get('type', '')} - "
                     f"{syn.get('details', '')}")
    for q in research.get("quotes", []):
        lines.append(f"Quote: \"{q['quote']}\" - {q.get('speaker', '')}" + (f" <{q['source_url']}>" if q.get("source_url") else ""))
    for r in research.get("recreations", []):
        lines.append(f"Recreation (not the original gear): {r.get('title', '')} by {r.get('author', '')}: "
                     f"{r.get('settings', '')}" + (f" <{r['url']}>" if r.get("url") else ""))
    return [x for x in lines if x]


def to_markdown(r: dict[str, Any]) -> str:
    out = ["# ReSynth report", "",
           f"**Similarity: {r['similarity']['label']}** (loss {r['similarity']['loss']}, template start "
           f"{r['similarity']['initial_loss']}). {r['similarity']['explanation']}", "",
           f"Template: `{r['template']}`. Files: " + ", ".join(f"`{f}`" for f in r["files"].values()), "",
           "## Gear report", ""]
    out += [f"- {x}" for x in _gear_lines(r.get("research"))]
    out += ["", "## Recipe (from the matched preset)", ""]
    for s in r["recipe"]:
        out += [f"**{s['title']}**", ""] + [f"- {x}" for x in s["lines"]] + [""]
    out += ["## On your gear", "", r["on_your_gear"], ""]
    if r["prose_source"] != "claude":
        out += [f"_Written without Claude ({r['prose_skipped_reason']})._", ""]
    out += ["## Recommended effects", ""] + [f"- **{e['effect']}**: {e['settings']} ({e['why']})" for e in r["effects"]]
    if r.get("notes"):
        out += ["", "## Tips", ""] + [f"- {n}" for n in r["notes"]]
    out += ["", "## Caveats", ""] + [f"- {c}" for c in r["caveats"]]
    return "\n".join(out) + "\n"


def _linkify(text: str) -> str:
    """Escape text and turn <https://...> markers into links."""
    import re

    parts = re.split(r"<(https?://[^>\s]+)>", text)
    out = []
    for i, p in enumerate(parts):
        if i % 2:
            u = html.escape(p, quote=True)
            out.append(f'<a href="{u}" rel="noopener noreferrer" target="_blank">source</a>')
        else:
            out.append(html.escape(p))
    return "".join(out)


def to_html(r: dict[str, Any]) -> str:
    md_lines = to_markdown(r).splitlines()
    body = []
    in_list = False
    import re

    for line in md_lines:
        if line.startswith("- "):
            if not in_list:
                body.append("<ul>")
                in_list = True
            item = _linkify(line[2:])
            item = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", item)
            body.append(f"<li>{item}</li>")
            continue
        if in_list:
            body.append("</ul>")
            in_list = False
        if line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.strip():
            p = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _linkify(line))
            p = re.sub(r"`(.+?)`", r"<code>\1</code>", p)
            p = re.sub(r"^_(.+)_$", r"<em>\1</em>", p)
            body.append(f"<p>{p}</p>")
    if in_list:
        body.append("</ul>")
    style = ("body{font:16px/1.5 system-ui,sans-serif;max-width:46rem;margin:2rem auto;padding:0 1rem;"
             "color:#1d1f23;background:#fbfbfa}h1,h2{line-height:1.2}code{background:#eee;padding:0 .2em}"
             "@media (prefers-color-scheme:dark){body{color:#e6e6e3;background:#16171a}code{background:#2a2c31}"
             "a{color:#8ab4ff}}")
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
            f"content=\"width=device-width,initial-scale=1\"><title>ReSynth report</title><style>{style}</style>"
            f"</head><body>{''.join(body)}</body></html>")


def run_report(job_dir: str | Path, gear: str = "", client: Any = None, offline: bool = False) -> dict[str, Any]:
    job = Path(job_dir)
    write_status(job, "report", "running", 0.2, "writing the report")
    try:
        rep = build_report(job, gear, client, offline)
        write_json(job / "report.json", rep)
        (job / "report.md").write_text(to_markdown(rep), encoding="utf-8")
        (job / "report.html").write_text(to_html(rep), encoding="utf-8")
        msg = f"similarity: {rep['similarity']['label']}"
        if rep["prose_source"] != "claude":
            msg += f"; gear notes written without Claude ({rep['prose_skipped_reason']})"
        write_status(job, "report", "done", 1.0, msg)
        return rep
    except Exception as e:
        write_status(job, "report", "failed", 0.0, f"{type(e).__name__}: {e}")
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description="ReSynth stage 6: report")
    ap.add_argument("job_dir")
    ap.add_argument("--gear", default="")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    rep = run_report(a.job_dir, a.gear, offline=a.offline)
    print(to_markdown(rep))


if __name__ == "__main__":
    main()

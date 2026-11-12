"""Stage 1: gear research with Claude and web search.

Input:  song, artist, the user's description of the sound
Output: <job>/research.json, <job>/status/research.json

research.json:
  status            "done" | "skipped" (with "reason", e.g. no API key)
  model_known       true only when a source names the synth model used for this sound
  model_statement   e.g. "Model unknown: no source names the synth used for ..."
  instruments       [{name, model, role, confidence: confirmed|community-sourced|speculated,
                      evidence, source_url, source_title, url_verified}]
  synthesis         {type, details, confidence, basis}
  recreations       [{title, author, url, settings, url_verified, is_original_gear: false}]
  quotes            [{quote, speaker, context, source_url, url_verified}]
  summary           short paragraph
  sources           every URL the search returned {url: title}

Rules enforced after Claude answers (not only asked for in the prompt):
- a URL Claude cites that the search never returned is flagged url_verified=false, and a
  "confirmed" claim resting on such a URL is downgraded to "speculated";
- recreations are always marked is_original_gear=false;
- if no confirmed instrument names a model, model_known is false and the statement says so.

Run: python -m app.stages.research <job_dir> --song ... --artist ... --description ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.stages.claude import ClaudeUnavailable, ask_json, get_client, search_notes
from app.stages.common import write_json, write_status

CONFIDENCE = ["confirmed", "community-sourced", "speculated"]

SEARCH_SYSTEM = """You research the gear behind a sound in a recorded song for a music producer.
Search the web. Look for: interviews with the artist or producer, gear lists, studio/tour rig
photos or rundowns, forum threads, and published sound recreations or tutorials of this sound.

Your search budget is small, so plan it:
- The artist's own words about this song matter most. Spend at least two searches on them,
  with different angles: "<artist> <song> interview", "<artist> favorite tracks" or
  "<artist> picks their best songs", and track-by-track or album commentary. Band members often
  describe a sound ("weird", "ugly", "cheap") without naming the instrument; record such remarks
  verbatim. Do not finish while budget remains and no quote about this song's sound is found.
- Spend one search on gear for the record or tour, and one on recreations or tutorials.
- Never repeat a query you have already run.

Sort every claim into one of three levels:
- confirmed: the artist, producer or engineer said it, or an authoritative source (interview,
  official rig rundown) states it for this song or record;
- community-sourced: forums, fan sites, tutorials or recreation authors say it;
- speculated: your own inference from the sound or the era.

Rules:
- A published recreation (a tutorial or preset that imitates the sound) is NOT the original gear.
  Report its settings as a recreation and never present them as what the artist used.
- If no source names the synth model used for this sound, say plainly that the model is unknown.
  Do not guess a model and present it as fact.
- Record any quote from the band about the sound or the instrument verbatim, with its source.
- Give the URL for every claim. Only cite URLs you actually found in search results."""

STRUCTURE_SYSTEM = """You turn research notes about the gear behind a sound into JSON.
Use only what the notes and the source list say; do not add facts. Keep each URL exactly as it
appears in the source list. Use "unknown" for a model no source names. A recreation goes in
"recreations", never in "instruments" as if it were the original gear."""

_STR = {"type": "string"}
SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["model_known", "model_statement", "instruments", "synthesis", "recreations", "quotes", "summary"],
    "properties": {
        "model_known": {"type": "boolean"},
        "model_statement": _STR,
        "instruments": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "model", "role", "confidence", "evidence", "source_url", "source_title"],
            "properties": {"name": _STR, "model": _STR, "role": _STR,
                           "confidence": {"type": "string", "enum": CONFIDENCE},
                           "evidence": _STR, "source_url": _STR, "source_title": _STR}}},
        "synthesis": {
            "type": "object", "additionalProperties": False,
            "required": ["type", "details", "confidence", "basis"],
            "properties": {"type": _STR, "details": _STR, "confidence": {"type": "string", "enum": CONFIDENCE},
                           "basis": _STR}},
        "recreations": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["title", "author", "url", "settings"],
            "properties": {"title": _STR, "author": _STR, "url": _STR, "settings": _STR}}},
        "quotes": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["quote", "speaker", "context", "source_url"],
            "properties": {"quote": _STR, "speaker": _STR, "context": _STR, "source_url": _STR}}},
        "summary": _STR,
    },
}


def _norm(url: str) -> str:
    return url.strip().rstrip("/").replace("http://", "https://").replace("://www.", "://").split("#")[0]


def enforce_rules(data: dict[str, Any], sources: dict[str, str]) -> dict[str, Any]:
    """Apply the research rules to Claude's structured answer (see module docstring)."""
    known = {_norm(u) for u in sources}

    def verified(url: str) -> bool:
        return bool(url) and _norm(url) in known

    for inst in data.get("instruments", []):
        inst["url_verified"] = verified(inst.get("source_url", ""))
        if inst.get("confidence") == "confirmed" and not inst["url_verified"]:
            inst["confidence"] = "speculated"
            inst["evidence"] = (inst.get("evidence", "") + " [downgraded: cited URL not found in search results]").strip()
        if inst.get("confidence") not in CONFIDENCE:
            inst["confidence"] = "speculated"
        if not inst.get("model", "").strip():
            inst["model"] = "unknown"
    for r in data.get("recreations", []):
        r["url_verified"] = verified(r.get("url", ""))
        r["is_original_gear"] = False
    for q in data.get("quotes", []):
        q["url_verified"] = verified(q.get("source_url", ""))
    syn = data.get("synthesis") or {}
    if syn.get("confidence") not in CONFIDENCE:
        syn["confidence"] = "speculated"

    named = [i for i in data.get("instruments", [])
             if i["confidence"] == "confirmed" and i["model"].strip().lower() not in ("", "unknown", "n/a")]
    if not named:
        data["model_known"] = False
        stmt = data.get("model_statement", "")
        if "unknown" not in stmt.lower():
            data["model_statement"] = ("Model unknown: no source found names the synth model used for this sound. "
                                       + stmt).strip()
    return data


def research(song: str, artist: str, description: str, client: Any = None, max_searches: int = 8) -> dict[str, Any]:
    client = client or get_client()
    prompt = (f"Song: {song}\nArtist: {artist}\nThe sound: {description}\n\n"
              "Find what instrument or synth made this sound, how it was likely synthesised, any published "
              "recreation of it (with its settings), and any quotes from the artist about it.")
    notes = search_notes(client, SEARCH_SYSTEM, prompt, max_searches=max_searches)
    src = "\n".join(f"- {u} ({t})" for u, t in notes.urls.items()) or "(none)"
    data = ask_json(client, STRUCTURE_SYSTEM,
                    f"Song: {song}\nArtist: {artist}\nSound: {description}\n\nResearch notes:\n{notes.text}\n\n"
                    f"Source list (URLs returned by the search):\n{src}", SCHEMA, effort="low")
    data = enforce_rules(data, notes.urls)
    data.update(status="done", song=song, artist=artist, description=description, sources=notes.urls,
                searches=notes.searches, notes=notes.text, usage=notes.usage)
    return data


def run_research(job_dir: str | Path, song: str, artist: str, description: str, client: Any = None) -> dict[str, Any]:
    """Stage entry point. Never raises for a missing key or API failure: writes status 'skipped'."""
    job = Path(job_dir)
    write_status(job, "research", "running", 0.1, "searching the web")
    if not (song.strip() or artist.strip()):
        out = {"status": "skipped", "reason": "no song or artist given"}
        write_json(job / "research.json", out)
        write_status(job, "research", "skipped", 1.0, out["reason"])
        return out
    try:
        out = research(song, artist, description, client)
    except ClaudeUnavailable as e:
        out = {"status": "skipped", "reason": str(e), "song": song, "artist": artist}
        write_json(job / "research.json", out)
        write_status(job, "research", "skipped", 1.0, f"research skipped: {e}")
        return out
    except Exception as e:
        out = {"status": "skipped", "reason": f"research failed ({type(e).__name__})", "song": song, "artist": artist}
        write_json(job / "research.json", out)
        write_status(job, "research", "skipped", 1.0, out["reason"])
        return out
    write_json(job / "research.json", out)
    n = len(out.get("instruments", []))
    write_status(job, "research", "done", 1.0,
                 f"{n} instrument claim(s), {len(out.get('recreations', []))} recreation(s); "
                 + ("model named" if out["model_known"] else "model unknown"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="ReSynth stage 1: gear research")
    ap.add_argument("job_dir")
    ap.add_argument("--song", required=True)
    ap.add_argument("--artist", required=True)
    ap.add_argument("--description", default="")
    a = ap.parse_args()
    Path(a.job_dir).mkdir(parents=True, exist_ok=True)
    out = run_research(a.job_dir, a.song, a.artist, a.description)
    print(json.dumps({k: v for k, v in out.items() if k not in ("notes", "sources")}, indent=1))


if __name__ == "__main__":
    main()

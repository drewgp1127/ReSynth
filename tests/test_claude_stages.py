"""Research, plan and report stages with a stubbed Anthropic client (no network, no key)."""
import json

import numpy as np
import pytest

from app.stages import claude as claude_mod
from app.stages.common import write_audio, write_json
from app.stages.plan import load_manifest, make_plan, run_plan, validate, InvalidPlan
from app.stages.report import run_report, similarity
from app.stages.research import run_research
from tests.fake_claude import FailingClient, FakeClient, json_response, response, search_result_block, text_block

REVIEW = "https://reverbmachine.example/lemon-glow-synth"
INTERVIEW = "https://magazine.example/interview"

STRUCTURED = {
    "model_known": False,
    "model_statement": "No source names the keyboard model.",
    "instruments": [
        {"name": "keyboard synth", "model": "unknown", "role": "main chords", "confidence": "community-sourced",
         "evidence": "recreation author describes it", "source_url": REVIEW, "source_title": "Recreation"},
        {"name": "vintage polysynth", "model": "Made-Up 9000", "role": "main chords", "confidence": "confirmed",
         "evidence": "claimed without a source", "source_url": "https://not-in-results.example/x", "source_title": ""},
    ],
    "synthesis": {"type": "subtractive", "details": "two detuned tri-saw oscillators with a pitch blip",
                  "confidence": "community-sourced", "basis": "recreation"},
    "recreations": [{"title": "Recreation", "author": "Someone", "url": REVIEW,
                     "settings": "two detuned oscillators, fast pitch envelope"}],
    "quotes": [{"quote": "placeholder quote about the keyboard", "speaker": "band member", "context": "interview",
                "source_url": INTERVIEW}],
    "summary": "type known, model unknown",
}


def research_responder(kw):
    if "tools" in kw:  # the web-search call
        return response([
            NS_server_tool_use(),
            search_result_block([{"url": REVIEW, "title": "Recreation"}, {"url": INTERVIEW, "title": "Interview"}]),
            text_block("Notes: no model named; a recreation exists.", [{"url": REVIEW, "title": "Recreation"}]),
        ])
    assert kw["output_config"]["format"]["type"] == "json_schema"
    return json_response(STRUCTURED)


def NS_server_tool_use():
    from types import SimpleNamespace
    return SimpleNamespace(type="server_tool_use", name="web_search", input={"query": "q"})


def test_research_with_stub_enforces_rules(tmp_path):
    client = FakeClient(research_responder)
    out = run_research(tmp_path, "Lemon Glow", "Beach House", "pulsing chords with a pitch blip", client=client)
    assert out["status"] == "done"
    assert out["model_known"] is False and "unknown" in out["model_statement"].lower()
    by_name = {i["name"]: i for i in out["instruments"]}
    # a "confirmed" model resting on a URL the search never returned is downgraded
    assert by_name["vintage polysynth"]["confidence"] == "speculated"
    assert not by_name["vintage polysynth"]["url_verified"]
    assert by_name["keyboard synth"]["url_verified"]
    assert out["recreations"][0]["is_original_gear"] is False and out["recreations"][0]["url_verified"]
    assert out["quotes"][0]["url_verified"]
    # request shape: web search tool on the first call, structured output on the second, model id
    assert client.calls[0]["tools"][0]["type"] == claude_mod.WEB_SEARCH_TOOL
    assert all(c["model"] == claude_mod.MODEL for c in client.calls)
    assert json.loads((tmp_path / "research.json").read_text())["status"] == "done"


def test_research_pause_turn_is_resumed(tmp_path):
    state = {"n": 0}

    def responder(kw):
        if "tools" in kw:
            state["n"] += 1
            if state["n"] == 1:
                return response([NS_server_tool_use()], stop_reason="pause_turn")
            assert kw["messages"][-1]["role"] == "assistant"
            return response([search_result_block([{"url": REVIEW, "title": "R"}]), text_block("done")])
        return json_response(STRUCTURED)

    out = run_research(tmp_path, "Song", "Artist", "desc", client=FakeClient(responder))
    assert state["n"] == 2 and REVIEW in out["sources"]


def test_research_skipped_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv(claude_mod.KEY_ENV, raising=False)
    out = run_research(tmp_path, "Song", "Artist", "desc")
    assert out["status"] == "skipped" and claude_mod.KEY_ENV in out["reason"]
    status = json.loads((tmp_path / "status" / "research.json").read_text())
    assert status["state"] == "skipped"


def test_research_api_error_is_skipped(tmp_path):
    out = run_research(tmp_path, "Song", "Artist", "desc", client=FailingClient(RuntimeError("boom")))
    assert out["status"] == "skipped" and "boom" in out["reason"]


def test_key_only_from_resynth_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-wrong")
    monkeypatch.delenv(claude_mod.KEY_ENV, raising=False)
    with pytest.raises(claude_mod.ClaudeUnavailable):
        claude_mod.get_client()
    monkeypatch.setenv(claude_mod.KEY_ENV, "sk-test-not-real")
    assert claude_mod.get_client().api_key == "sk-test-not-real"


# ---- plan ------------------------------------------------------------------------------
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


def test_claude_plan_validated_and_merged():
    raw = {"template": "pitch_blip_keys", "params": ["filter_cutoff", "pitch_env_amount", "osc_detune_cents"],
           "ranges": [{"param": "filter_cutoff", "lo": 1000, "hi": 99999}],
           "initial": [{"param": "filter_cutoff", "value": 4000}], "reasons": ["blip"]}
    plan = make_plan(blip_analysis(), None, "desc", client=FakeClient(lambda kw: json_response(raw)))
    assert plan["source"] == "claude" and plan["template"] == "pitch_blip_keys"
    assert plan["ranges"]["filter_cutoff"] == [1000.0, 16000.0]  # clipped to the param bounds
    assert plan["ranges"]["pitch_env_amount"] == [2.0, 24.0]  # filled from the analysis
    assert plan["initial"]["filter_cutoff"] == 4000.0


@pytest.mark.parametrize("bad", [
    {"template": "no_such_template", "params": ["filter_cutoff"], "ranges": [], "initial": [], "reasons": []},
    {"template": "pad", "params": ["not_a_param"], "ranges": [], "initial": [], "reasons": []},
    {"template": "pad", "params": [], "ranges": [], "initial": [], "reasons": []},
    {"template": "pad", "params": ["filter_cutoff"], "ranges": [{"param": "filter_cutoff", "lo": 50000, "hi": 90000}],
     "initial": [], "reasons": []},
])
def test_invalid_claude_plan_falls_back(bad):
    with pytest.raises(InvalidPlan):
        validate(bad, load_manifest())
    plan = make_plan(blip_analysis(), None, "", client=FakeClient(lambda kw: json_response(bad)))
    assert plan["source"] == "rules" and plan["fallback_reason"]


def test_plan_offline_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv(claude_mod.KEY_ENV, raising=False)
    write_json(tmp_path / "analysis.json", blip_analysis())
    plan = run_plan(tmp_path, "desc")
    assert plan["source"] == "rules" and claude_mod.KEY_ENV in plan["fallback_reason"]
    assert json.loads((tmp_path / "plan.json").read_text())["template"] == "pitch_blip_keys"


# ---- report ------------------------------------------------------------------------------
def fake_match_job(tmp_path):
    from app.stages.match import TEMPLATES_DIR
    from app.synth.vital import load_template, save_vital

    patch = load_template(TEMPLATES_DIR / "pitch_blip_keys.vital")
    for i in (1, 2, 3):
        save_vital(patch, tmp_path / f"match_{i}.vital", name=f"ReSynth match {i}")
    t = np.linspace(0, 1, 44100, dtype=np.float32)
    write_audio(tmp_path / "target.wav", 0.1 * np.sin(2 * np.pi * 220 * t))
    write_audio(tmp_path / "match_1.wav", 0.3 * np.sin(2 * np.pi * 220 * t))
    write_json(tmp_path / "analysis.json", blip_analysis())
    write_json(tmp_path / "plan.json", {"template": "pitch_blip_keys", "params": ["pitch_env_amount"], "fixed": {}})
    vals = {"pitch_env_amount": 6.5, "filter_cutoff": 4800.0}
    write_json(tmp_path / "match.json", {"initial_loss": 1.9, "best": [
        {"loss": 1.1, "values": vals}, {"loss": 1.2, "values": vals}, {"loss": 1.3, "values": vals}],
        "fixed": {}, "window_s": [0.0, 1.2]})


def test_report_offline(tmp_path, monkeypatch):
    monkeypatch.delenv(claude_mod.KEY_ENV, raising=False)
    fake_match_job(tmp_path)
    write_json(tmp_path / "research.json", {"status": "skipped", "reason": "no API key"})
    rep = run_report(tmp_path, gear="Serum")
    for f in ("patch.vital", "patch_alt1.vital", "patch_alt2.vital", "ab.wav", "report.json", "report.md", "report.html"):
        assert (tmp_path / f).exists(), f
    assert rep["similarity"]["label"] == "same family"
    text = (tmp_path / "report.md").read_text()
    # recipe comes from the final values, not the plan
    assert "+6.5 semitones" in text and "4.8 kHz" in text and "tri-saw" in text
    assert "Research skipped" in text and "Serum" in text
    assert rep["prose_source"] == "rules"
    json.loads((tmp_path / "patch.vital").read_text())


def test_report_with_stubbed_claude(tmp_path):
    fake_match_job(tmp_path)
    prose = {"on_your_gear": "In Serum: set OSC A to a tri-saw ...", "effects": [
        {"effect": "Chorus", "settings": "slow", "why": "width"}], "notes": ["trust your ears"]}
    client = FakeClient(lambda kw: json_response(prose))
    rep = run_report(tmp_path, gear="Serum", client=client)
    assert rep["prose_source"] == "claude" and rep["on_your_gear"].startswith("In Serum")
    sent = json.loads(client.calls[0]["messages"][0]["content"])
    assert "+6.5 semitones" in sent["recipe"]
    assert "<script" not in (tmp_path / "report.html").read_text()


def test_similarity_labels():
    assert similarity(0.5) == "close" and similarity(1.2) == "same family" and similarity(1.6) == "weak"

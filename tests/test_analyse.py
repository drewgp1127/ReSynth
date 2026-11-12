import json

import numpy as np
import pytest

from app.stages.analyse import analyse_audio, run_analyse
from app.stages.common import write_audio
from app.stages.match import TEMPLATES_DIR
from app.synth.params import PARAMS
from app.synth.vital import Note, load_template, render_patch


def render(name: str, notes: list[Note], dur: float, values: dict[str, float] | None = None) -> np.ndarray:
    raw: dict[str, float] = {}
    for k, v in (values or {}).items():
        raw.update(PARAMS[k].to_raw(v))
    return render_patch(load_template(TEMPLATES_DIR / f"{name}.vital"), notes, dur, raw)


def analyse(tmp_path, audio: np.ndarray, method: str = "auto") -> dict:
    write_audio(tmp_path / "target.wav", audio)
    return analyse_audio(audio, tmp_path / "target.wav", method)


CHORDS = [Note(57, 0.0, 0.5), Note(61, 0.0, 0.5), Note(64, 0.0, 0.5),
          Note(55, 0.7, 0.5), Note(59, 0.7, 0.5), Note(62, 0.7, 0.5), Note(67, 1.4, 0.5)]


def test_notes_from_chords(tmp_path):
    r = analyse(tmp_path, render("saw_lead", CHORDS, 2.4))
    assert r["note_method"] == "basic-pitch"
    found = sorted((n["midi"], round(n["onset"], 1)) for n in r["notes"])
    assert found == sorted((n.midi, round(n.onset, 1)) for n in CHORDS), found
    for n in r["notes"]:
        truth = min(CHORDS, key=lambda c: abs(c.onset - n["onset"]))
        assert abs(n["onset"] - truth.onset) < 0.04


def test_pyin_fallback_monophonic(tmp_path):
    notes = [Note(60, 0.0, 0.4), Note(64, 0.5, 0.4), Note(67, 1.0, 0.4)]
    r = analyse(tmp_path, render("saw_lead", notes, 1.8), method="pyin")
    assert r["note_method"] == "pyin"
    assert [n["midi"] for n in r["notes"]] == [60, 64, 67]


def test_pitch_blip_detected_on_chords(tmp_path):
    r = analyse(tmp_path, render("pitch_blip_keys", CHORDS, 2.4))
    pm = r["pitch_movement"]
    assert pm["detected"] and pm["blip_count"] == pm["group_count"] == 3
    assert 2.0 < pm["peak_semitones"] < 7.0
    assert pm["settle_time_s"] < 0.2


def test_no_blip_without_pitch_envelope(tmp_path):
    r = analyse(tmp_path, render("saw_lead", CHORDS, 2.4))
    assert not r["pitch_movement"]["detected"] and r["pitch_movement"]["blip_count"] == 0


def test_negative_blip_detected(tmp_path):
    r = analyse(tmp_path, render("saw_lead", CHORDS, 2.4, {"pitch_env_amount": -4.0, "pitch_env_decay": 0.06}))
    assert r["pitch_movement"]["detected"] and r["pitch_movement"]["peak_semitones"] < -1.5


def test_vibrato_rate(tmp_path):
    r = analyse(tmp_path, render("saw_lead", [Note(60, 0.0, 2.0)], 2.6, {"vibrato_rate": 6.0, "vibrato_depth": 30.0}))
    assert r["vibrato"]["rate_hz"] == pytest.approx(6.0, abs=0.3)
    plain = analyse(tmp_path, render("saw_lead", [Note(60, 0.0, 2.0)], 2.6))
    assert plain["vibrato"]["rate_hz"] is None


def test_tremolo_and_filter_lfo_rates(tmp_path):
    trem = analyse(tmp_path, render("saw_lead", [Note(57, 0.0, 2.0)], 2.6, {"tremolo_rate": 4.0, "tremolo_depth": 0.5}))
    assert trem["tremolo"]["rate_hz"] == pytest.approx(4.0, abs=0.3)
    lfo = analyse(tmp_path, render("saw_lead", [Note(57, 0.0, 2.0)], 2.6,
                                   {"filter_lfo_rate": 3.0, "filter_lfo_depth": 24.0, "filter_cutoff": 800.0}))
    assert lfo["filter_lfo"]["rate_hz"] == pytest.approx(3.0, abs=0.3)


def test_beating_gives_detune(tmp_path):
    notes = [Note(48, 0.0, 2.0), Note(55, 0.0, 2.0), Note(64, 0.0, 2.0)]
    r = analyse(tmp_path, render("pitch_blip_keys", notes, 2.6, {"osc_detune_cents": 12.0}))
    assert r["beating"]["detune_cents_estimate"] == pytest.approx(12.0, abs=3.0)


def test_attack_release_and_timbre(tmp_path):
    note = [Note(57, 0.0, 1.0)]
    slow = analyse(tmp_path, render("saw_lead", note, 2.2, {"amp_attack": 0.3, "amp_release": 0.6}))
    fast = analyse(tmp_path, render("saw_lead", note, 2.2, {"amp_attack": 0.002, "amp_release": 0.1}))
    assert slow["envelope"]["attack_s"] > 0.1 > fast["envelope"]["attack_s"]
    assert slow["envelope"]["release_s"] > 2 * fast["envelope"]["release_s"]
    square = analyse(tmp_path, render("saw_lead", note, 1.4, {"osc1_wave": 170 / 255, "filter_cutoff": 16000.0}))
    saw = analyse(tmp_path, render("saw_lead", note, 1.4, {"filter_cutoff": 16000.0}))
    assert square["spectral"]["odd_even_ratio"] > 10 * saw["spectral"]["odd_even_ratio"]
    noise = analyse(tmp_path, render("noise_sweep", note, 1.4))
    assert noise["spectral"]["flatness_mean"] > 10 * saw["spectral"]["flatness_mean"]


def test_stereo_width(tmp_path):
    wide = analyse(tmp_path, render("supersaw", [Note(57, 0.0, 1.0)], 1.4))
    mono = analyse(tmp_path, render("saw_lead", [Note(57, 0.0, 1.0)], 1.4))
    assert wide["stereo"]["width"] > 0.2 and mono["stereo"]["width"] < 0.01


def test_stage_writes_json_and_status(tmp_path):
    write_audio(tmp_path / "target.wav", render("pluck", CHORDS, 2.4))
    run_analyse(tmp_path)
    data = json.loads((tmp_path / "analysis.json").read_text())
    assert data["notes"] and data["duration"] == pytest.approx(2.4, abs=0.01)
    status = json.loads((tmp_path / "status" / "analyse.json").read_text())
    assert status["state"] == "done"

import json

import numpy as np

from app.synth.params import SearchSpace
from app.synth.vital import Note, VitalRenderer, load_template, render_patch, save_vital

DUR = 1.4


def hf_db(a: np.ndarray, sr: int = 44100) -> float:
    """Share of energy above 2 kHz, in dB."""
    m = np.abs(np.fft.rfft(a.mean(0))) ** 2
    f = np.fft.rfftfreq(a.shape[1], 1 / sr)
    return float(10 * np.log10(m[f > 2000].sum() / m.sum()))


def test_render_chords_non_silent_and_stereo(patch, chords):
    a = render_patch(patch, chords, DUR)
    assert a.shape == (2, round(DUR * 44100))
    assert np.abs(a).max() > 0.05
    # second chord starts at 0.6 s: energy right after it
    assert np.abs(a[:, int(0.62 * 44100):int(0.7 * 44100)]).max() > 0.05


def test_controls_path_matches_saved_preset(patch, chords, tmp_path):
    sp = SearchSpace.build(["filter_cutoff", "filter_env_amount", "amp_attack", "osc_detune_cents", "pitch_env_amount"])
    vals = {"filter_cutoff": 700.0, "filter_env_amount": 24.0, "amp_attack": 0.02, "osc_detune_cents": 12.0,
            "pitch_env_amount": 5.0}
    r = VitalRenderer(patch)
    r.set_params(sp.raw(sp.values_from_unit([0.1] * 5)))  # leave other values set first
    r.set_params(sp.raw(vals))
    live = r.render_notes(chords, DUR)
    path = save_vital(patch, tmp_path / "x.vital", sp.raw(vals), name="x")
    json.loads(path.read_text())
    reloaded = render_patch(load_template(path), chords, DUR)
    assert np.abs(live - reloaded).max() < 1e-4


def test_parameters_and_modulation_change_sound(patch):
    note = [Note(57, 0.0, 0.5)]
    sp = SearchSpace.build(["filter_cutoff", "filter_env_amount", "filt_decay"])
    closed = render_patch(patch, note, 0.8, sp.raw({"filter_cutoff": 300.0, "filter_env_amount": 0.0, "filt_decay": 0.2}))
    env = render_patch(patch, note, 0.8, sp.raw({"filter_cutoff": 300.0, "filter_env_amount": 48.0, "filt_decay": 0.2}))
    early = slice(0, int(0.05 * 44100))
    assert hf_db(env[:, early]) > hf_db(closed[:, early]) + 6

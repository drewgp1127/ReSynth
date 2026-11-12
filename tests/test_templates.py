import json
from pathlib import Path

import numpy as np
import pytest

from app.synth.builder import STANDARD_MODULATIONS
from app.synth.loss import LossFn
from app.synth.params import PARAMS
from app.synth.vital import Note, load_template, modulation_slots, render_patch

TEMPLATES = Path(__file__).resolve().parent.parent / "app" / "templates"
NAMES = sorted(p.stem for p in TEMPLATES.glob("*.vital"))
NOTES = [Note(48, 0.0, 0.8), Note(55, 0.0, 0.8), Note(60, 0.0, 0.8), Note(64, 0.0, 0.8), Note(67, 1.0, 0.4)]
DUR = 1.8


@pytest.fixture(scope="module")
def renders() -> dict[str, np.ndarray]:
    return {n: render_patch(load_template(TEMPLATES / f"{n}.vital"), NOTES, DUR) for n in NAMES}


def test_twelve_templates_with_manifest():
    manifest = json.loads((TEMPLATES / "manifest.json").read_text())
    assert len(NAMES) == 12 and set(manifest) == set(NAMES)
    for name, entry in manifest.items():
        assert entry["description"]
        assert all(p in PARAMS for p in entry["search"]), name


@pytest.mark.parametrize("name", NAMES)
def test_template_structure(name):
    patch = load_template(TEMPLATES / f"{name}.vital")
    slots = modulation_slots(patch)
    for src, dst, _ in STANDARD_MODULATIONS:
        assert f"{src}->{dst}" in slots
    st = patch["settings"]
    assert all(st[f"osc_{i}_random_phase"] == 0.0 for i in (1, 2, 3))
    for fx in ("chorus_on", "delay_on", "reverb_on", "distortion_on", "phaser_on", "flanger_on"):
        assert st[fx] == 0.0, fx  # global effects stay off (not summable per note)


@pytest.mark.parametrize("name", NAMES)
def test_template_renders_non_silent_and_deterministic(name, renders):
    a = renders[name]
    rms = float(np.sqrt(np.mean(a ** 2)))
    assert rms > 0.01, rms
    assert np.isfinite(a).all()
    again = render_patch(load_template(TEMPLATES / f"{name}.vital"), NOTES, DUR)
    assert np.abs(again - a).max() < 1e-4


def test_templates_distinct(renders):
    worst = (np.inf, "", "")
    for a in NAMES:
        loss = LossFn(renders[a], onsets=[0.0, 1.0])
        for b in NAMES:
            if a != b:
                d = loss(renders[b])[0]
                worst = min(worst, (d, a, b))
    assert worst[0] > 0.5, worst

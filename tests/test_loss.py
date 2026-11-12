import numpy as np

from app.synth.loss import LossFn
from app.synth.params import SearchSpace
from app.synth.vital import VitalRenderer

DUR = 1.4


def test_loss_properties(patch, chords):
    sp = SearchSpace.build(["filter_cutoff"])
    r = VitalRenderer(patch)
    r.set_params(sp.raw({"filter_cutoff": 900.0}))
    target = r.render_notes(chords, DUR)
    loss = LossFn(target)
    assert loss(target)[0] < 1e-6
    assert loss(target * 0.2)[0] < 1e-4  # loudness-normalised
    assert loss(np.zeros_like(target))[0] > 10
    losses = []
    for hz in (900.0, 1300.0, 2500.0, 6000.0):
        r.set_params(sp.raw({"filter_cutoff": hz}))
        losses.append(loss(r.render_notes(chords, DUR))[0])
    assert losses == sorted(losses), losses


def test_pitch_term(patch, chords):
    sp = SearchSpace.build(["filter_cutoff"])
    r = VitalRenderer(patch)
    target = r.render_notes(chords[:1], 0.8)
    from app.synth.loss import LossWeights

    loss = LossFn(target, weights=LossWeights(pitch=1.0))
    _, parts = loss(r.render_notes([chords[0].__class__(59, 0.0, 0.4)], 0.8))
    assert parts["pitch"] > 0.1
    assert loss(target)[1]["pitch"] < 1e-6

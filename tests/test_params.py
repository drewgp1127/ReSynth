import math

import pytest

from app.synth.params import PARAMS, SearchSpace


@pytest.mark.parametrize("name", sorted(PARAMS))
def test_unit_and_raw_round_trip(name):
    p = PARAMS[name]
    for u in (0.0, 0.3, 0.77, 1.0):
        v = p.from_unit(u)
        assert p.lo <= v <= p.hi
        if not p.integer:
            assert math.isclose(p.to_unit(v), u, abs_tol=1e-9)
        raw = p.to_raw(v)
        assert math.isclose(p.from_raw(raw[p.targets[0].control]), v, rel_tol=1e-6, abs_tol=1e-9)


def test_known_scalings():
    assert PARAMS["filter_cutoff"].to_raw(440.0) == {"filter_1_cutoff": 69.0}
    assert math.isclose(PARAMS["amp_attack"].to_raw(0.0081)["env_1_attack"], 0.3)
    assert PARAMS["osc_detune_cents"].to_raw(20.0) == {"osc_1_tune": -0.1, "osc_2_tune": 0.1}
    assert math.isclose(PARAMS["pitch_env_amount"].to_raw(12.0)["env_3->voice_transpose"], 0.125)


def test_search_space_fixed_and_ranges():
    sp = SearchSpace.build(["filter_cutoff", "amp_attack"], ranges={"filter_cutoff": (200, 2000)},
                           fixed={"filter_resonance": 0.3})
    assert sp.params[0].lo == 200
    raw = sp.raw(sp.values_from_unit([0.5, 0.5]))
    assert raw["filter_1_resonance"] == 0.3
    assert set(raw) == {"filter_1_cutoff", "env_1_attack", "filter_1_resonance"}

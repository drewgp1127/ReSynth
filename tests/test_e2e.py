from pathlib import Path

import pytest

from tests.e2e_pipeline import run_e2e


@pytest.mark.slow
def test_end_to_end_pitch_blip(tmp_path):
    pytest.importorskip("audio_separator")
    s = run_e2e(Path(tmp_path), budget_s=120)
    print(s)
    assert s["analysis"]["pitch_blip_detected"]
    assert s["match"]["loss_vs_target_final"] < s["match"]["loss_vs_target_initial"]
    assert s["match"]["loss_vs_clean_synth_match"] < s["match"]["loss_vs_clean_synth_template_default"]

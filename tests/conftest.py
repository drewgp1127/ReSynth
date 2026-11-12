from __future__ import annotations

import warnings

import pytest

from app.synth.builder import base_patch
from app.synth.vital import Note

warnings.filterwarnings("ignore", message="Could not import matplotlib")


def two_osc_patch() -> dict:
    """Two saw oscillators, 3-voice unison on osc 1 (test-only patch, built from init)."""
    p = base_patch("test two osc")
    st = p["settings"]
    st["osc_1_unison_voices"] = 3.0
    st["osc_2_on"] = 1.0
    st["osc_2_level"] = 0.6
    return p


@pytest.fixture
def patch() -> dict:
    return two_osc_patch()


CHORDS = [Note(57, 0.0, 0.4), Note(61, 0.0, 0.4), Note(64, 0.0, 0.4),
          Note(55, 0.6, 0.4), Note(59, 0.6, 0.4), Note(62, 0.6, 0.4)]


@pytest.fixture
def chords() -> list[Note]:
    return list(CHORDS)

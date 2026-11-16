"""Sanity checks for ion / charge bookkeeping (stub)."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 1: implement after ions_atp dynamics exist")
def test_sodium_potassium_pump_stoichiometry():
    """Pump should move 3 Na out and 2 K in per ATP (discrete bookkeeping)."""
    assert False


@pytest.mark.skip(reason="Phase 1: implement after point model exists")
def test_resting_state_finite():
    """Unstimulated trajectory should remain finite over a short horizon."""
    assert False

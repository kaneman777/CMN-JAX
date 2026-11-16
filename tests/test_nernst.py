"""Nernst potentials and outward-positive K/Na current signs."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jnpnp.models.hh_point import default_hh_params, ionic_currents
from jnpnp.models.ions_atp import E_K, E_Na


def test_ek_equal_concentrations_is_zero():
    ek = float(E_K(140.0, 140.0))
    assert abs(ek) < 1e-6


def test_ek_physiological_is_negative():
    ek = float(E_K(4.0, 140.0))
    assert ek < 0.0


def test_ena_physiological_is_positive():
    ena = float(E_Na(10.0, 145.0))
    assert ena > 0.0


def test_ik_outward_when_v_above_ek():
    params = default_hh_params()
    ek = E_K(4.0, 140.0)
    ena = E_Na(10.0, 145.0)
    # n≈1 so gK*(V-EK) dominates sign
    m, h, n = 0.0, 1.0, 1.0
    v_hi = ek + 40.0
    v_lo = ek - 40.0
    _, ik_hi, _ = ionic_currents(v_hi, m, h, n, ena, ek, params)
    _, ik_lo, _ = ionic_currents(v_lo, m, h, n, ena, ek, params)
    assert float(ik_hi) > 0.0
    assert float(ik_lo) < 0.0


def test_ina_inward_when_v_below_ena():
    """Outward-positive: V < E_Na ⇒ I_Na < 0 (Na influx)."""
    params = default_hh_params()
    ek = E_K(4.0, 140.0)
    ena = E_Na(10.0, 145.0)
    m, h, n = 1.0, 1.0, 0.0
    v_lo = ena - 40.0
    v_hi = ena + 40.0
    ina_lo, _, _ = ionic_currents(v_lo, m, h, n, ena, ek, params)
    ina_hi, _, _ = ionic_currents(v_hi, m, h, n, ena, ek, params)
    assert float(ina_lo) < 0.0
    assert float(ina_hi) > 0.0

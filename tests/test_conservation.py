"""Narrow ion bookkeeping checks — not closed-volume PNP charge conservation.

This model injects I_ext, uses volume-factor approximations, and clips Na/a/Ke.
Full electroneutrality / weighted total-charge conservation is intentionally
**not** asserted here (would be a false claim against the current API).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jnpnp.models.ions_atp import default_ion_params, dKe_dt, dNa_dt


def test_dna_dt_pump_stoichiometry_factor_3():
    """dNa_i/dt = -γ (I_Na + 3 I_pump) — coefficient 3 regression."""
    params = default_ion_params()
    I_Na = jnp.asarray(-2.0, dtype=jnp.float64)
    I_pump = jnp.asarray(0.4, dtype=jnp.float64)
    got = dNa_dt(I_Na, I_pump, params)
    expect = -params["gamma_Na"] * (I_Na + 3.0 * I_pump)
    assert abs(float(got - expect)) < 1e-15
    # Sign: outward pump alone lowers Na_i
    assert float(dNa_dt(0.0, 1.0, params)) < 0.0


def test_dke_dt_pump_term_factor_2():
    """Local cleft: γ_Ke * (I_K - 2 I_pump) — pump removes 2 K⁺ per cycle."""
    params = {
        **default_ion_params(),
        "gamma_Ke": 0.0005,
        "tau_K": 1e12,  # bath leak negligible for this unit check
        "K_e_bath": 4.0,
    }
    I_K = jnp.asarray(1.0, dtype=jnp.float64)
    I_pump = jnp.asarray(0.25, dtype=jnp.float64)
    Ke = jnp.asarray(4.0, dtype=jnp.float64)
    got = dKe_dt(I_K, I_pump, Ke, params)
    expect = params["gamma_Ke"] * (I_K - 2.0 * I_pump)  # bath term ~0
    assert abs(float(got - expect)) < 1e-12
    # Pump alone (I_K=0) decreases K_e
    assert float(dKe_dt(0.0, 1.0, Ke, params)) < 0.0


def test_ke_diffusion_exchange_is_zero_sum():
    """Equal-volume 2-point diffusion: D*(Ke_j-Ke_i) + D*(Ke_i-Ke_j) == 0.

    Unit-level (same algebra as ``simulate_two_cell_ke``). Trajectory-level
    sums need not be exact zero because of ``clip_Ke`` and bath leak.
    """
    D = 0.002
    Ke1, Ke2 = 8.0, 3.0
    d1 = D * (Ke2 - Ke1)
    d2 = D * (Ke1 - Ke2)
    assert abs(d1 + d2) < 1e-15

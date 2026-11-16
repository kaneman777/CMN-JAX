"""Concentration / ATP positivity under driven short sims (with known clips)."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jnpnp.models.ions_atp import clip_Ke, clip_Na, clip_a, da_dt, default_ion_params
from jnpnp.models.point_sim import DT, I_ext_step, merge_params, simulate_driven, simulate_driven_ke


def test_driven_trajectory_respects_clip_floors():
    t_end = 120.0
    t = jnp.arange(0.0, t_end + DT / 2, DT)
    I = I_ext_step(t, 10.0, 100.0, 15.0)
    traj = simulate_driven(I, params=merge_params(), dt=DT)
    na = np.asarray(traj["Na_i"])
    a = np.asarray(traj["a"])
    assert np.all(np.isfinite(na))
    assert np.all(np.isfinite(a))
    assert float(na.min()) >= 5.0 - 1e-9
    assert float(a.min()) >= 1e-3 - 1e-12


def test_driven_ke_trajectory_respects_ke_floor():
    t_end = 100.0
    t = jnp.arange(0.0, t_end + DT / 2, DT)
    I = I_ext_step(t, 10.0, 80.0, 12.0)
    params = merge_params(gamma_Ke=0.0005, tau_K=1000.0, K_e_bath=4.0)
    traj = simulate_driven_ke(I, ion_on=True, params=params, dt=DT)
    ke = np.asarray(traj["K_e"])
    assert np.all(np.isfinite(ke))
    assert float(ke.min()) >= 2.0 - 1e-9


def test_da_dt_production_positive_below_amax():
    params = default_ion_params()
    a = jnp.asarray(0.4, dtype=jnp.float64)
    # zero pump → pure production
    rhs = float(da_dt(a, 0.0, params, a_max=1.0))
    assert rhs > 0.0
    # a above a_max → production term negative (restore toward ceiling)
    rhs_hi = float(da_dt(jnp.asarray(1.2), 0.0, params, a_max=1.0))
    assert rhs_hi < 0.0


def test_clip_helpers_match_documented_bounds():
    assert float(clip_Na(-10.0)) == 5.0
    assert float(clip_a(-1.0, 1.0)) == 1e-3
    assert float(clip_Ke(0.5)) == 2.0

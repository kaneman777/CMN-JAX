"""Short-horizon jax.grad smoke through simulate_driven."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jnpnp.models.ions_atp import da_dt, default_ion_params
from jnpnp.models.point_sim import DT, I_ext_step, merge_params, soft_spike, simulate_driven


def _short_grid(t_end: float = 40.0, dt: float = DT) -> jnp.ndarray:
    n = int(round(t_end / dt))
    return jnp.arange(n + 1, dtype=jnp.float64) * dt


def test_grad_stim_gain_finite_nonzero():
    t = _short_grid(40.0)
    base_I = I_ext_step(t, 5.0, 35.0, 8.0)
    params = merge_params()

    def loss(stim_gain: jnp.ndarray) -> jnp.ndarray:
        traj = simulate_driven(stim_gain * base_I, params=params, dt=DT)
        return jnp.mean(soft_spike(traj["V"]))

    g = jax.grad(loss)(jnp.asarray(1.0, dtype=jnp.float64))
    g_f = float(g)
    assert jnp.isfinite(g)
    assert abs(g_f) > 1e-12


def test_grad_da_dt_wrt_a_max_finite_nonzero():
    """RHS-level: production term depends on a_max (full sim float()-casts a0)."""
    params = default_ion_params()
    a = jnp.asarray(0.5, dtype=jnp.float64)
    I_pump = jnp.asarray(0.1, dtype=jnp.float64)

    def loss(a_max: jnp.ndarray) -> jnp.ndarray:
        return da_dt(a, I_pump, params, a_max=a_max) ** 2

    g = jax.grad(loss)(jnp.asarray(1.0, dtype=jnp.float64))
    assert jnp.isfinite(g)
    assert abs(float(g)) > 1e-12

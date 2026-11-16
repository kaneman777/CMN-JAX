"""Local metabolic 3-factor learning helpers (Phase 4-Learn).

Eligibility × ATP gate (× optional global M≡1). Not a claim of biological
credit-assignment; toy local update for comparison with Adam/BPTT.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import lax


def integrate_eligibility(
    I_chan: Any,
    s: Any,
    *,
    dt: float,
    tau_e: float,
    I_ref: float = 15.0,
) -> Any:
    """Forward Euler eligibility: ė = -e/τ_e + (I_chan/I_ref) * s(t).

    ``I_ref`` keeps e O(1) for typical pulse amplitudes. Returns e(t) on the
    same grid as ``I_chan`` / ``s``.
    """
    I_chan = jnp.asarray(I_chan, dtype=jnp.float64) / jnp.maximum(I_ref, 1e-6)
    s = jnp.asarray(s, dtype=jnp.float64)
    decay = jnp.exp(-dt / tau_e)

    def step(e, inputs):
        I_t, s_t = inputs
        e_new = decay * e + I_t * s_t * dt
        return e_new, e_new

    _e_final, e_traj = lax.scan(step, jnp.asarray(0.0, dtype=jnp.float64), (I_chan, s))
    return e_traj


def atp_gate(a: Any, *, a_crit: float, beta: float) -> Any:
    """g(a) = tanh((a - a_crit) / β). High a → potentiate; low a → prune."""
    return jnp.tanh((a - a_crit) / beta)


def local_delta_w(
    I_sig: Any,
    I_noise: Any,
    s: Any,
    a: Any,
    *,
    dt: float,
    tau_e: float,
    a_crit: float,
    beta: float,
    eta: float,
    M: float = 1.0,
    I_ref: float = 15.0,
) -> Any:
    """Δw_i = η * M * mean_t(e_i(t) * g(a(t))) for i∈{sig, noise}."""
    e1 = integrate_eligibility(I_sig, s, dt=dt, tau_e=tau_e, I_ref=I_ref)
    e2 = integrate_eligibility(I_noise, s, dt=dt, tau_e=tau_e, I_ref=I_ref)
    g = atp_gate(a, a_crit=a_crit, beta=beta)
    dw1 = eta * M * jnp.mean(e1 * g)
    dw2 = eta * M * jnp.mean(e2 * g)
    return jnp.stack([dw1, dw2])


def task_assist_delta(
    mean_S: Any,
    mean_N: Any,
    *,
    s_hi: float,
    s_lo: float,
    eta_assist: float,
) -> Any:
    """Minimal task push to avoid silence collapse (optional; not pure local).

    Raises w1 if mean_S too low; lowers w2 if mean_N too high.
    """
    dw1 = eta_assist * jnp.maximum(s_hi * 0.5 - mean_S, 0.0)
    dw2 = -eta_assist * jnp.maximum(mean_N - s_lo * 2.0, 0.0)
    return jnp.stack([dw1, dw2])


def firing_rate_homeo_delta(
    mean_S: Any,
    *,
    s_target: float,
    eta_h: float,
) -> Any:
    """Interval-S firing-rate homeostasis on w1 only (Phase 4-Learn A1).

    err_S = mean_S - s_target;  Δw1 = -η_h * err_S;  Δw2 = 0.
    This is Local+homeo (not pure local). Does not claim cortical homeostasis.
    """
    err_S = mean_S - s_target
    dw1 = -eta_h * err_S
    dw2 = jnp.asarray(0.0, dtype=jnp.float64)
    return jnp.stack([dw1, dw2])

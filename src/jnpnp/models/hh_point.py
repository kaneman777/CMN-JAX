"""Single-compartment Hodgkin–Huxley membrane + gating (Phase 1).

See docs/model.md for the formal equations (E1–E2, E4 gating part).
Currents are outward-positive; time in ms, V in mV.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def _safe_div(numer: Any, denom: Any, eps: float = 1e-6) -> Any:
    """Avoid 0/0 in HH alpha forms when (V + offset) ~ 0."""
    return numer / jnp.where(jnp.abs(denom) < eps, eps, denom)


def alpha_beta(V: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Classical HH rate functions (mV, ms^{-1})."""
    # m
    am = _safe_div(0.1 * (V + 40.0), 1.0 - jnp.exp(-(V + 40.0) / 10.0))
    bm = 4.0 * jnp.exp(-(V + 65.0) / 18.0)
    # h
    ah = 0.07 * jnp.exp(-(V + 65.0) / 20.0)
    bh = 1.0 / (1.0 + jnp.exp(-(V + 35.0) / 10.0))
    # n
    an = _safe_div(0.01 * (V + 55.0), 1.0 - jnp.exp(-(V + 55.0) / 10.0))
    bn = 0.125 * jnp.exp(-(V + 65.0) / 80.0)
    return am, bm, ah, bh, an, bn


def steady_gates(V: Any) -> tuple[Any, Any, Any]:
    """Inflection / steady-state gates at voltage V."""
    am, bm, ah, bh, an, bn = alpha_beta(V)
    m_inf = am / (am + bm)
    h_inf = ah / (ah + bh)
    n_inf = an / (an + bn)
    return m_inf, h_inf, n_inf


def ionic_currents(
    V: Any,
    m: Any,
    h: Any,
    n: Any,
    E_Na: Any,
    E_K: Any,
    params: dict[str, Any],
) -> tuple[Any, Any, Any]:
    """I_Na, I_K, I_L (outward positive)."""
    gNa = params["gNa_bar"]
    gK = params["gK_bar"]
    gL = params["gL_bar"]
    E_L = params["E_L"]
    I_Na = gNa * (m**3) * h * (V - E_Na)
    I_K = gK * (n**4) * (V - E_K)
    I_L = gL * (V - E_L)
    return I_Na, I_K, I_L


def step_gates_exp_euler(V: Any, m: Any, h: Any, n: Any, dt: Any) -> tuple[Any, Any, Any]:
    """Exponential Euler update for m, h, n."""
    am, bm, ah, bh, an, bn = alpha_beta(V)
    m_inf = am / (am + bm)
    h_inf = ah / (ah + bh)
    n_inf = an / (an + bn)
    tau_m = 1.0 / (am + bm)
    tau_h = 1.0 / (ah + bh)
    tau_n = 1.0 / (an + bn)
    m_new = m_inf + (m - m_inf) * jnp.exp(-dt / tau_m)
    h_new = h_inf + (h - h_inf) * jnp.exp(-dt / tau_h)
    n_new = n_inf + (n - n_inf) * jnp.exp(-dt / tau_n)
    return m_new, h_new, n_new


def dV_dt(
    V: Any,
    I_Na: Any,
    I_K: Any,
    I_L: Any,
    I_pump: Any,
    I_ext: Any,
    params: dict[str, Any],
    I_KATP: Any = 0.0,
) -> Any:
    """Membrane voltage RHS (E4): Cm dV/dt = -sum I + I_ext.

    ``I_KATP`` is the Phase-4 simplified ATP-sensitive K current (default 0).
    """
    Cm = params["Cm"]
    return (-I_Na - I_K - I_L - I_pump - I_KATP + I_ext) / Cm


def default_hh_params() -> dict[str, float]:
    """Default HH membrane parameters (docs/model.md §4)."""
    return {
        "gNa_bar": 120.0,
        "gK_bar": 36.0,
        "gL_bar": 0.3,
        "E_L": -54.4,
        "Cm": 1.0,
    }

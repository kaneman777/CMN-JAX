"""Intracellular Na, Nernst E_Na, Na/K-ATPase, and ATP (Phase 1).

See docs/model.md for formal equations (E1, E3, E5, E6).
K_i and extracellular concentrations are fixed buffers in Phase 1.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


RT_OVER_F = 26.7  # mV


def E_Na(Na_i: Any, Na_o: Any = 145.0) -> Any:
    """Nernst sodium potential [mV] (E1)."""
    return RT_OVER_F * jnp.log(Na_o / Na_i)


def E_K(K_o: Any = 4.0, K_i: Any = 140.0) -> Any:
    """Nernst potassium potential [mV] (E1); constant when K fixed."""
    return RT_OVER_F * jnp.log(K_o / K_i)


def pump_current(a: Any, Na_i: Any, K_o: Any, params: dict[str, Any]) -> Any:
    """Na/K-ATPase current (E3), outward positive.

    I_pump = I_pump_bar * a * 1/(1+(K_Na/Na_i)^3) * 1/(1+(K_K/K_o)^2)
    """
    I_bar = params["I_pump_bar"]
    K_Na = params["K_Na"]
    K_K = params["K_K"]
    sat_Na = 1.0 / (1.0 + (K_Na / Na_i) ** 3)
    sat_K = 1.0 / (1.0 + (K_K / K_o) ** 2)
    return I_bar * a * sat_Na * sat_K


def dNa_dt(I_Na: Any, I_pump: Any, params: dict[str, Any]) -> Any:
    """Intracellular Na RHS (E5): dNa_i/dt = -gamma_Na * (I_Na + 3*I_pump)."""
    return -params["gamma_Na"] * (I_Na + 3.0 * I_pump)


def da_dt(
    a: Any, I_pump: Any, params: dict[str, Any], a_max: Any | None = None
) -> Any:
    """ATP RHS (E6): da/dt = eta_prod*(a_max - a) - eta_cost*max(I_pump, 0).

    Optional ``a_max`` overrides ``params["a_max"]`` (per-cell stress in two_cell).
    """
    if a_max is None:
        a_max = params.get("a_max", 1.0)
    return params["eta_prod"] * (a_max - a) - params["eta_cost"] * jnp.maximum(I_pump, 0.0)


def clip_Na(Na_i: Any) -> Any:
    return jnp.clip(Na_i, 5.0, 80.0)


def clip_a(a: Any, a_max: Any = 1.0) -> Any:
    """Clip ATP; ``a_max`` enables Phase-3c supply stress (< 1)."""
    return jnp.clip(a, 1e-3, a_max)


def sigma_KATP(a: Any, params: dict[str, Any]) -> Any:
    """ATP-dependent open fraction for simplified K_ATP (Phase 4-Bio).

    Hill form (docs/model.md): σ(a) = 1 / (1 + (a / a_half)^n).
    Low a → σ → 1 (channel opens); high a → σ → 0.
    """
    a_half = params.get("a_KATP_half", 0.35)
    n = params.get("n_KATP", 2.0)
    a_safe = jnp.maximum(a, 1e-6)
    return 1.0 / (1.0 + (a_safe / a_half) ** n)


def I_KATP_current(V: Any, a: Any, E_K: Any, params: dict[str, Any]) -> Any:
    """Simplified K_ATP current (outward positive).

    I_KATP = g_KATP_bar * σ(a) * (V - E_K).
    Default g_KATP_bar=0 keeps Phases 1–3 unchanged.
    """
    g_bar = params.get("g_KATP_bar", 0.0)
    return g_bar * sigma_KATP(a, params) * (V - E_K)


def clip_Ke(K_e: Any) -> Any:
    """Local extracellular K [mM] bounds (Phase 3b)."""
    return jnp.clip(K_e, 2.0, 20.0)


def dKe_dt(I_K: Any, I_pump: Any, K_e: Any, params: dict[str, Any]) -> Any:
    """Narrow extracellular K pool (Phase 3b / RQ2).

    d(K_e)/dt = γ_Ke * (I_K - 2*I_pump) - (K_e - K_e_bath) / τ_K

    Outward I_K raises K_e; pump removes 2 K⁺ from the cleft per cycle.
    """
    bath = params.get("K_e_bath", params.get("K_o", 4.0))
    gamma = params.get("gamma_Ke", 0.0)
    tau = params.get("tau_K", 500.0)
    # ion-off / clamp: tau_K <= 0 or gamma_Ke == 0 and explicit clamp handled by caller
    tau_safe = jnp.maximum(tau, 1e-6)
    return gamma * (I_K - 2.0 * I_pump) - (K_e - bath) / tau_safe


def default_ion_params() -> dict[str, float]:
    """Default ion / pump / ATP parameters (docs/model.md).

    Tunable scales (gamma_Na, I_pump_bar, eta_*) are set for the Phase 1 smoke
    qualitative target; override in the driver script as needed.
    """
    return {
        "Na_o": 145.0,
        "K_o": 4.0,
        "K_i": 140.0,
        "K_Na": 15.0,
        "K_K": 3.5,
        "I_pump_bar": 0.6,
        "gamma_Na": 0.0004,
        "eta_prod": 0.00025,
        "eta_cost": 0.05,
        "a_max": 1.0,
        # Phase 3b local K_e (disabled by default: gamma_Ke=0 → bath clamp via caller)
        "K_e_bath": 4.0,
        "gamma_Ke": 0.0,
        "tau_K": 500.0,
        # Phase 4-Bio simplified K_ATP (off by default)
        "g_KATP_bar": 0.0,
        "a_KATP_half": 0.35,
        "n_KATP": 2.0,
    }

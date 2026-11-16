"""Differentiable fixed-step point-neuron simulator (Phase 1 physics, Phase 2a grads).

Shared core for forward smoke and ``scripts/01_grad_smoke.py``. Time in ms, V in mV.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import lax

from jnpnp.models.hh_point import (
    default_hh_params,
    dV_dt,
    ionic_currents,
    steady_gates,
    step_gates_exp_euler,
)
from jnpnp.models.ions_atp import (
    E_K,
    E_Na,
    I_KATP_current,
    clip_Ke,
    clip_Na,
    clip_a,
    da_dt,
    default_ion_params,
    dKe_dt,
    dNa_dt,
    pump_current,
)

# Phase-1 tuned defaults (docs/model.md / STATUS.md)
DT = 0.025
T_END = 450.0
T_ON = 20.0
T_OFF = 400.0
I0 = 10.0
GAMMA_NA = 0.0004
I_PUMP_BAR = 0.6
ETA_PROD = 0.00025
ETA_COST = 0.05
NA_I0 = 10.0
V0 = -65.0


def merge_params(**overrides: float) -> dict[str, float]:
    p = {**default_hh_params(), **default_ion_params()}
    p["gamma_Na"] = GAMMA_NA
    p["I_pump_bar"] = I_PUMP_BAR
    p["eta_prod"] = ETA_PROD
    p["eta_cost"] = ETA_COST
    p.update(overrides)
    return p


def I_ext_step(t: Any, ton: Any, toff: Any, i_amp: Any) -> Any:
    return jnp.where((t >= ton) & (t < toff), i_amp, 0.0)


def soft_spike(V: Any, *, V_th: float = 0.0, k: float = 0.5) -> Any:
    """Smooth firing proxy s = sigmoid(k * (V - V_th))."""
    return jax_sigmoid(k * (V - V_th))


def jax_sigmoid(x: Any) -> Any:
    return 1.0 / (1.0 + jnp.exp(-x))


def atp_gNa_scale(a: Any, params: dict[str, Any]) -> Any:
    """Optional phenomenological ATP→Na excitability (RQ3 preview).

    If ``atp_gNa_floor`` is unset or ≤0, returns 1 (biophysical-only path).
    Otherwise ``I_Na`` is scaled by ``floor + (1-floor)*clip(a,0,1)``.
    Default floor=0 keeps Phase 1–3b numerics unchanged.
    """
    floor = float(params.get("atp_gNa_floor", 0.0) or 0.0)
    floor_j = jnp.asarray(floor, dtype=jnp.float64)
    a_clip = jnp.clip(a, 0.0, 1.0)
    gated = floor_j + (1.0 - floor_j) * a_clip
    return jnp.where(floor_j > 0.0, gated, jnp.asarray(1.0, dtype=jnp.float64))


def _initial_state(params: dict[str, float]) -> tuple[Any, Any]:
    """Return (y0, ek) with ATP initialized at ``a_max`` (Phase 3c stress)."""
    ek = E_K(params["K_o"], params["K_i"])
    m0, h0, n0 = steady_gates(jnp.asarray(V0, dtype=jnp.float64))
    a0 = float(params.get("a_max", 1.0))
    y0 = jnp.array([V0, m0, h0, n0, NA_I0, a0], dtype=jnp.float64)
    return y0, ek


def simulate_driven(
    I_ext_t: Any,
    *,
    fix_a: bool = False,
    dt: float = DT,
    t_end: float | None = None,
    params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Integrate with a prescribed external current time series.

    Parameters
    ----------
    I_ext_t :
        Array of shape ``(n_steps + 1,)`` [µA/cm²], sampled on the same grid as
        ``t = arange(n_steps+1) * dt``. Must be differentiable w.r.t. upstream
        parameters (e.g. channel weights).
    """
    if params is None:
        params = merge_params()
    I_ext_t = jnp.asarray(I_ext_t, dtype=jnp.float64)
    n_steps = int(I_ext_t.shape[0]) - 1
    if t_end is None:
        t_end = n_steps * dt
    times = jnp.arange(n_steps + 1, dtype=jnp.float64) * dt
    y0, ek = _initial_state(params)

    dt_j = jnp.asarray(dt, dtype=jnp.float64)
    ek_j = jnp.asarray(ek, dtype=jnp.float64)
    fix_a_j = jnp.asarray(fix_a)
    a_max = jnp.asarray(params.get("a_max", 1.0), dtype=jnp.float64)
    na_o = params["Na_o"]
    k_o = params["K_o"]

    def step(y, inputs):
        t, Ie = inputs
        V, m, h, n, Na_i, a = y
        a_eff = jnp.where(fix_a_j, a_max, a)
        ena = E_Na(Na_i, na_o)
        I_Na, I_K, I_L = ionic_currents(V, m, h, n, ena, ek_j, params)
        I_Na = I_Na * atp_gNa_scale(a_eff, params)
        I_p = pump_current(a_eff, Na_i, k_o, params)
        I_ka = I_KATP_current(V, a_eff, ek_j, params)

        V_new = V + dt_j * dV_dt(V, I_Na, I_K, I_L, I_p, Ie, params, I_KATP=I_ka)
        m_new, h_new, n_new = step_gates_exp_euler(V, m, h, n, dt_j)
        Na_new = clip_Na(Na_i + dt_j * dNa_dt(I_Na, I_p, params))
        a_new = jnp.where(
            fix_a_j,
            a_max,
            clip_a(a + dt_j * da_dt(a, I_p, params), a_max),
        )
        y_new = jnp.stack([V_new, m_new, h_new, n_new, Na_new, a_new])
        out = jnp.stack([V, Na_i, a_eff, I_p, Ie, ena, I_ka])
        return y_new, out

    scan_in = (times[:-1], I_ext_t[:-1])
    y_final, traj = lax.scan(step, y0, scan_in)
    V_f, m_f, h_f, n_f, Na_f, a_f = y_final
    a_eff_f = jnp.where(fix_a_j, a_max, a_f)
    ena_f = E_Na(Na_f, na_o)
    I_p_f = pump_current(a_eff_f, Na_f, k_o, params)
    I_ka_f = I_KATP_current(V_f, a_eff_f, ek_j, params)
    last = jnp.stack([V_f, Na_f, a_eff_f, I_p_f, I_ext_t[-1], ena_f, I_ka_f])[None, :]
    traj = jnp.concatenate([traj, last], axis=0)

    return {
        "t": times,
        "V": traj[:, 0],
        "Na_i": traj[:, 1],
        "a": traj[:, 2],
        "I_pump": traj[:, 3],
        "I_ext": traj[:, 4],
        "E_Na": traj[:, 5],
        "I_KATP": traj[:, 6],
        "params": params,
        "E_K": ek_j,
    }


def simulate_trajectory(
    stim_gain: Any = 1.0,
    *,
    fix_a: bool = False,
    dt: float = DT,
    t_end: float = T_END,
    ton: float = T_ON,
    toff: float = T_OFF,
    i0: float = I0,
    params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Integrate y=(V,m,h,n,Na_i,a); returns JAX arrays.

    Effective stimulus amplitude is ``stim_gain * i0``.
    """
    n_steps = int(round(t_end / dt))
    times = jnp.arange(n_steps + 1, dtype=jnp.float64) * dt
    i_amp = stim_gain * i0
    I_ext_t = I_ext_step(times, ton, toff, i_amp)
    return simulate_driven(I_ext_t, fix_a=fix_a, dt=dt, t_end=t_end, params=params)


def dummy_soft_spike_loss(
    stim_gain: Any,
    *,
    target: float = 0.05,
    V_th: float = 0.0,
    k: float = 0.5,
    **sim_kwargs: Any,
) -> Any:
    """L = (mean_t sigmoid(k*(V-V_th)) - target)^2."""
    traj = simulate_trajectory(stim_gain, **sim_kwargs)
    mean_s = jnp.mean(soft_spike(traj["V"], V_th=V_th, k=k))
    return (mean_s - target) ** 2


def simulate_driven_ke(
    I_ext_t: Any,
    *,
    fix_a: bool = False,
    ion_on: bool = True,
    dt: float = DT,
    t_end: float | None = None,
    params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Like ``simulate_driven`` but with local extracellular K_e (Phase 3b).

    State y = (V, m, h, n, Na_i, a, K_e). E_K = 26.7 ln(K_e / K_i).
    Pump K-saturation uses K_e. If ``ion_on`` is False, K_e is clamped to bath.
    """
    if params is None:
        params = merge_params()
    params = dict(params)
    if ion_on and params.get("gamma_Ke", 0.0) == 0.0:
        params["gamma_Ke"] = 0.002
    if "tau_K" not in params or params["tau_K"] <= 0:
        params["tau_K"] = 500.0

    I_ext_t = jnp.asarray(I_ext_t, dtype=jnp.float64)
    n_steps = int(I_ext_t.shape[0]) - 1
    if t_end is None:
        t_end = n_steps * dt
    times = jnp.arange(n_steps + 1, dtype=jnp.float64) * dt

    bath = float(params.get("K_e_bath", params.get("K_o", 4.0)))
    a_max_f = float(params.get("a_max", 1.0))
    m0, h0, n0 = steady_gates(jnp.asarray(V0, dtype=jnp.float64))
    y0 = jnp.array([V0, m0, h0, n0, NA_I0, a_max_f, bath], dtype=jnp.float64)

    dt_j = jnp.asarray(dt, dtype=jnp.float64)
    fix_a_j = jnp.asarray(fix_a)
    ion_on_j = jnp.asarray(ion_on)
    a_max = jnp.asarray(a_max_f, dtype=jnp.float64)
    na_o = params["Na_o"]
    k_i = params["K_i"]
    bath_j = jnp.asarray(bath, dtype=jnp.float64)

    def step(y, inputs):
        _t, Ie = inputs
        V, m, h, n, Na_i, a, K_e = y
        a_eff = jnp.where(fix_a_j, a_max, a)
        K_eff = jnp.where(ion_on_j, K_e, bath_j)
        ena = E_Na(Na_i, na_o)
        ek = E_K(K_eff, k_i)
        I_Na, I_K, I_L = ionic_currents(V, m, h, n, ena, ek, params)
        I_Na = I_Na * atp_gNa_scale(a_eff, params)
        I_p = pump_current(a_eff, Na_i, K_eff, params)
        I_ka = I_KATP_current(V, a_eff, ek, params)

        V_new = V + dt_j * dV_dt(V, I_Na, I_K, I_L, I_p, Ie, params, I_KATP=I_ka)
        m_new, h_new, n_new = step_gates_exp_euler(V, m, h, n, dt_j)
        Na_new = clip_Na(Na_i + dt_j * dNa_dt(I_Na, I_p, params))
        a_new = jnp.where(
            fix_a_j,
            a_max,
            clip_a(a + dt_j * da_dt(a, I_p, params), a_max),
        )
        Ke_new = jnp.where(
            ion_on_j,
            clip_Ke(K_e + dt_j * dKe_dt(I_K + I_ka, I_p, K_e, params)),
            bath_j,
        )
        y_new = jnp.stack([V_new, m_new, h_new, n_new, Na_new, a_new, Ke_new])
        out = jnp.stack([V, Na_i, a_eff, I_p, Ie, ena, K_eff, ek, I_ka])
        return y_new, out

    scan_in = (times[:-1], I_ext_t[:-1])
    y_final, traj = lax.scan(step, y0, scan_in)
    V_f, m_f, h_f, n_f, Na_f, a_f, Ke_f = y_final
    a_eff_f = jnp.where(fix_a_j, a_max, a_f)
    K_eff_f = jnp.where(ion_on_j, Ke_f, bath_j)
    ena_f = E_Na(Na_f, na_o)
    ek_f = E_K(K_eff_f, k_i)
    I_p_f = pump_current(a_eff_f, Na_f, K_eff_f, params)
    I_ka_f = I_KATP_current(V_f, a_eff_f, ek_f, params)
    last = jnp.stack(
        [V_f, Na_f, a_eff_f, I_p_f, I_ext_t[-1], ena_f, K_eff_f, ek_f, I_ka_f]
    )[None, :]
    traj = jnp.concatenate([traj, last], axis=0)

    return {
        "t": times,
        "V": traj[:, 0],
        "Na_i": traj[:, 1],
        "a": traj[:, 2],
        "I_pump": traj[:, 3],
        "I_ext": traj[:, 4],
        "E_Na": traj[:, 5],
        "K_e": traj[:, 6],
        "E_K": traj[:, 7],
        "I_KATP": traj[:, 8],
        "params": params,
    }

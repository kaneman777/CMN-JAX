"""Minimal 2-cell point network (Phase B1 synapse / Phase B2 K_e diffusion).

B1: independent (V, m, h, n, Na_i, a); coupling synaptic only.
B2: per-cell local K_e plus optional 2-point diffusion
    d(K_e,i)/dt += D * (K_e,j - K_e,i)  (not a full 1D PDE).

Default brakes (K_ATP, ATP→gNa) stay off via ``merge_params()``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import lax

from jnpnp.models.hh_point import (
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
    dKe_dt,
    dNa_dt,
    pump_current,
)
from jnpnp.models.point_sim import (
    DT,
    NA_I0,
    V0,
    atp_gNa_scale,
    merge_params,
)


def _cell_y0(*, V0_cell: float, a0_cell: float) -> Any:
    m0, h0, n0 = steady_gates(jnp.asarray(V0_cell, dtype=jnp.float64))
    return jnp.array([V0_cell, m0, h0, n0, NA_I0, a0_cell], dtype=jnp.float64)


def _step_cell(
    y: Any,
    Ie_total: Any,
    *,
    dt_j: Any,
    ek_j: Any,
    fix_a_j: Any,
    a_max: Any,
    na_o: float,
    k_o: float,
    params: dict[str, Any],
) -> tuple[Any, Any]:
    V, m, h, n, Na_i, a = y
    a_eff = jnp.where(fix_a_j, a_max, a)
    ena = E_Na(Na_i, na_o)
    I_Na, I_K, I_L = ionic_currents(V, m, h, n, ena, ek_j, params)
    I_Na = I_Na * atp_gNa_scale(a_eff, params)
    I_p = pump_current(a_eff, Na_i, k_o, params)
    I_ka = I_KATP_current(V, a_eff, ek_j, params)

    V_new = V + dt_j * dV_dt(V, I_Na, I_K, I_L, I_p, Ie_total, params, I_KATP=I_ka)
    m_new, h_new, n_new = step_gates_exp_euler(V, m, h, n, dt_j)
    Na_new = clip_Na(Na_i + dt_j * dNa_dt(I_Na, I_p, params))
    a_new = jnp.where(
        fix_a_j,
        a_max,
        clip_a(a + dt_j * da_dt(a, I_p, params, a_max=a_max), a_max),
    )
    y_new = jnp.stack([V_new, m_new, h_new, n_new, Na_new, a_new])
    return y_new, jnp.stack([V, a_eff, I_p, Ie_total])


def simulate_two_cell(
    I_drive_t: Any,
    *,
    w12: float,
    w21: float,
    dt: float = DT,
    t_end: float | None = None,
    params: dict[str, float] | None = None,
    fix_a: bool = False,
    V0_1: float = V0,
    V0_2: float = V0,
    a0_1: float | None = None,
    a0_2: float | None = None,
    a_max_1: float | None = None,
    a_max_2: float | None = None,
    V_th: float = 0.0,
    k_soft: float = 0.5,
    tau_syn: float = 5.0,
    plastic_B: bool = False,
    plastic_alpha: float = 0.0,
    plastic_a_ref: float = 1.0,
) -> dict[str, Any]:
    """Integrate two weakly coupled point cells.

    Parameters
    ----------
    I_drive_t :
        Shared external drive ``(n_steps+1,)`` [µA/cm²].
    w12, w21 :
        Synaptic weights into cell1 from cell2 / into cell2 from cell1.
        Positive = excitatory; negative = inhibitory.
    tau_syn :
        Synaptic filter time constant [ms]. ``ṡ̄ = -s̄/τ + soft_spike(V)``.
    V0_*, a0_* :
        Optional initial asymmetry (document in STATUS if used).
    a_max_1, a_max_2 :
        Per-cell ATP ceiling (Phase B1-② supply stress). Defaults to
        ``params["a_max"]``. Initial ``a0_*`` defaults to the cell's a_max.
    plastic_B :
        Plan-B minimal plasticity (not plan A): weaken |w21| when cell1 is
        ATP-poor. ``w21_eff = w21 * (1 - α u1)`` with
        ``u1 = clip(1 - a1 / plastic_a_ref, 0, 1)``. ``w12`` stays fixed.
    plastic_alpha :
        Strength α ∈ [0, 1]. Ignored unless ``plastic_B``.
    plastic_a_ref :
        Healthy ATP reference for u1 (default 1). Using ``a_max1`` as the
        denominator keeps u1≈0 when a tracks a lowered ceiling.
    """
    if params is None:
        params = merge_params()
    I_drive_t = jnp.asarray(I_drive_t, dtype=jnp.float64)
    n_steps = int(I_drive_t.shape[0]) - 1
    if t_end is None:
        t_end = n_steps * dt
    times = jnp.arange(n_steps + 1, dtype=jnp.float64) * dt

    a_max_default = float(params.get("a_max", 1.0))
    a_max1_f = float(a_max_1) if a_max_1 is not None else a_max_default
    a_max2_f = float(a_max_2) if a_max_2 is not None else a_max_default
    if a0_1 is None:
        a0_1 = a_max1_f
    if a0_2 is None:
        a0_2 = a_max2_f

    y1_0 = _cell_y0(V0_cell=float(V0_1), a0_cell=float(a0_1))
    y2_0 = _cell_y0(V0_cell=float(V0_2), a0_cell=float(a0_2))
    # state: y1(6), y2(6), sbar1, sbar2
    y0 = jnp.concatenate(
        [y1_0, y2_0, jnp.array([0.0, 0.0], dtype=jnp.float64)]
    )

    dt_j = jnp.asarray(dt, dtype=jnp.float64)
    ek_j = jnp.asarray(E_K(params["K_o"], params["K_i"]), dtype=jnp.float64)
    fix_a_j = jnp.asarray(fix_a)
    a_max1 = jnp.asarray(a_max1_f, dtype=jnp.float64)
    a_max2 = jnp.asarray(a_max2_f, dtype=jnp.float64)
    na_o = params["Na_o"]
    k_o = params["K_o"]
    w12_j = jnp.asarray(w12, dtype=jnp.float64)
    w21_j = jnp.asarray(w21, dtype=jnp.float64)
    V_th_j = jnp.asarray(V_th, dtype=jnp.float64)
    k_soft_j = jnp.asarray(k_soft, dtype=jnp.float64)
    tau_j = jnp.asarray(max(tau_syn, 1e-6), dtype=jnp.float64)
    plastic_B_j = jnp.asarray(plastic_B)
    alpha_j = jnp.asarray(float(plastic_alpha), dtype=jnp.float64)
    a_ref_j = jnp.asarray(max(float(plastic_a_ref), 1e-9), dtype=jnp.float64)

    def w21_effective(a1: Any) -> Any:
        a_eff = jnp.where(fix_a_j, a_max1, a1)
        u1 = jnp.clip(1.0 - a_eff / a_ref_j, 0.0, 1.0)
        scale = jnp.where(plastic_B_j, 1.0 - alpha_j * u1, 1.0)
        return w21_j * scale

    def step(y, inputs):
        _t, I_drive = inputs
        y1 = y[:6]
        y2 = y[6:12]
        sbar1, sbar2 = y[12], y[13]

        s1 = 1.0 / (1.0 + jnp.exp(-k_soft_j * (y1[0] - V_th_j)))
        s2 = 1.0 / (1.0 + jnp.exp(-k_soft_j * (y2[0] - V_th_j)))
        w21_eff = w21_effective(y1[5])
        I1 = I_drive + w12_j * sbar2
        I2 = I_drive + w21_eff * sbar1

        y1_n, o1 = _step_cell(
            y1,
            I1,
            dt_j=dt_j,
            ek_j=ek_j,
            fix_a_j=fix_a_j,
            a_max=a_max1,
            na_o=na_o,
            k_o=k_o,
            params=params,
        )
        y2_n, o2 = _step_cell(
            y2,
            I2,
            dt_j=dt_j,
            ek_j=ek_j,
            fix_a_j=fix_a_j,
            a_max=a_max2,
            na_o=na_o,
            k_o=k_o,
            params=params,
        )
        sbar1_n = sbar1 + dt_j * (-sbar1 / tau_j + s1)
        sbar2_n = sbar2 + dt_j * (-sbar2 / tau_j + s2)
        y_new = jnp.concatenate([y1_n, y2_n, jnp.stack([sbar1_n, sbar2_n])])
        # out: V,a,Ip,Itot,s,sbar ×2 + w21_eff
        out = jnp.concatenate(
            [
                jnp.concatenate([o1, jnp.stack([s1, sbar1])]),
                jnp.concatenate([o2, jnp.stack([s2, sbar2])]),
                jnp.stack([w21_eff]),
            ]
        )
        return y_new, out

    scan_in = (times[:-1], I_drive_t[:-1])
    y_final, traj = lax.scan(step, y0, scan_in)

    V1_f, a1_f = y_final[0], y_final[5]
    V2_f, a2_f = y_final[6], y_final[11]
    sbar1_f, sbar2_f = y_final[12], y_final[13]
    s1_f = 1.0 / (1.0 + jnp.exp(-k_soft_j * (V1_f - V_th_j)))
    s2_f = 1.0 / (1.0 + jnp.exp(-k_soft_j * (V2_f - V_th_j)))
    w21_eff_f = w21_effective(a1_f)
    I1_f = I_drive_t[-1] + w12_j * sbar2_f
    I2_f = I_drive_t[-1] + w21_eff_f * sbar1_f
    I_p1_f = pump_current(a1_f, y_final[4], k_o, params)
    I_p2_f = pump_current(a2_f, y_final[10], k_o, params)
    last = jnp.concatenate(
        [
            jnp.stack([V1_f, a1_f, I_p1_f, I1_f, s1_f, sbar1_f]),
            jnp.stack([V2_f, a2_f, I_p2_f, I2_f, s2_f, sbar2_f]),
            jnp.stack([w21_eff_f]),
        ]
    )[None, :]
    traj = jnp.concatenate([traj, last], axis=0)

    return {
        "t": times,
        "V1": traj[:, 0],
        "a1": traj[:, 1],
        "I_pump1": traj[:, 2],
        "I_tot1": traj[:, 3],
        "s1": traj[:, 4],
        "sbar1": traj[:, 5],
        "V2": traj[:, 6],
        "a2": traj[:, 7],
        "I_pump2": traj[:, 8],
        "I_tot2": traj[:, 9],
        "s2": traj[:, 10],
        "sbar2": traj[:, 11],
        "w21_eff": traj[:, 12],
        "w12": w12,
        "w21": w21,
        "a_max_1": a_max1_f,
        "a_max_2": a_max2_f,
        "tau_syn": tau_syn,
        "plastic_B": plastic_B,
        "plastic_alpha": float(plastic_alpha),
        "plastic_a_ref": float(plastic_a_ref),
        "params": params,
        "I_drive": I_drive_t,
    }


def simulate_two_cell_ke(
    I1_t: Any,
    I2_t: Any | None = None,
    *,
    w12: float = 0.0,
    w21: float = 0.0,
    D_ke: float = 0.0,
    ion_on: bool = True,
    dt: float = DT,
    t_end: float | None = None,
    params: dict[str, float] | None = None,
    fix_a: bool = False,
    V0_1: float = V0,
    V0_2: float = V0,
    a0_1: float | None = None,
    a0_2: float | None = None,
    a_max_1: float | None = None,
    a_max_2: float | None = None,
    V_th: float = 0.0,
    k_soft: float = 0.5,
    tau_syn: float = 5.0,
) -> dict[str, Any]:
    """Two cells with local K_e and optional 2-point diffusion (Phase B2).

    Parameters
    ----------
    I1_t, I2_t :
        Per-cell external drives ``(n_steps+1,)``. If ``I2_t`` is None, uses ``I1_t``.
    D_ke :
        Diffusion rate [1/ms]: ``dK_e,i/dt += D_ke * (K_e,j - K_e,i)``.
        ``D_ke=0`` is the diffusion-off control.
    ion_on :
        If False, each K_e is clamped to bath (no local accumulation / diffusion).
    w12, w21 :
        Synaptic weights (B2 default 0 = diffusion-only coupling).
    """
    if params is None:
        params = merge_params()
    params = dict(params)
    if ion_on and float(params.get("gamma_Ke", 0.0)) == 0.0:
        params["gamma_Ke"] = 0.0005
    if float(params.get("tau_K", 0.0)) <= 0.0:
        params["tau_K"] = 1000.0

    I1_t = jnp.asarray(I1_t, dtype=jnp.float64)
    if I2_t is None:
        I2_t = I1_t
    else:
        I2_t = jnp.asarray(I2_t, dtype=jnp.float64)
    n_steps = int(I1_t.shape[0]) - 1
    if t_end is None:
        t_end = n_steps * dt
    times = jnp.arange(n_steps + 1, dtype=jnp.float64) * dt

    bath = float(params.get("K_e_bath", params.get("K_o", 4.0)))
    a_max_default = float(params.get("a_max", 1.0))
    a_max1_f = float(a_max_1) if a_max_1 is not None else a_max_default
    a_max2_f = float(a_max_2) if a_max_2 is not None else a_max_default
    if a0_1 is None:
        a0_1 = a_max1_f
    if a0_2 is None:
        a0_2 = a_max2_f

    m1, h1, n1 = steady_gates(jnp.asarray(V0_1, dtype=jnp.float64))
    m2, h2, n2 = steady_gates(jnp.asarray(V0_2, dtype=jnp.float64))
    y1_0 = jnp.array(
        [V0_1, m1, h1, n1, NA_I0, float(a0_1), bath], dtype=jnp.float64
    )
    y2_0 = jnp.array(
        [V0_2, m2, h2, n2, NA_I0, float(a0_2), bath], dtype=jnp.float64
    )
    y0 = jnp.concatenate(
        [y1_0, y2_0, jnp.array([0.0, 0.0], dtype=jnp.float64)]
    )

    dt_j = jnp.asarray(dt, dtype=jnp.float64)
    fix_a_j = jnp.asarray(fix_a)
    ion_on_j = jnp.asarray(ion_on)
    a_max1 = jnp.asarray(a_max1_f, dtype=jnp.float64)
    a_max2 = jnp.asarray(a_max2_f, dtype=jnp.float64)
    bath_j = jnp.asarray(bath, dtype=jnp.float64)
    D_j = jnp.asarray(D_ke, dtype=jnp.float64)
    na_o = params["Na_o"]
    k_i = params["K_i"]
    w12_j = jnp.asarray(w12, dtype=jnp.float64)
    w21_j = jnp.asarray(w21, dtype=jnp.float64)
    V_th_j = jnp.asarray(V_th, dtype=jnp.float64)
    k_soft_j = jnp.asarray(k_soft, dtype=jnp.float64)
    tau_j = jnp.asarray(max(tau_syn, 1e-6), dtype=jnp.float64)

    def step_one(
        y: Any,
        Ie: Any,
        Ke_other: Any,
        *,
        a_max: Any,
    ) -> tuple[Any, Any]:
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
            clip_a(a + dt_j * da_dt(a, I_p, params, a_max=a_max), a_max),
        )
        dke_local = dKe_dt(I_K + I_ka, I_p, K_e, params)
        dke_diff = D_j * (Ke_other - K_e)
        Ke_new = jnp.where(
            ion_on_j,
            clip_Ke(K_e + dt_j * (dke_local + dke_diff)),
            bath_j,
        )
        y_new = jnp.stack([V_new, m_new, h_new, n_new, Na_new, a_new, Ke_new])
        out = jnp.stack([V, a_eff, I_p, Ie, K_eff, ek])
        return y_new, out

    def step(y, inputs):
        _t, I1, I2 = inputs
        y1 = y[:7]
        y2 = y[7:14]
        sbar1, sbar2 = y[14], y[15]
        Ke1, Ke2 = y1[6], y2[6]

        s1 = 1.0 / (1.0 + jnp.exp(-k_soft_j * (y1[0] - V_th_j)))
        s2 = 1.0 / (1.0 + jnp.exp(-k_soft_j * (y2[0] - V_th_j)))
        Ie1 = I1 + w12_j * sbar2
        Ie2 = I2 + w21_j * sbar1

        y1_n, o1 = step_one(y1, Ie1, Ke2, a_max=a_max1)
        y2_n, o2 = step_one(y2, Ie2, Ke1, a_max=a_max2)
        sbar1_n = sbar1 + dt_j * (-sbar1 / tau_j + s1)
        sbar2_n = sbar2 + dt_j * (-sbar2 / tau_j + s2)
        y_new = jnp.concatenate([y1_n, y2_n, jnp.stack([sbar1_n, sbar2_n])])
        out = jnp.concatenate(
            [
                jnp.concatenate([o1, jnp.stack([s1, sbar1])]),
                jnp.concatenate([o2, jnp.stack([s2, sbar2])]),
            ]
        )
        return y_new, out

    scan_in = (times[:-1], I1_t[:-1], I2_t[:-1])
    y_final, traj = lax.scan(step, y0, scan_in)

    def finalize_cell(yf: Any, Ie_f: Any, a_max: Any) -> Any:
        V_f, _m, _h, _n, Na_f, a_f, Ke_f = yf
        a_eff = jnp.where(fix_a_j, a_max, a_f)
        K_eff = jnp.where(ion_on_j, Ke_f, bath_j)
        ek = E_K(K_eff, k_i)
        I_p = pump_current(a_eff, Na_f, K_eff, params)
        s = 1.0 / (1.0 + jnp.exp(-k_soft_j * (V_f - V_th_j)))
        return jnp.stack([V_f, a_eff, I_p, Ie_f, K_eff, ek, s])

    y1_f = y_final[:7]
    y2_f = y_final[7:14]
    sbar1_f, sbar2_f = y_final[14], y_final[15]
    Ie1_f = I1_t[-1] + w12_j * sbar2_f
    Ie2_f = I2_t[-1] + w21_j * sbar1_f
    fin1 = finalize_cell(y1_f, Ie1_f, a_max1)
    fin2 = finalize_cell(y2_f, Ie2_f, a_max2)
    # traj out per step: V,a,Ip,Ie,Ke,ek,s,sbar × 2  (8+8)
    # step out was V,a,Ip,Ie,Ke,ek + s,sbar = 8 per cell
    last = jnp.concatenate(
        [
            jnp.concatenate([fin1[:6], jnp.stack([fin1[6], sbar1_f])]),
            jnp.concatenate([fin2[:6], jnp.stack([fin2[6], sbar2_f])]),
        ]
    )[None, :]
    traj = jnp.concatenate([traj, last], axis=0)

    return {
        "t": times,
        "V1": traj[:, 0],
        "a1": traj[:, 1],
        "I_pump1": traj[:, 2],
        "I_tot1": traj[:, 3],
        "K_e1": traj[:, 4],
        "E_K1": traj[:, 5],
        "s1": traj[:, 6],
        "sbar1": traj[:, 7],
        "V2": traj[:, 8],
        "a2": traj[:, 9],
        "I_pump2": traj[:, 10],
        "I_tot2": traj[:, 11],
        "K_e2": traj[:, 12],
        "E_K2": traj[:, 13],
        "s2": traj[:, 14],
        "sbar2": traj[:, 15],
        "w12": w12,
        "w21": w21,
        "D_ke": D_ke,
        "ion_on": ion_on,
        "a_max_1": a_max1_f,
        "a_max_2": a_max2_f,
        "tau_syn": tau_syn,
        "params": params,
        "I1": I1_t,
        "I2": I2_t,
    }

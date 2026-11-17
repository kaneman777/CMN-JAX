"""Minimal 3-cell nets (Phase D1/D2 synaptic; Phase E1 chain + Ke).

Chain: edges 1↔2 and 2↔3 (``w13=w31=0``).
Triangle: also 1↔3 (synaptic only; no triangle Ke bus here).
E1: ``simulate_three_cell_chain_ke`` adds local Ke with diffusion along
chain edges only (1↔2, 2↔3; never 1↔3).
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
from jnpnp.models.point_sim import DT, NA_I0, V0, atp_gNa_scale, merge_params
from jnpnp.models.two_cell import _cell_y0, _step_cell


def simulate_three_cell_chain(
    I_drive_t: Any,
    *,
    w12: float,
    w21: float,
    w23: float,
    w32: float,
    w13: float = 0.0,
    w31: float = 0.0,
    dt: float = DT,
    t_end: float | None = None,
    params: dict[str, float] | None = None,
    fix_a: bool = False,
    V0_1: float = V0,
    V0_2: float = V0,
    V0_3: float = V0,
    a0_1: float | None = None,
    a0_2: float | None = None,
    a0_3: float | None = None,
    a_max_1: float | None = None,
    a_max_2: float | None = None,
    a_max_3: float | None = None,
    V_th: float = 0.0,
    k_soft: float = 0.5,
    tau_syn: float = 5.0,
) -> dict[str, Any]:
    """Integrate three point cells (chain or triangle).

    Parameters
    ----------
    I_drive_t :
        Shared external drive ``(n_steps+1,)``.
    w12, w21 :
        Synapses on edge 1–2 (into 1 from 2 / into 2 from 1).
    w23, w32 :
        Synapses on edge 2–3 (into 2 from 3 / into 3 from 2).
    w13, w31 :
        Synapses on edge 1–3 (default 0 = chain; nonzero = triangle).
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
    a_max3_f = float(a_max_3) if a_max_3 is not None else a_max_default
    if a0_1 is None:
        a0_1 = a_max1_f
    if a0_2 is None:
        a0_2 = a_max2_f
    if a0_3 is None:
        a0_3 = a_max3_f

    y1_0 = _cell_y0(V0_cell=float(V0_1), a0_cell=float(a0_1))
    y2_0 = _cell_y0(V0_cell=float(V0_2), a0_cell=float(a0_2))
    y3_0 = _cell_y0(V0_cell=float(V0_3), a0_cell=float(a0_3))
    y0 = jnp.concatenate(
        [
            y1_0,
            y2_0,
            y3_0,
            jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64),
        ]
    )

    dt_j = jnp.asarray(dt, dtype=jnp.float64)
    ek_j = jnp.asarray(E_K(params["K_o"], params["K_i"]), dtype=jnp.float64)
    fix_a_j = jnp.asarray(fix_a)
    a_max1 = jnp.asarray(a_max1_f, dtype=jnp.float64)
    a_max2 = jnp.asarray(a_max2_f, dtype=jnp.float64)
    a_max3 = jnp.asarray(a_max3_f, dtype=jnp.float64)
    na_o = params["Na_o"]
    k_o = params["K_o"]
    w12_j = jnp.asarray(w12, dtype=jnp.float64)
    w21_j = jnp.asarray(w21, dtype=jnp.float64)
    w23_j = jnp.asarray(w23, dtype=jnp.float64)
    w32_j = jnp.asarray(w32, dtype=jnp.float64)
    w13_j = jnp.asarray(w13, dtype=jnp.float64)
    w31_j = jnp.asarray(w31, dtype=jnp.float64)
    V_th_j = jnp.asarray(V_th, dtype=jnp.float64)
    k_soft_j = jnp.asarray(k_soft, dtype=jnp.float64)
    tau_j = jnp.asarray(max(tau_syn, 1e-6), dtype=jnp.float64)

    def soft(V: Any) -> Any:
        return 1.0 / (1.0 + jnp.exp(-k_soft_j * (V - V_th_j)))

    def step(y, inputs):
        _t, I_drive = inputs
        y1 = y[0:6]
        y2 = y[6:12]
        y3 = y[12:18]
        sbar1, sbar2, sbar3 = y[18], y[19], y[20]

        s1 = soft(y1[0])
        s2 = soft(y2[0])
        s3 = soft(y3[0])
        I1 = I_drive + w12_j * sbar2 + w13_j * sbar3
        I2 = I_drive + w21_j * sbar1 + w23_j * sbar3
        I3 = I_drive + w32_j * sbar2 + w31_j * sbar1

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
        y3_n, o3 = _step_cell(
            y3,
            I3,
            dt_j=dt_j,
            ek_j=ek_j,
            fix_a_j=fix_a_j,
            a_max=a_max3,
            na_o=na_o,
            k_o=k_o,
            params=params,
        )
        sbar1_n = sbar1 + dt_j * (-sbar1 / tau_j + s1)
        sbar2_n = sbar2 + dt_j * (-sbar2 / tau_j + s2)
        sbar3_n = sbar3 + dt_j * (-sbar3 / tau_j + s3)
        y_new = jnp.concatenate(
            [y1_n, y2_n, y3_n, jnp.stack([sbar1_n, sbar2_n, sbar3_n])]
        )
        out = jnp.concatenate(
            [
                jnp.concatenate([o1, jnp.stack([s1, sbar1])]),
                jnp.concatenate([o2, jnp.stack([s2, sbar2])]),
                jnp.concatenate([o3, jnp.stack([s3, sbar3])]),
            ]
        )
        return y_new, out

    scan_in = (times[:-1], I_drive_t[:-1])
    y_final, traj = lax.scan(step, y0, scan_in)

    def finalize(yf: Any, Ie_f: Any) -> Any:
        V_f, _m, _h, _n, Na_f, a_f = yf
        s = soft(V_f)
        I_p = pump_current(a_f, Na_f, k_o, params)
        return jnp.stack([V_f, a_f, I_p, Ie_f, s])

    y1_f, y2_f, y3_f = y_final[0:6], y_final[6:12], y_final[12:18]
    sbar1_f, sbar2_f, sbar3_f = y_final[18], y_final[19], y_final[20]
    I1_f = I_drive_t[-1] + w12_j * sbar2_f + w13_j * sbar3_f
    I2_f = I_drive_t[-1] + w21_j * sbar1_f + w23_j * sbar3_f
    I3_f = I_drive_t[-1] + w32_j * sbar2_f + w31_j * sbar1_f
    fin1 = finalize(y1_f, I1_f)
    fin2 = finalize(y2_f, I2_f)
    fin3 = finalize(y3_f, I3_f)
    last = jnp.concatenate(
        [
            jnp.concatenate([fin1, jnp.stack([sbar1_f])]),
            jnp.concatenate([fin2, jnp.stack([sbar2_f])]),
            jnp.concatenate([fin3, jnp.stack([sbar3_f])]),
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
        "V3": traj[:, 12],
        "a3": traj[:, 13],
        "I_pump3": traj[:, 14],
        "I_tot3": traj[:, 15],
        "s3": traj[:, 16],
        "sbar3": traj[:, 17],
        "w12": w12,
        "w21": w21,
        "w23": w23,
        "w32": w32,
        "w13": w13,
        "w31": w31,
        "a_max_1": a_max1_f,
        "a_max_2": a_max2_f,
        "a_max_3": a_max3_f,
        "tau_syn": tau_syn,
        "params": params,
        "I_drive": I_drive_t,
    }


def simulate_three_cell_chain_ke(
    I_drive_t: Any,
    *,
    w12: float,
    w21: float,
    w23: float,
    w32: float,
    D_ke: float = 0.0,
    ion_on: bool = True,
    dt: float = DT,
    t_end: float | None = None,
    params: dict[str, float] | None = None,
    fix_a: bool = False,
    V0_1: float = V0,
    V0_2: float = V0,
    V0_3: float = V0,
    a0_1: float | None = None,
    a0_2: float | None = None,
    a0_3: float | None = None,
    a_max_1: float | None = None,
    a_max_2: float | None = None,
    a_max_3: float | None = None,
    V_th: float = 0.0,
    k_soft: float = 0.5,
    tau_syn: float = 5.0,
) -> dict[str, Any]:
    """3-cell **chain** with local Ke and edge diffusion (Phase E1).

    Synapses and Ke diffusion only on 1↔2 and 2↔3 (no 1–3 bus / no triangle).
    ``D_ke`` [1/ms]: each edge contributes ``D*(Ke_nbr - Ke_i)``.
    Middle cell sums both neighbors.
    """
    if params is None:
        params = merge_params()
    params = dict(params)
    if ion_on and float(params.get("gamma_Ke", 0.0)) == 0.0:
        params["gamma_Ke"] = 0.0005
    if float(params.get("tau_K", 0.0)) <= 0.0:
        params["tau_K"] = 1000.0

    I_drive_t = jnp.asarray(I_drive_t, dtype=jnp.float64)
    n_steps = int(I_drive_t.shape[0]) - 1
    if t_end is None:
        t_end = n_steps * dt
    times = jnp.arange(n_steps + 1, dtype=jnp.float64) * dt

    bath = float(params.get("K_e_bath", params.get("K_o", 4.0)))
    a_max_default = float(params.get("a_max", 1.0))
    a_max1_f = float(a_max_1) if a_max_1 is not None else a_max_default
    a_max2_f = float(a_max_2) if a_max_2 is not None else a_max_default
    a_max3_f = float(a_max_3) if a_max_3 is not None else a_max_default
    if a0_1 is None:
        a0_1 = a_max1_f
    if a0_2 is None:
        a0_2 = a_max2_f
    if a0_3 is None:
        a0_3 = a_max3_f

    def y0_cell(V0c: float, a0c: float) -> Any:
        m0, h0, n0 = steady_gates(jnp.asarray(V0c, dtype=jnp.float64))
        return jnp.array(
            [V0c, m0, h0, n0, NA_I0, float(a0c), bath], dtype=jnp.float64
        )

    y0 = jnp.concatenate(
        [
            y0_cell(float(V0_1), float(a0_1)),
            y0_cell(float(V0_2), float(a0_2)),
            y0_cell(float(V0_3), float(a0_3)),
            jnp.zeros(3, dtype=jnp.float64),
        ]
    )

    dt_j = jnp.asarray(dt, dtype=jnp.float64)
    fix_a_j = jnp.asarray(fix_a)
    ion_on_j = jnp.asarray(ion_on)
    a_max1 = jnp.asarray(a_max1_f, dtype=jnp.float64)
    a_max2 = jnp.asarray(a_max2_f, dtype=jnp.float64)
    a_max3 = jnp.asarray(a_max3_f, dtype=jnp.float64)
    bath_j = jnp.asarray(bath, dtype=jnp.float64)
    D_j = jnp.asarray(D_ke, dtype=jnp.float64)
    na_o = params["Na_o"]
    k_i = params["K_i"]
    w12_j = jnp.asarray(w12, dtype=jnp.float64)
    w21_j = jnp.asarray(w21, dtype=jnp.float64)
    w23_j = jnp.asarray(w23, dtype=jnp.float64)
    w32_j = jnp.asarray(w32, dtype=jnp.float64)
    V_th_j = jnp.asarray(V_th, dtype=jnp.float64)
    k_soft_j = jnp.asarray(k_soft, dtype=jnp.float64)
    tau_j = jnp.asarray(max(tau_syn, 1e-6), dtype=jnp.float64)

    def soft(V: Any) -> Any:
        return 1.0 / (1.0 + jnp.exp(-k_soft_j * (V - V_th_j)))

    def step_one(
        y: Any,
        Ie: Any,
        dke_diff: Any,
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
        Ke_new = jnp.where(
            ion_on_j,
            clip_Ke(K_e + dt_j * (dke_local + dke_diff)),
            bath_j,
        )
        y_new = jnp.stack([V_new, m_new, h_new, n_new, Na_new, a_new, Ke_new])
        out = jnp.stack([V, a_eff, I_p, Ie, K_eff, ek])
        return y_new, out

    def step(y, inputs):
        _t, I_drive = inputs
        y1 = y[0:7]
        y2 = y[7:14]
        y3 = y[14:21]
        sbar1, sbar2, sbar3 = y[21], y[22], y[23]
        Ke1, Ke2, Ke3 = y1[6], y2[6], y3[6]

        s1 = soft(y1[0])
        s2 = soft(y2[0])
        s3 = soft(y3[0])
        # chain synapses only
        I1 = I_drive + w12_j * sbar2
        I2 = I_drive + w21_j * sbar1 + w23_j * sbar3
        I3 = I_drive + w32_j * sbar2
        # chain Ke diffusion only (no 1–3)
        d1 = D_j * (Ke2 - Ke1)
        d2 = D_j * (Ke1 - Ke2) + D_j * (Ke3 - Ke2)
        d3 = D_j * (Ke2 - Ke3)

        y1_n, o1 = step_one(y1, I1, d1, a_max=a_max1)
        y2_n, o2 = step_one(y2, I2, d2, a_max=a_max2)
        y3_n, o3 = step_one(y3, I3, d3, a_max=a_max3)
        sbar1_n = sbar1 + dt_j * (-sbar1 / tau_j + s1)
        sbar2_n = sbar2 + dt_j * (-sbar2 / tau_j + s2)
        sbar3_n = sbar3 + dt_j * (-sbar3 / tau_j + s3)
        y_new = jnp.concatenate(
            [y1_n, y2_n, y3_n, jnp.stack([sbar1_n, sbar2_n, sbar3_n])]
        )
        out = jnp.concatenate(
            [
                jnp.concatenate([o1, jnp.stack([s1, sbar1])]),
                jnp.concatenate([o2, jnp.stack([s2, sbar2])]),
                jnp.concatenate([o3, jnp.stack([s3, sbar3])]),
            ]
        )
        return y_new, out

    scan_in = (times[:-1], I_drive_t[:-1])
    y_final, traj = lax.scan(step, y0, scan_in)

    def fin_cell(yf: Any, Ie_f: Any, a_max: Any) -> Any:
        V_f, _m, _h, _n, Na_f, a_f, Ke_f = yf
        a_eff = jnp.where(fix_a_j, a_max, a_f)
        K_eff = jnp.where(ion_on_j, Ke_f, bath_j)
        ek = E_K(K_eff, k_i)
        I_p = pump_current(a_eff, Na_f, K_eff, params)
        s = soft(V_f)
        return jnp.stack([V_f, a_eff, I_p, Ie_f, K_eff, ek, s])

    y1_f, y2_f, y3_f = y_final[0:7], y_final[7:14], y_final[14:21]
    sbar1_f, sbar2_f, sbar3_f = y_final[21], y_final[22], y_final[23]
    I1_f = I_drive_t[-1] + w12_j * sbar2_f
    I2_f = I_drive_t[-1] + w21_j * sbar1_f + w23_j * sbar3_f
    I3_f = I_drive_t[-1] + w32_j * sbar2_f
    fin1 = fin_cell(y1_f, I1_f, a_max1)
    fin2 = fin_cell(y2_f, I2_f, a_max2)
    fin3 = fin_cell(y3_f, I3_f, a_max3)
    # traj step out: V,a,Ip,Ie,Ke,ek + s,sbar = 8 per cell
    last = jnp.concatenate(
        [
            jnp.concatenate([fin1[:6], jnp.stack([fin1[6], sbar1_f])]),
            jnp.concatenate([fin2[:6], jnp.stack([fin2[6], sbar2_f])]),
            jnp.concatenate([fin3[:6], jnp.stack([fin3[6], sbar3_f])]),
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
        "V3": traj[:, 16],
        "a3": traj[:, 17],
        "I_pump3": traj[:, 18],
        "I_tot3": traj[:, 19],
        "K_e3": traj[:, 20],
        "E_K3": traj[:, 21],
        "s3": traj[:, 22],
        "sbar3": traj[:, 23],
        "w12": w12,
        "w21": w21,
        "w23": w23,
        "w32": w32,
        "D_ke": D_ke,
        "params": params,
        "I_drive": I_drive_t,
    }

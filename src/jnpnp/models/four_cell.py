"""Minimal 4-cell star / Y (Phase D3): hub + 3 leaves, synaptic only.

Edges are hub–leaf only (no leaf–leaf). K_e diffusion not included (D≡0).
Default brakes stay off via ``merge_params()``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import lax

from jnpnp.models.ions_atp import E_K, pump_current
from jnpnp.models.point_sim import DT, V0, merge_params
from jnpnp.models.two_cell import _cell_y0, _step_cell


def simulate_four_cell_star(
    I_drive_t: Any,
    *,
    w_hub_from_leaf: float,
    w_leaf_from_hub: float,
    dt: float = DT,
    t_end: float | None = None,
    params: dict[str, float] | None = None,
    fix_a: bool = False,
    V0_h: float = V0,
    V0_l1: float = V0,
    V0_l2: float = V0,
    V0_l3: float = V0,
    a0_h: float | None = None,
    a0_l1: float | None = None,
    a0_l2: float | None = None,
    a0_l3: float | None = None,
    a_max: float | None = None,
    V_th: float = 0.0,
    k_soft: float = 0.5,
    tau_syn: float = 5.0,
) -> dict[str, Any]:
    """Integrate hub H + leaves L1,L2,L3 (star).

    Parameters
    ----------
    w_hub_from_leaf :
        Synaptic weight into hub from each leaf (same on all arms).
    w_leaf_from_hub :
        Synaptic weight into each leaf from hub.
    """
    if params is None:
        params = merge_params()
    I_drive_t = jnp.asarray(I_drive_t, dtype=jnp.float64)
    n_steps = int(I_drive_t.shape[0]) - 1
    if t_end is None:
        t_end = n_steps * dt
    times = jnp.arange(n_steps + 1, dtype=jnp.float64) * dt

    a_max_f = float(params.get("a_max", 1.0) if a_max is None else a_max)
    if a0_h is None:
        a0_h = a_max_f
    if a0_l1 is None:
        a0_l1 = a_max_f
    if a0_l2 is None:
        a0_l2 = a_max_f
    if a0_l3 is None:
        a0_l3 = a_max_f

    y0 = jnp.concatenate(
        [
            _cell_y0(V0_cell=float(V0_h), a0_cell=float(a0_h)),
            _cell_y0(V0_cell=float(V0_l1), a0_cell=float(a0_l1)),
            _cell_y0(V0_cell=float(V0_l2), a0_cell=float(a0_l2)),
            _cell_y0(V0_cell=float(V0_l3), a0_cell=float(a0_l3)),
            jnp.zeros(4, dtype=jnp.float64),  # sbar H,L1,L2,L3
        ]
    )

    dt_j = jnp.asarray(dt, dtype=jnp.float64)
    ek_j = jnp.asarray(E_K(params["K_o"], params["K_i"]), dtype=jnp.float64)
    fix_a_j = jnp.asarray(fix_a)
    a_max_j = jnp.asarray(a_max_f, dtype=jnp.float64)
    na_o = params["Na_o"]
    k_o = params["K_o"]
    w_hl = jnp.asarray(w_hub_from_leaf, dtype=jnp.float64)
    w_lh = jnp.asarray(w_leaf_from_hub, dtype=jnp.float64)
    V_th_j = jnp.asarray(V_th, dtype=jnp.float64)
    k_soft_j = jnp.asarray(k_soft, dtype=jnp.float64)
    tau_j = jnp.asarray(max(tau_syn, 1e-6), dtype=jnp.float64)

    def soft(V: Any) -> Any:
        return 1.0 / (1.0 + jnp.exp(-k_soft_j * (V - V_th_j)))

    def step_one(y: Any, Ie: Any) -> tuple[Any, Any]:
        return _step_cell(
            y,
            Ie,
            dt_j=dt_j,
            ek_j=ek_j,
            fix_a_j=fix_a_j,
            a_max=a_max_j,
            na_o=na_o,
            k_o=k_o,
            params=params,
        )

    def step(y, inputs):
        _t, I_drive = inputs
        yh = y[0:6]
        yl1 = y[6:12]
        yl2 = y[12:18]
        yl3 = y[18:24]
        sbh, sbl1, sbl2, sbl3 = y[24], y[25], y[26], y[27]

        sh = soft(yh[0])
        sl1 = soft(yl1[0])
        sl2 = soft(yl2[0])
        sl3 = soft(yl3[0])
        # star only: no leaf–leaf
        Ih = I_drive + w_hl * (sbl1 + sbl2 + sbl3)
        Il1 = I_drive + w_lh * sbh
        Il2 = I_drive + w_lh * sbh
        Il3 = I_drive + w_lh * sbh

        yh_n, oh = step_one(yh, Ih)
        yl1_n, ol1 = step_one(yl1, Il1)
        yl2_n, ol2 = step_one(yl2, Il2)
        yl3_n, ol3 = step_one(yl3, Il3)
        sbh_n = sbh + dt_j * (-sbh / tau_j + sh)
        sbl1_n = sbl1 + dt_j * (-sbl1 / tau_j + sl1)
        sbl2_n = sbl2 + dt_j * (-sbl2 / tau_j + sl2)
        sbl3_n = sbl3 + dt_j * (-sbl3 / tau_j + sl3)
        y_new = jnp.concatenate(
            [
                yh_n,
                yl1_n,
                yl2_n,
                yl3_n,
                jnp.stack([sbh_n, sbl1_n, sbl2_n, sbl3_n]),
            ]
        )
        out = jnp.concatenate(
            [
                jnp.concatenate([oh, jnp.stack([sh, sbh])]),
                jnp.concatenate([ol1, jnp.stack([sl1, sbl1])]),
                jnp.concatenate([ol2, jnp.stack([sl2, sbl2])]),
                jnp.concatenate([ol3, jnp.stack([sl3, sbl3])]),
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

    yh_f, yl1_f = y_final[0:6], y_final[6:12]
    yl2_f, yl3_f = y_final[12:18], y_final[18:24]
    sbh_f, sbl1_f, sbl2_f, sbl3_f = y_final[24], y_final[25], y_final[26], y_final[27]
    Ih_f = I_drive_t[-1] + w_hl * (sbl1_f + sbl2_f + sbl3_f)
    Il1_f = I_drive_t[-1] + w_lh * sbh_f
    Il2_f = I_drive_t[-1] + w_lh * sbh_f
    Il3_f = I_drive_t[-1] + w_lh * sbh_f
    last = jnp.concatenate(
        [
            jnp.concatenate([finalize(yh_f, Ih_f), jnp.stack([sbh_f])]),
            jnp.concatenate([finalize(yl1_f, Il1_f), jnp.stack([sbl1_f])]),
            jnp.concatenate([finalize(yl2_f, Il2_f), jnp.stack([sbl2_f])]),
            jnp.concatenate([finalize(yl3_f, Il3_f), jnp.stack([sbl3_f])]),
        ]
    )[None, :]
    traj = jnp.concatenate([traj, last], axis=0)

    return {
        "t": times,
        "VH": traj[:, 0],
        "aH": traj[:, 1],
        "I_pumpH": traj[:, 2],
        "I_totH": traj[:, 3],
        "sH": traj[:, 4],
        "sbarH": traj[:, 5],
        "VL1": traj[:, 6],
        "aL1": traj[:, 7],
        "sL1": traj[:, 10],
        "sbarL1": traj[:, 11],
        "VL2": traj[:, 12],
        "aL2": traj[:, 13],
        "sL2": traj[:, 16],
        "sbarL2": traj[:, 17],
        "VL3": traj[:, 18],
        "aL3": traj[:, 19],
        "sL3": traj[:, 22],
        "sbarL3": traj[:, 23],
        "w_hub_from_leaf": w_hub_from_leaf,
        "w_leaf_from_hub": w_leaf_from_hub,
        "params": params,
        "I_drive": I_drive_t,
    }

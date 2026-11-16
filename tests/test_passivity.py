"""No free lunch / rest sanity (not a closed thermodynamic proof).

Does **not** require V → 0 mV. HH + leak may sit near a resting potential
(≈ −65 mV with defaults). We only ask: finite states, no spontaneous barrage
when ATP is clamped depleted and I_ext = 0.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jnpnp.models.point_sim import DT, merge_params, soft_spike, simulate_driven


def _zeros(t_end: float, dt: float = DT) -> jnp.ndarray:
    n = int(round(t_end / dt))
    return jnp.zeros(n + 1, dtype=jnp.float64)


def test_unstimulated_short_horizon_finite():
    """Healthy params, I_ext=0: states stay finite (no blow-up)."""
    I = _zeros(80.0)
    traj = simulate_driven(I, params=merge_params(), dt=DT)
    for key in ("V", "Na_i", "a"):
        x = np.asarray(traj[key])
        assert np.all(np.isfinite(x)), key
    # Soft resting proxy should stay well below a driven mean-spike regime
    mean_s = float(np.mean(np.asarray(soft_spike(traj["V"]))))
    assert mean_s < 0.05


def test_depleted_atp_no_spontaneous_barrage():
    """fix_a with tiny a_max ≈ ATP floor: no spontaneous firing without drive.

    This is a perpetual-motion / passivity smoke check, not a Lyapunov proof.
    """
    I = _zeros(200.0)
    params = merge_params(a_max=1e-3)
    traj = simulate_driven(I, fix_a=True, params=params, dt=DT)
    v = np.asarray(traj["V"])
    a = np.asarray(traj["a"])
    assert np.all(np.isfinite(v))
    assert np.all(np.isfinite(a))
    assert float(a.max()) <= 1e-3 + 1e-9
    mean_s = float(np.mean(np.asarray(soft_spike(traj["V"]))))
    # Resting soft-spike under defaults is ≪ 0.01; leave headroom
    assert mean_s < 0.02

"""Forward smoke test: spike train -> Na load -> pump -> ATP drop -> firing suppression.

Run from the repo root after ``pip install -e ".[dev]"``::

    python scripts/00_forward_smoke.py

Saves ``figures/forward_smoke.png`` (V, Na_i, a, I_pump) with an a≡1 control overlay.
Uses CPU JAX with ``jax_enable_x64=True`` and a fixed-step exponential-Euler / Euler loop
(docs/model.md). No training, networks, or spatial PNP.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np

from jnpnp.models.point_sim import (
    DT,
    I0,
    T_OFF,
    T_ON,
    simulate_trajectory,
)


def simulate(*, fix_a: bool = False) -> dict[str, np.ndarray]:
    """Phase-1 forward trajectory (numpy arrays for plotting)."""
    traj = simulate_trajectory(stim_gain=1.0, fix_a=fix_a)
    return {
        "t": np.asarray(traj["t"]),
        "V": np.asarray(traj["V"]),
        "Na_i": np.asarray(traj["Na_i"]),
        "a": np.asarray(traj["a"]),
        "I_pump": np.asarray(traj["I_pump"]),
        "I_ext": np.asarray(traj["I_ext"]),
        "E_Na": np.asarray(traj["E_Na"]),
        "params": traj["params"],
        "E_K": float(traj["E_K"]),
    }


def detect_spikes(t: np.ndarray, V: np.ndarray, thr: float = 0.0) -> np.ndarray:
    """Return spike times as upward crossings of thr."""
    above = V >= thr
    crossings = np.where(~above[:-1] & above[1:])[0]
    return t[crossings]


def plot_results(
    dyn: dict[str, np.ndarray],
    ctrl: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    t = dyn["t"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(t, dyn["V"], color="C0", lw=0.9, label="dynamic a")
    ax.plot(t, ctrl["V"], color="C0", lw=0.7, alpha=0.45, ls="--", label="a≡1")
    spikes = detect_spikes(t, dyn["V"])
    if spikes.size:
        ax.plot(spikes, np.full_like(spikes, 45.0), "|", color="k", ms=8, label="spikes")
    ax.set_ylabel("V [mV]")
    ax.set_title("Phase 1 forward smoke: HH + Na + pump + ATP")
    ax.axvspan(T_ON, T_OFF, color="0.92", zorder=0)
    ax.axvspan(250.0, T_OFF, color="#f5e6c8", alpha=0.7, zorder=0, label="late / suppress")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(t, dyn["Na_i"], color="C1", lw=1.2, label="dynamic a")
    ax.plot(t, ctrl["Na_i"], color="C1", lw=1.0, alpha=0.45, ls="--", label="a≡1")
    ax.set_ylabel(r"Na$_i$ [mM]")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    ax.plot(t, dyn["a"], color="C2", lw=1.2, label="dynamic a")
    ax.plot(t, ctrl["a"], color="C2", lw=1.0, alpha=0.45, ls="--", label="a≡1")
    ax.set_ylabel("a [ATP]")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[3]
    ax.plot(t, dyn["I_pump"], color="C3", lw=1.2, label=r"$I_{\mathrm{pump}}$ dyn")
    ax.plot(t, ctrl["I_pump"], color="C3", lw=1.0, alpha=0.45, ls="--", label=r"$I_{\mathrm{pump}}$ a≡1")
    ax.plot(t, dyn["I_ext"], color="0.4", lw=0.8, label=r"$I_{\mathrm{ext}}$")
    ax.set_ylabel(r"I [µA/cm²]")
    ax.set_xlabel("t [ms]")
    ax.legend(loc="upper right", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    print("Simulating dynamic ATP ...")
    dyn = simulate(fix_a=False)
    print("Simulating a≡1 control ...")
    ctrl = simulate(fix_a=True)

    out_path = Path("figures") / "forward_smoke.png"
    plot_results(dyn, ctrl, out_path)

    spikes_dyn = detect_spikes(dyn["t"], dyn["V"])
    spikes_ctrl = detect_spikes(ctrl["t"], ctrl["V"])
    p = dyn["params"]
    print("--- summary ---")
    print(f"saved: {out_path.resolve()}")
    print(
        f"params: gamma_Na={p['gamma_Na']}, I_pump_bar={p['I_pump_bar']}, "
        f"eta_prod={p['eta_prod']}, eta_cost={p['eta_cost']}, "
        f"K_Na={p['K_Na']}, K_K={p['K_K']}, I0={I0}, dt={DT}"
    )
    print(f"Na_i: {dyn['Na_i'][0]:.2f} -> {dyn['Na_i'][-1]:.2f} mM (dyn)")
    print(f"a:    {dyn['a'][0]:.3f} -> {dyn['a'].min():.3f} min / {dyn['a'][-1]:.3f} end")
    print(f"spikes: dyn={spikes_dyn.size}, a≡1={spikes_ctrl.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

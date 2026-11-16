"""Phase 3b — RQ2 minimal: slow local K_e relaxation as a short-term trace.

Run from the repo root::

    python scripts/05_rq2_ke_memory.py

Step B0: pulse train → K_e / E_K / V (``figures/rq2_ke_trace.png``).
Step B1: Protocol X cue → delay → weak probe; ion-on vs ion-off
(``figures/rq2_delay_probe.png``).

Claim ceiling: ``slow K_e as a short-term trace preview`` — not a WM implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jnpnp.models.point_sim import DT, merge_params, soft_spike, simulate_driven_ke

# ---------------------------------------------------------------------------
# Tuned Phase-3b defaults
# ---------------------------------------------------------------------------
GAMMA_KE = 0.0005
TAU_K = 1000.0  # ms
K_BATH = 4.0

# B0 smoke
B0_T = 800.0
B0_AMP = 12.0
B0_TON = 50.0
B0_TOFF = 120.0

# B1 Protocol X
B1_T = 400.0
CUE_TON = 20.0
CUE_TOFF = 45.0
CUE_STRONG = 15.0
CUE_WEAK = 0.0
DELAY_MS = 120.0  # cue end → probe start
PROBE_DUR = 40.0
PROBE_AMP = 5.0  # near-threshold: elevated K_e boosts probe firing
V_TH = 0.0
K_SOFT = 0.5


def ke_params(**overrides: float) -> dict[str, float]:
    p = merge_params()
    p["K_e_bath"] = K_BATH
    p["K_o"] = K_BATH
    p["gamma_Ke"] = GAMMA_KE
    p["tau_K"] = TAU_K
    p.update(overrides)
    return p


def make_I(t: jnp.ndarray, segments: list[tuple[float, float, float]]) -> jnp.ndarray:
    """Piecewise-constant I_ext from (ton, toff, amp) segments."""
    I = jnp.zeros_like(t)
    for ton, toff, amp in segments:
        I = I + jnp.where((t >= ton) & (t < toff), amp, 0.0)
    return I


def time_grid(t_end: float, dt: float = DT) -> jnp.ndarray:
    n = int(round(t_end / dt))
    return jnp.arange(n + 1, dtype=jnp.float64) * dt


def run_b0(out_path: Path) -> dict:
    print("--- B0: K_e trace smoke ---")
    t = time_grid(B0_T)
    I = make_I(t, [(B0_TON, B0_TOFF, B0_AMP)])
    params = ke_params()
    traj_on = simulate_driven_ke(I, ion_on=True, dt=DT, t_end=B0_T, params=params)
    traj_off = simulate_driven_ke(I, ion_on=False, dt=DT, t_end=B0_T, params=params)

    t_np = np.asarray(traj_on["t"])
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    axes[0].plot(t_np, np.asarray(traj_on["V"]), color="C0", lw=0.8, label="ion-on")
    axes[0].plot(t_np, np.asarray(traj_off["V"]), color="0.5", lw=0.7, alpha=0.7, ls="--", label="ion-off")
    axes[0].axvspan(B0_TON, B0_TOFF, color="0.9", zorder=0)
    axes[0].set_ylabel("V [mV]")
    axes[0].set_title(rf"B0: local $K_e$ trace ($\tau_K$={TAU_K:g} ms, $\gamma_{{Ke}}$={GAMMA_KE})")
    axes[0].legend(fontsize=8)

    axes[1].plot(t_np, np.asarray(traj_on["K_e"]), color="C1", lw=1.2, label="ion-on")
    axes[1].plot(t_np, np.asarray(traj_off["K_e"]), color="0.5", lw=1.0, ls="--", label="ion-off")
    axes[1].axhline(K_BATH, color="0.3", ls=":", lw=0.8)
    axes[1].set_ylabel(r"$K_e$ [mM]")
    axes[1].legend(fontsize=8)

    axes[2].plot(t_np, np.asarray(traj_on["E_K"]), color="C2", lw=1.2, label="ion-on")
    axes[2].plot(t_np, np.asarray(traj_off["E_K"]), color="0.5", lw=1.0, ls="--", label="ion-off")
    axes[2].set_ylabel(r"$E_K$ [mV]")
    axes[2].legend(fontsize=8)

    axes[3].plot(t_np, np.asarray(traj_on["I_ext"]), color="0.3", lw=0.9)
    axes[3].set_ylabel(r"$I_{\mathrm{ext}}$")
    axes[3].set_xlabel("t [ms]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    ke = np.asarray(traj_on["K_e"])
    print(
        f"K_e: start={ke[0]:.3f} max={ke.max():.3f} end={ke[-1]:.3f} "
        f"(bath={K_BATH}); saved {out_path}"
    )
    return {"on": traj_on, "off": traj_off}


def probe_metrics(traj: dict, probe_ton: float, probe_toff: float) -> dict[str, float]:
    t = np.asarray(traj["t"])
    mask = (t >= probe_ton) & (t < probe_toff)
    V = np.asarray(traj["V"])
    s = np.asarray(soft_spike(traj["V"], V_th=V_TH, k=K_SOFT))
    Ke = np.asarray(traj["K_e"])
    return {
        "mean_s_probe": float(s[mask].mean()) if mask.any() else float("nan"),
        "max_V_probe": float(V[mask].max()) if mask.any() else float("nan"),
        "K_e_at_probe": float(Ke[mask][0]) if mask.any() else float("nan"),
        "K_e_mean_probe": float(Ke[mask].mean()) if mask.any() else float("nan"),
    }


def run_b1(out_path: Path) -> dict:
    print("--- B1: Protocol X cue → delay → probe ---")
    t = time_grid(B1_T)
    probe_ton = CUE_TOFF + DELAY_MS
    probe_toff = probe_ton + PROBE_DUR
    params = ke_params()

    conditions = {
        "strong": CUE_STRONG,
        "weak": CUE_WEAK,
    }
    results: dict[str, dict] = {}
    for cue_name, cue_amp in conditions.items():
        I = make_I(
            t,
            [
                (CUE_TON, CUE_TOFF, cue_amp),
                (probe_ton, probe_toff, PROBE_AMP),
            ],
        )
        for ion_label, ion_on in (("on", True), ("off", False)):
            key = f"{cue_name}_{ion_label}"
            traj = simulate_driven_ke(
                I, ion_on=ion_on, dt=DT, t_end=B1_T, params=params
            )
            m = probe_metrics(traj, probe_ton, probe_toff)
            results[key] = {"traj": traj, "metrics": m, "I": np.asarray(I)}
            print(
                f"  {key}: mean_s={m['mean_s_probe']:.5f} maxV={m['max_V_probe']:.2f} "
                f"Ke@probe={m['K_e_at_probe']:.3f}"
            )

    # Figure: V and K_e for strong/weak × on/off (focus panels)
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex="col", constrained_layout=True)
    for col, cue_name in enumerate(("strong", "weak")):
        for ion_label, color, ls in (("on", "C0", "-"), ("off", "C3", "--")):
            key = f"{cue_name}_{ion_label}"
            tr = results[key]["traj"]
            tt = np.asarray(tr["t"])
            axes[0, col].plot(
                tt, np.asarray(tr["V"]), color=color, ls=ls, lw=0.8, label=f"ion-{ion_label}"
            )
            axes[1, col].plot(
                tt, np.asarray(tr["K_e"]), color=color, ls=ls, lw=1.1, label=f"ion-{ion_label}"
            )
        axes[0, col].axvspan(CUE_TON, CUE_TOFF, color="#d0e8ff", zorder=0)
        axes[0, col].axvspan(probe_ton, probe_toff, color="#ffe8c8", zorder=0)
        axes[1, col].axvspan(CUE_TON, CUE_TOFF, color="#d0e8ff", zorder=0)
        axes[1, col].axvspan(probe_ton, probe_toff, color="#ffe8c8", zorder=0)
        axes[0, col].set_title(f"cue={cue_name}")
        axes[0, col].set_ylabel("V [mV]")
        axes[1, col].set_ylabel(r"$K_e$ [mM]")
        axes[1, col].axhline(K_BATH, color="0.4", ls=":", lw=0.8)
        axes[0, col].legend(fontsize=7, loc="upper right")
        axes[1, col].legend(fontsize=7, loc="upper right")

        I = results[f"{cue_name}_on"]["I"]
        axes[2, col].plot(np.asarray(t), I, color="0.25", lw=0.9)
        axes[2, col].axvspan(CUE_TON, CUE_TOFF, color="#d0e8ff", zorder=0)
        axes[2, col].axvspan(probe_ton, probe_toff, color="#ffe8c8", zorder=0)
        axes[2, col].set_ylabel(r"$I_{\mathrm{ext}}$")
        axes[2, col].set_xlabel("t [ms]")

    fig.suptitle(
        f"B1 Protocol X: delay={DELAY_MS:g} ms, probe={PROBE_AMP:g} µA/cm² "
        f"(blue=cue, orange=probe)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")

    # Separation scores
    def sep(ion: str) -> float:
        return (
            results[f"strong_{ion}"]["metrics"]["mean_s_probe"]
            - results[f"weak_{ion}"]["metrics"]["mean_s_probe"]
        )

    sep_on, sep_off = sep("on"), sep("off")
    print(f"probe mean_s separation (strong-weak): ion-on={sep_on:.5f}, ion-off={sep_off:.5f}")
    return {
        "results": results,
        "sep_on": sep_on,
        "sep_off": sep_off,
        "probe_ton": probe_ton,
        "probe_toff": probe_toff,
    }


def main() -> int:
    print(
        f"Phase 3b RQ2: gamma_Ke={GAMMA_KE}, tau_K={TAU_K} ms, bath={K_BATH} mM, dt={DT}"
    )
    b0_path = Path("figures") / "rq2_ke_trace.png"
    b1_path = Path("figures") / "rq2_delay_probe.png"
    b0 = run_b0(b0_path)
    b1 = run_b1(b1_path)

    ke_max = float(np.asarray(b0["on"]["K_e"]).max())
    ok_b0 = ke_max > K_BATH + 0.2
    # Cue-dependent probe separation (signed or absolute) larger on ion-on.
    ok_b1 = abs(b1["sep_on"]) > abs(b1["sep_off"]) + 1e-3
    print("--- summary ---")
    print(f"B0 K_e rise visible: {ok_b0} (max={ke_max:.3f})")
    print(
        f"B1 |sep| ion-on > ion-off: {ok_b1} "
        f"(on={b1['sep_on']:.5f} vs off={b1['sep_off']:.5f})"
    )
    if ok_b0 and ok_b1:
        print("OK: slow K_e short-term trace preview (not a WM claim)")
    elif ok_b0:
        print("PARTIAL: B0 ok; B1 weak/absent — record honestly in STATUS")
    else:
        print("WARN: retune gamma_Ke / tau_K / probe")
    return 0


if __name__ == "__main__":
    sys.exit(main())

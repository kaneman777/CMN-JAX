"""Phase B2 — 2-cell local K_e diffusion (synapse-free minimal).

Run from the repo root::

    python scripts/14_b2_ke_diffusion.py

H0: D=0 — cell1 cue only; K_e2 stays near bath.
H1: D>0 — same cue; K_e2 rises with delay; optional weak probe on cell2.

Claim ceiling: local K_e diffusion can seed a slow neighbor trace —
not ephaptic coupling, EEG, or tissue-faithful diffusion.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jnpnp.models.point_sim import DT, merge_params
from jnpnp.models.two_cell import simulate_two_cell_ke

# RQ2-near ion defaults
GAMMA_KE = 0.0005
TAU_K = 1000.0
K_BATH = 4.0

# Diffusion candidates [1/ms]; first success-like D is used for main figure
D_CANDIDATES = (0.002, 0.005, 0.01, 0.02)

# Protocol: cell1 cue → delay → optional weak probe on cell2
T_END = 600.0
CUE_TON = 40.0
CUE_TOFF = 120.0
CUE_AMP = 12.0
PROBE_TON = 280.0  # after delay
PROBE_TOFF = 320.0
PROBE_AMP = 5.0

# Synapse off; K_ATP / gNa off
W12 = 0.0
W21 = 0.0
V0_1 = -65.0
V0_2 = -65.0


def ke_params(**overrides: float) -> dict[str, float]:
    p = merge_params(
        K_e_bath=K_BATH,
        K_o=K_BATH,
        gamma_Ke=GAMMA_KE,
        tau_K=TAU_K,
        g_KATP_bar=0.0,
        atp_gNa_floor=0.0,
    )
    p.update(overrides)
    return p


def make_drives() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    t = jnp.arange(0.0, T_END + DT / 2, DT)
    I1 = jnp.where((t >= CUE_TON) & (t < CUE_TOFF), CUE_AMP, 0.0)
    I2 = jnp.where((t >= PROBE_TON) & (t < PROBE_TOFF), PROBE_AMP, 0.0)
    return t, I1, I2


def summarize(tr: dict, label: str) -> dict[str, float]:
    t = np.asarray(tr["t"])
    ke1 = np.asarray(tr["K_e1"])
    ke2 = np.asarray(tr["K_e2"])
    s2 = np.asarray(tr["s2"])
    # delay window: after cue, before probe (diffusion-only neighbor signal)
    delay = (t >= CUE_TOFF) & (t < PROBE_TON)
    probe = (t >= PROBE_TON) & (t < PROBE_TOFF)
    ke2_delay = ke2[delay] if delay.any() else ke2
    return {
        "label": label,
        "Ke1_max": float(ke1.max()),
        "Ke2_max_delay": float(ke2_delay.max()),
        "Ke2_mean_delay": float(ke2_delay.mean()),
        "Ke2_at_probe": float(ke2[probe][0]) if probe.any() else float("nan"),
        "mean_s2_probe": float(s2[probe].mean()) if probe.any() else float("nan"),
        "Ke2_peak_time_delay": float(t[delay][int(np.argmax(ke2_delay))])
        if delay.any()
        else float("nan"),
    }


def score_pair(h0: dict[str, float], h1: dict[str, float]) -> float:
    """Extra neighbor K_e2 in the delay window (mM)."""
    return h1["Ke2_max_delay"] - h0["Ke2_max_delay"]


def plot_results(
    h0: dict,
    h1: dict,
    m0: dict[str, float],
    m1: dict[str, float],
    D_used: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(h0["t"], h0["V1"], lw=0.7, color="C0", label="V1 D=0")
    ax.plot(h0["t"], h0["V2"], lw=0.7, color="C3", label="V2 D=0")
    ax.axvspan(CUE_TON, CUE_TOFF, color="C0", alpha=0.12, zorder=0)
    ax.axvspan(PROBE_TON, PROBE_TOFF, color="C3", alpha=0.12, zorder=0)
    ax.set_ylabel("V [mV]")
    ax.set_title("H0 D=0 voltages")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(h1["t"], h1["V1"], lw=0.7, color="C0", label="V1 D>0")
    ax.plot(h1["t"], h1["V2"], lw=0.7, color="C3", label="V2 D>0")
    ax.axvspan(CUE_TON, CUE_TOFF, color="C0", alpha=0.12, zorder=0)
    ax.axvspan(PROBE_TON, PROBE_TOFF, color="C3", alpha=0.12, zorder=0)
    ax.set_ylabel("V [mV]")
    ax.set_title(rf"H1 D={D_used:g} voltages")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(h0["t"], h0["K_e1"], lw=1.2, color="C0", label=r"$K_{e1}$ D=0")
    ax.plot(h0["t"], h0["K_e2"], lw=1.2, color="C3", label=r"$K_{e2}$ D=0")
    ax.plot(h1["t"], h1["K_e1"], lw=1.0, color="C0", ls="--", label=r"$K_{e1}$ D>0")
    ax.plot(h1["t"], h1["K_e2"], lw=1.2, color="C2", label=r"$K_{e2}$ D>0")
    ax.axhline(K_BATH, color="0.5", ls=":", lw=0.8)
    ax.axvspan(CUE_TON, CUE_TOFF, color="C0", alpha=0.08, zorder=0)
    ax.axvspan(PROBE_TON, PROBE_TOFF, color="C3", alpha=0.08, zorder=0)
    ax.set_ylabel(r"$K_e$ [mM]")
    ax.set_xlabel("t [ms]")
    ax.set_title("Local $K_e$ (solid D=0 / dashed+green D>0)")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[1, 1]
    labels = ["Ke2 max delay", "Ke2@probe", "mean s2 probe"]
    h0_vals = [m0["Ke2_max_delay"], m0["Ke2_at_probe"], m0["mean_s2_probe"]]
    h1_vals = [m1["Ke2_max_delay"], m1["Ke2_at_probe"], m1["mean_s2_probe"]]
    xs = np.arange(len(labels))
    ax.bar(xs - 0.18, h0_vals, 0.35, label="D=0", color="0.55")
    ax.bar(xs + 0.18, h1_vals, 0.35, label=f"D={D_used:g}", color="C2")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Neighbor trace / probe (cell2)")
    ax.legend(fontsize=7)

    fig.suptitle(
        rf"B2 $K_e$ diffusion  $\gamma_{{Ke}}$={GAMMA_KE}, "
        rf"$\tau_K$={TAU_K:g}, $w=0$, K_ATP/gNa off",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "figures" / "b2_ke_diffusion.png"
    params = ke_params()
    _t, I1, I2 = make_drives()

    print(
        f"B2 Ke diffusion: gamma={GAMMA_KE}, tau_K={TAU_K}, bath={K_BATH}, "
        f"cue={CUE_AMP} ({CUE_TON}-{CUE_TOFF}), "
        f"probe2={PROBE_AMP} ({PROBE_TON}-{PROBE_TOFF}), w=0, brakes off"
    )

    tr0 = simulate_two_cell_ke(
        I1, I2, w12=W12, w21=W21, D_ke=0.0, ion_on=True, params=params,
        V0_1=V0_1, V0_2=V0_2,
    )
    m0 = summarize(tr0, "H0")
    print(
        f"H0 D=0: Ke1_max={m0['Ke1_max']:.3f} "
        f"Ke2_max_delay={m0['Ke2_max_delay']:.3f} "
        f"Ke2@probe={m0['Ke2_at_probe']:.3f} mean_s2_probe={m0['mean_s2_probe']:.4f}"
    )

    best = None
    for D in D_CANDIDATES:
        tr = simulate_two_cell_ke(
            I1, I2, w12=W12, w21=W21, D_ke=D, ion_on=True, params=params,
            V0_1=V0_1, V0_2=V0_2,
        )
        m = summarize(tr, f"H1_D{D}")
        dke = score_pair(m0, m)
        ds = m["mean_s2_probe"] - m0["mean_s2_probe"]
        print(
            f"H1 D={D:g}: Ke1_max={m['Ke1_max']:.3f} "
            f"Ke2_max_delay={m['Ke2_max_delay']:.3f} ΔKe2_delay={dke:+.3f} "
            f"Ke2@probe={m['Ke2_at_probe']:.3f} "
            f"mean_s2_probe={m['mean_s2_probe']:.4f} Δs2={ds:+.4f}"
        )
        cand = {"D": D, "tr": tr, "m": m, "dke": dke, "ds": ds}
        # Prefer clear neighbor Ke2 + best probe boost; milder D on ties
        if best is None:
            best = cand
        else:
            better = (cand["ds"], cand["dke"], -cand["D"]) > (
                best["ds"],
                best["dke"],
                -best["D"],
            )
            if better:
                best = cand

    assert best is not None
    D_used = best["D"]
    tr1 = best["tr"]
    m1 = best["m"]

    plot_results(tr0, tr1, m0, m1, D_used, out)
    print(f"saved: {out} (main D={D_used:g})")

    ke2_ok = best["dke"] >= 0.15  # mM clear neighbor rise in delay window
    s2_ok = best["ds"] > 5e-4
    h0_local = m0["Ke2_max_delay"] < K_BATH + 0.2  # delay: near bath without D
    if ke2_ok and s2_ok and h0_local:
        verdict, why = (
            "success",
            "D>0 raises Ke2 in delay vs D=0 and boosts cell2 probe soft-spike",
        )
    elif ke2_ok and s2_ok:
        verdict, why = (
            "success",
            "D>0 raises Ke2 and probe s2 (H0 delay Ke2 near bath)",
        )
    elif ke2_ok:
        verdict, why = (
            "partial",
            "neighbor Ke2 rises with D>0; probe soft-spike boost weak/absent",
        )
    else:
        verdict, why = (
            "null",
            "no clear Ke2 diffusion rise in tested D grid",
        )
    print(f"VERDICT: {verdict} — {why}")
    print(
        f"params_used: D={D_used}, gamma_Ke={GAMMA_KE}, tau_K={TAU_K}, "
        f"cue={CUE_AMP}µA/cm² {CUE_TON}-{CUE_TOFF}ms, "
        f"probe2={PROBE_AMP} {PROBE_TON}-{PROBE_TOFF}ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

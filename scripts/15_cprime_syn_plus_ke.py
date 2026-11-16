"""Phase C′ — synapse + K_e diffusion coexisting in one 2-cell toy.

Run from the repo root::

    python scripts/15_cprime_syn_plus_ke.py

Conditions (same B2-style cue→probe drive):
  neither: w=0, D=0
  syn:     w=(-15,-3), D=0
  diff:    w=0, D=0.005
  both:    w=(-15,-3), D=0.005

Claim ceiling: synaptic current and K_e diffusion can coexist in one toy —
not hybrid brain communication or optimal train/bus scheduling.
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

# B2 ion / protocol; B1 hygiene weights
GAMMA_KE = 0.0005
TAU_K = 1000.0
K_BATH = 4.0
D_KE = 0.005
W_ON = (-15.0, -3.0)
TAU_SYN = 8.0

T_END = 600.0
CUE_TON = 40.0
CUE_TOFF = 120.0
CUE_AMP = 12.0
PROBE_TON = 280.0
PROBE_TOFF = 320.0
PROBE_AMP = 5.0

V0_1 = -65.0
V0_2 = -65.0

CONDITIONS = (
    ("neither", (0.0, 0.0), 0.0),
    ("syn", W_ON, 0.0),
    ("diff", (0.0, 0.0), D_KE),
    ("both", W_ON, D_KE),
)


def ke_params() -> dict[str, float]:
    return merge_params(
        K_e_bath=K_BATH,
        K_o=K_BATH,
        gamma_Ke=GAMMA_KE,
        tau_K=TAU_K,
        g_KATP_bar=0.0,
        atp_gNa_floor=0.0,
    )


def make_drives() -> tuple[jnp.ndarray, jnp.ndarray]:
    t = jnp.arange(0.0, T_END + DT / 2, DT)
    I1 = jnp.where((t >= CUE_TON) & (t < CUE_TOFF), CUE_AMP, 0.0)
    I2 = jnp.where((t >= PROBE_TON) & (t < PROBE_TOFF), PROBE_AMP, 0.0)
    return I1, I2


def summarize(tr: dict, name: str, w12: float, w21: float, D: float) -> dict:
    t = np.asarray(tr["t"])
    s1 = np.asarray(tr["s1"])
    s2 = np.asarray(tr["s2"])
    ke1 = np.asarray(tr["K_e1"])
    ke2 = np.asarray(tr["K_e2"])
    cue = (t >= CUE_TON) & (t < CUE_TOFF)
    delay = (t >= CUE_TOFF) & (t < PROBE_TON)
    probe = (t >= PROBE_TON) & (t < PROBE_TOFF)
    return {
        "name": name,
        "w12": w12,
        "w21": w21,
        "D": D,
        "r1_cue": float(s1[cue].mean()),
        "r2_cue": float(s2[cue].mean()),
        "r1_probe": float(s1[probe].mean()),
        "r2_probe": float(s2[probe].mean()),
        "Ke1_max": float(ke1.max()),
        "Ke2_max_delay": float(ke2[delay].max()) if delay.any() else float("nan"),
        "Ke2_at_probe": float(ke2[probe][0]) if probe.any() else float("nan"),
    }


def plot_results(
    traces: dict[str, dict],
    rows: list[dict],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = {
        "neither": "0.55",
        "syn": "C0",
        "diff": "C2",
        "both": "C3",
    }

    ax = axes[0, 0]
    for name in ("neither", "syn", "diff", "both"):
        tr = traces[name]
        ax.plot(tr["t"], tr["V2"], lw=0.7, color=colors[name], label=name)
    ax.axvspan(CUE_TON, CUE_TOFF, color="C0", alpha=0.08, zorder=0)
    ax.axvspan(PROBE_TON, PROBE_TOFF, color="C3", alpha=0.08, zorder=0)
    ax.set_ylabel("V2 [mV]")
    ax.set_title("Cell2 voltage (probe target)")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    for name in ("neither", "syn", "diff", "both"):
        tr = traces[name]
        ax.plot(tr["t"], tr["K_e2"], lw=1.1, color=colors[name], label=name)
    ax.axhline(K_BATH, color="0.4", ls=":", lw=0.8)
    ax.axvspan(CUE_TON, CUE_TOFF, color="C0", alpha=0.08, zorder=0)
    ax.axvspan(PROBE_TON, PROBE_TOFF, color="C3", alpha=0.08, zorder=0)
    ax.set_ylabel(r"$K_{e2}$ [mM]")
    ax.set_title(r"Neighbor $K_e$ (bus)")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 0]
    names = [r["name"] for r in rows]
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [r["r2_probe"] for r in rows], 0.4, label="r2 probe", color="C3")
    ax.bar(xs + 0.2, [r["r1_cue"] for r in rows], 0.4, label="r1 cue", color="C0", alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean soft-spike")
    ax.set_title("Rates: cue (cell1) / probe (cell2)")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.bar(xs - 0.2, [r["Ke2_at_probe"] for r in rows], 0.4, label="Ke2@probe", color="C2")
    ax.bar(xs + 0.2, [r["Ke1_max"] for r in rows], 0.4, label="Ke1_max", color="C1", alpha=0.75)
    ax.axhline(K_BATH, color="0.4", ls=":", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel(r"$K_e$ [mM]")
    ax.set_title(r"$K_e$ summary")
    ax.legend(fontsize=7)

    fig.suptitle(
        rf"C′ syn+Ke  $w={W_ON}$, $D={D_KE}$, "
        rf"cue→probe (B2-style), K_ATP/gNa off",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "figures" / "cprime_syn_plus_ke.png"
    params = ke_params()
    I1, I2 = make_drives()

    print(
        f"C′ syn+Ke: W_ON={W_ON}, D={D_KE}, tau_syn={TAU_SYN}, "
        f"gamma={GAMMA_KE}, tau_K={TAU_K}"
    )
    print(
        f"protocol: cell1 cue {CUE_AMP} ({CUE_TON}-{CUE_TOFF}), "
        f"cell2 probe {PROBE_AMP} ({PROBE_TON}-{PROBE_TOFF}); "
        f"K_ATP/gNa off — B2-style drive"
    )

    traces: dict[str, dict] = {}
    rows: list[dict] = []
    for name, (w12, w21), D in CONDITIONS:
        tr = simulate_two_cell_ke(
            I1,
            I2,
            w12=w12,
            w21=w21,
            D_ke=D,
            ion_on=True,
            tau_syn=TAU_SYN,
            params=params,
            V0_1=V0_1,
            V0_2=V0_2,
        )
        m = summarize(tr, name, w12, w21, D)
        traces[name] = tr
        rows.append(m)
        print(
            f"{name:7s}: r1_cue={m['r1_cue']:.4f} r2_cue={m['r2_cue']:.4f} "
            f"r1_probe={m['r1_probe']:.4f} r2_probe={m['r2_probe']:.4f} "
            f"Ke1_max={m['Ke1_max']:.3f} Ke2@probe={m['Ke2_at_probe']:.3f}"
        )

    plot_results(traces, rows, out)
    print(f"saved: {out}")

    by = {r["name"]: r for r in rows}
    d_ke = by["diff"]["Ke2_at_probe"] - by["neither"]["Ke2_at_probe"]
    b_ke = by["both"]["Ke2_at_probe"] - by["neither"]["Ke2_at_probe"]
    s_ke = by["syn"]["Ke2_at_probe"] - by["neither"]["Ke2_at_probe"]
    d_s2 = by["diff"]["r2_probe"] - by["neither"]["r2_probe"]
    b_s2 = by["both"]["r2_probe"] - by["neither"]["r2_probe"]
    s_s2 = by["syn"]["r2_probe"] - by["neither"]["r2_probe"]
    # syn-only vs neither on cue (train window) and both≈diff on bus metrics
    s_r2c = by["syn"]["r2_cue"] - by["neither"]["r2_cue"]
    both_vs_diff_ke = abs(by["both"]["Ke2_at_probe"] - by["diff"]["Ke2_at_probe"])
    both_vs_diff_s2 = abs(by["both"]["r2_probe"] - by["diff"]["r2_probe"])

    print("--- contrasts vs neither ---")
    print(f"  diff:  ΔKe2@probe={d_ke:+.3f} Δr2_probe={d_s2:+.4f}")
    print(f"  syn:   ΔKe2@probe={s_ke:+.3f} Δr2_probe={s_s2:+.4f} Δr2_cue={s_r2c:+.4f}")
    print(
        f"  both:  ΔKe2@probe={b_ke:+.3f} Δr2_probe={b_s2:+.4f} "
        f"|both−diff| Ke={both_vs_diff_ke:.3f} s2={both_vs_diff_s2:.4f}"
    )

    bus_in_both = b_ke >= 0.5 * max(d_ke, 1e-6) and b_ke > 0.15 and b_s2 > 5e-4
    train_alone = abs(s_s2) > 5e-3 or abs(s_r2c) > 5e-3
    # under cue→probe, syn barely moves bus metrics; both ≈ diff
    both_like_diff = both_vs_diff_ke < 0.2 and both_vs_diff_s2 < 0.005
    train_in_both = (not both_like_diff) and train_alone

    if bus_in_both and train_in_both:
        verdict = "coexist"
        why = "both shows delayed Ke2/probe boost and a clear synaptic rate shift"
    elif bus_in_both and both_like_diff:
        verdict = "diff-dominated"
        why = (
            "both ≈ diff on Ke2/probe (bus lives with w≠0); "
            "syn-only barely moves this cue→probe readout (train weak here)"
        )
    elif train_alone and not bus_in_both:
        verdict = "syn-dominated"
        why = "synaptic rate effects present; Ke neighbor rise weak in both"
    else:
        verdict = "null"
        why = "no clear dual-channel signature in this protocol"
    print(f"VERDICT: {verdict} — {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

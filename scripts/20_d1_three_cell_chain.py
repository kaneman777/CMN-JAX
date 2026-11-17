"""Phase D1 — 3-cell linear chain, diffusion off, role split (minimal).

Run from the repo root::

    python scripts/20_d1_three_cell_chain.py

Topology: **1—2—3 only** (no 1–3, no Y / triangle / ring).
K_e D=0, K_ATP/gNa/learning/compensation/rhythm off.

Success: coupling on → rate differentiation; off → nearly uniform.
If weights are fully symmetric, any end–middle split is **topology
(degree) structure**, not spontaneous symmetry breaking.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jnpnp.models.point_sim import DT, I_ext_step, merge_params
from jnpnp.models.three_cell import simulate_three_cell_chain

T_END = 500.0
T_ON = 20.0
T_OFF = 480.0
I_DRIVE = 9.0
TAU_SYN = 8.0

# Identical V0 / a0: no seed asymmetry (document if changed)
V0_1 = V0_2 = V0_3 = -65.0
A0 = 1.0

# Fully symmetric mutual inhibition on both edges (2-cell weak scale)
# Middle has degree 2 → more inhibition than ends (topology, not SSB)
W_EDGE = -8.0
W_ON = (W_EDGE, W_EDGE, W_EDGE, W_EDGE)  # w12, w21, w23, w32
W_OFF = (0.0, 0.0, 0.0, 0.0)

# If Q1 ratio stays weak, try slightly stronger |W| (still avoid deep hyperpol)
W_FALLBACK = -12.0


def make_drive() -> jnp.ndarray:
    t = jnp.arange(0.0, T_END + DT / 2, DT)
    return I_ext_step(t, T_ON, T_OFF, I_DRIVE)


def metrics(tr: dict) -> dict[str, float]:
    t = np.asarray(tr["t"])
    mask = (t >= T_ON) & (t < T_OFF)
    r1 = float(np.asarray(tr["s1"])[mask].mean())
    r2 = float(np.asarray(tr["s2"])[mask].mean())
    r3 = float(np.asarray(tr["s3"])[mask].mean())
    rates = np.array([r1, r2, r3])
    ratio = float(rates.max() / max(rates.min(), 1e-9))
    end_mean = 0.5 * (r1 + r3)
    return {
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "ratio": ratio,
        "end_mean": end_mean,
        "mid_vs_end": float(r2 - end_mean),
        "da1": float(tr["a1"][0] - tr["a1"][-1]),
        "da2": float(tr["a2"][0] - tr["a2"][-1]),
        "da3": float(tr["a3"][0] - tr["a3"][-1]),
        "Vmin": float(
            min(
                np.asarray(tr["V1"]).min(),
                np.asarray(tr["V2"]).min(),
                np.asarray(tr["V3"]).min(),
            )
        ),
    }


def run_case(
    I: jnp.ndarray, weights: tuple[float, float, float, float]
) -> tuple[dict, dict]:
    w12, w21, w23, w32 = weights
    tr = simulate_three_cell_chain(
        I,
        w12=w12,
        w21=w21,
        w23=w23,
        w32=w32,
        V0_1=V0_1,
        V0_2=V0_2,
        V0_3=V0_3,
        a0_1=A0,
        a0_2=A0,
        a0_3=A0,
        tau_syn=TAU_SYN,
        params=merge_params(),  # K_ATP/gNa/Ke defaults off
    )
    return tr, metrics(tr)


def plot_results(
    *,
    tr_off: dict,
    tr_on: dict,
    m_off: dict,
    m_on: dict,
    w_used: tuple[float, float, float, float],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    ax = axes[0, 0]
    for k, lab in (("V1", "V1"), ("V2", "V2"), ("V3", "V3")):
        ax.plot(tr_on["t"], tr_on[k], lw=0.65, label=lab)
    ax.axvspan(T_ON, T_OFF, color="0.9", zorder=0)
    ax.set_ylabel("V [mV]")
    ax.set_title("Q1 on: voltages")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for k, lab in (("a1", "a1"), ("a2", "a2"), ("a3", "a3")):
        ax.plot(tr_on["t"], tr_on[k], lw=1.0, label=lab)
    ax.axvspan(T_ON, T_OFF, color="0.9", zorder=0)
    ax.set_ylabel("ATP a")
    ax.set_title("Q1 on: ATP")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    xs = np.arange(3)
    ax.bar(xs - 0.2, [m_off["r1"], m_off["r2"], m_off["r3"]], 0.4, label="Q0 off")
    ax.bar(xs + 0.2, [m_on["r1"], m_on["r2"], m_on["r3"]], 0.4, label="Q1 on")
    ax.set_xticks(xs)
    ax.set_xticklabels(["cell1", "cell2", "cell3"])
    ax.set_ylabel("mean soft-spike")
    ax.set_title(
        f"Rates  off ratio={m_off['ratio']:.2f}  on ratio={m_on['ratio']:.2f}"
    )
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.axis("off")
    txt = (
        f"topology: 1—2—3 only (no 1–3)\n"
        f"w12,w21,w23,w32 = {w_used}\n"
        f"V0 identical={V0_1}; seed asymmetry=no\n"
        f"D=0; K_ATP/gNa off\n\n"
        f"Q0: r=({m_off['r1']:.4f},{m_off['r2']:.4f},{m_off['r3']:.4f})\n"
        f"Q1: r=({m_on['r1']:.4f},{m_on['r2']:.4f},{m_on['r3']:.4f})\n"
        f"mid−end_mean={m_on['mid_vs_end']:+.4f}\n"
        f"Δa=({m_on['da1']:.3f},{m_on['da2']:.3f},{m_on['da3']:.3f})\n"
        f"Vmin={m_on['Vmin']:.1f}"
    )
    ax.text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=9)

    fig.suptitle(
        rf"D1 three-cell chain  $w_{{\mathrm{{edge}}}}$={w_used[0]}  "
        r"(diffusion off; not Y/△)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "figures" / "d1_three_cell_chain.png"
    I = make_drive()

    print(
        "D1 three-cell chain 1—2—3: D=0, K_ATP/gNa off, "
        f"I={I_DRIVE}, τ={TAU_SYN}, V0 identical, no 1–3"
    )
    print("next topologies (NOT run): Y / triangle / ring = unimplemented")

    tr_off, m_off = run_case(I, W_OFF)
    print(
        f"Q0 off: r=({m_off['r1']:.4f},{m_off['r2']:.4f},{m_off['r3']:.4f}) "
        f"ratio={m_off['ratio']:.3f}"
    )

    w_used = W_ON
    tr_on, m_on = run_case(I, w_used)
    print(
        f"Q1 on |w|={abs(W_EDGE)}: "
        f"r=({m_on['r1']:.4f},{m_on['r2']:.4f},{m_on['r3']:.4f}) "
        f"ratio={m_on['ratio']:.3f} mid−end={m_on['mid_vs_end']:+.4f} "
        f"Vmin={m_on['Vmin']:.1f} "
        f"Δa=({m_on['da1']:.3f},{m_on['da2']:.3f},{m_on['da3']:.3f})"
    )

    # Gentle fallback if differentiation weak but V still healthy
    if m_on["ratio"] < 1.3 and m_on["Vmin"] > -120.0:
        w_fb = (W_FALLBACK,) * 4
        tr_fb, m_fb = run_case(I, w_fb)
        print(
            f"Q1 fallback |w|={abs(W_FALLBACK)}: "
            f"r=({m_fb['r1']:.4f},{m_fb['r2']:.4f},{m_fb['r3']:.4f}) "
            f"ratio={m_fb['ratio']:.3f} Vmin={m_fb['Vmin']:.1f}"
        )
        if m_fb["ratio"] > m_on["ratio"] and m_fb["Vmin"] > -150.0:
            tr_on, m_on, w_used = tr_fb, m_fb, w_fb
            print("  → using fallback weights for figure / verdict")

    plot_results(
        tr_off=tr_off,
        tr_on=tr_on,
        m_off=m_off,
        m_on=m_on,
        w_used=w_used,
        out_path=out,
    )
    print(f"saved: {out}")

    off_uniform = m_off["ratio"] < 1.08
    on_split = m_on["ratio"] >= 1.3
    ends_vs_mid = abs(m_on["mid_vs_end"]) > 5e-4
    healthy = m_on["Vmin"] > -150.0

    if off_uniform and on_split and healthy:
        verdict = "success"
        why = (
            "off≈uniform; on ratio≥1.3 with healthy V "
            "(topology degree / chain structure — not SSB)"
        )
    elif off_uniform and on_split:
        verdict = "partial"
        why = "rate split on but Vmin deep — hygiene concern"
    elif off_uniform and ends_vs_mid and m_on["ratio"] > m_off["ratio"] * 1.05:
        verdict = "partial"
        why = "end–middle contrast visible but ratio below 1.3 bar"
    else:
        verdict = "null"
        why = "no clear coupling-on role split under tested |w|"
    print(f"VERDICT: {verdict} — {why}")
    print(
        "claim ceiling: linear 3-cell diffusion-off role preview — "
        "not brain topology / Y / triangle / rhythms"
    )
    print(
        f"seed/structure: V0 identical; w fully symmetric {w_used}; "
        "any split = chain degree asymmetry"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase D2 — 3-cell complete triangle, mutual inhibition, g=8.

Run from the repo root::

    python scripts/22_d2_three_cell_triangle.py

Topology: complete graph on {1,2,3} (edges 12, 23, **and** 31).
Fully symmetric inhibition ``w_ij = w_ji = -8``. Same drive / V0 / D=0 /
K_ATP-off as D1 chain (``scripts/20``, ``21``). Chain and Y are not run here.

Question: does chain LHL survive, or does the triangle flatten / change pattern?
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
V0 = -65.0
A0 = 1.0
G = 8.0

# D1 chain reference at g=8 (STATUS / script 20; not re-swept)
CHAIN_REF = {
    "r1": 0.0182,
    "r2": 0.0287,
    "r3": 0.0182,
    "ratio": 1.580,
    "pattern": "LHL",
    "Vmin": -94.5,
}


def make_drive() -> jnp.ndarray:
    t = jnp.arange(0.0, T_END + DT / 2, DT)
    return I_ext_step(t, T_ON, T_OFF, I_DRIVE)


def metrics(tr: dict) -> dict[str, float | str]:
    t = np.asarray(tr["t"])
    mask = (t >= T_ON) & (t < T_OFF)
    r1 = float(np.asarray(tr["s1"])[mask].mean())
    r2 = float(np.asarray(tr["s2"])[mask].mean())
    r3 = float(np.asarray(tr["s3"])[mask].mean())
    rates = np.array([r1, r2, r3])
    ratio = float(rates.max() / max(rates.min(), 1e-9))
    vmin = float(
        min(
            np.asarray(tr["V1"]).min(),
            np.asarray(tr["V2"]).min(),
            np.asarray(tr["V3"]).min(),
        )
    )
    # Rank / dispersion label (avoid LHL which needs edge orientation)
    spread = float(rates.max() - rates.min())
    if ratio < 1.08:
        pattern = "uniform"
    elif spread > 0.002 and (rates == rates.max()).sum() == 1:
        pattern = "one_cell_high"
    else:
        pattern = "other"
    return {
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "ratio": ratio,
        "spread": spread,
        "pattern": pattern,
        "Vmin": vmin,
        "da1": float(tr["a1"][0] - tr["a1"][-1]),
        "da2": float(tr["a2"][0] - tr["a2"][-1]),
        "da3": float(tr["a3"][0] - tr["a3"][-1]),
    }


def run_triangle(I: jnp.ndarray, g: float) -> tuple[dict, dict]:
    w = -float(g)
    tr = simulate_three_cell_chain(
        I,
        w12=w,
        w21=w,
        w23=w,
        w32=w,
        w13=w,
        w31=w,
        V0_1=V0,
        V0_2=V0,
        V0_3=V0,
        a0_1=A0,
        a0_2=A0,
        a0_3=A0,
        tau_syn=TAU_SYN,
        params=merge_params(),
    )
    return tr, metrics(tr)


def run_off(I: jnp.ndarray) -> tuple[dict, dict]:
    tr = simulate_three_cell_chain(
        I,
        w12=0.0,
        w21=0.0,
        w23=0.0,
        w32=0.0,
        w13=0.0,
        w31=0.0,
        V0_1=V0,
        V0_2=V0,
        V0_3=V0,
        a0_1=A0,
        a0_2=A0,
        a0_3=A0,
        tau_syn=TAU_SYN,
        params=merge_params(),
    )
    return tr, metrics(tr)


def plot_results(
    *,
    tr_off: dict,
    tr_on: dict,
    m_off: dict,
    m_on: dict,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    ax = axes[0, 0]
    for k, lab in (("V1", "V1"), ("V2", "V2"), ("V3", "V3")):
        ax.plot(tr_on["t"], tr_on[k], lw=0.65, label=lab)
    ax.axvspan(T_ON, T_OFF, color="0.9", zorder=0)
    ax.set_ylabel("V [mV]")
    ax.set_title("Triangle on: voltages")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for k, lab in (("a1", "a1"), ("a2", "a2"), ("a3", "a3")):
        ax.plot(tr_on["t"], tr_on[k], lw=1.0, label=lab)
    ax.axvspan(T_ON, T_OFF, color="0.9", zorder=0)
    ax.set_ylabel("ATP a")
    ax.set_title("Triangle on: ATP")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    xs = np.arange(3)
    ax.bar(
        xs - 0.2,
        [m_off["r1"], m_off["r2"], m_off["r3"]],
        0.4,
        label="off",
    )
    ax.bar(
        xs + 0.2,
        [m_on["r1"], m_on["r2"], m_on["r3"]],
        0.4,
        label="triangle",
    )
    ax.plot(
        xs,
        [CHAIN_REF["r1"], CHAIN_REF["r2"], CHAIN_REF["r3"]],
        "ks--",
        lw=1.0,
        markersize=5,
        label="chain D1 g=8",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(["cell1", "cell2", "cell3"])
    ax.set_ylabel("mean soft-spike")
    ax.set_title(
        f"Rates  tri ratio={m_on['ratio']:.2f}  chain ratio={CHAIN_REF['ratio']:.2f}"
    )
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.axis("off")
    txt = (
        f"topology: complete K3 (12,23,31)\n"
        f"w_ij=w_ji=-{G:.0f}  (fully symmetric)\n"
        f"V0 identical={V0}; seed=no; D=0\n\n"
        f"off:  r=({m_off['r1']:.4f},{m_off['r2']:.4f},{m_off['r3']:.4f})\n"
        f"      ratio={m_off['ratio']:.3f}  {m_off['pattern']}\n"
        f"tri:  r=({m_on['r1']:.4f},{m_on['r2']:.4f},{m_on['r3']:.4f})\n"
        f"      ratio={m_on['ratio']:.3f}  {m_on['pattern']}\n"
        f"      Vmin={m_on['Vmin']:.1f}\n"
        f"chain D1 g=8: r=({CHAIN_REF['r1']:.4f},"
        f"{CHAIN_REF['r2']:.4f},{CHAIN_REF['r3']:.4f}) LHL\n"
    )
    ax.text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=9)

    fig.suptitle(
        rf"D2 three-cell triangle  $|w|$={G:.0f}  (vs chain LHL; no Y)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "figures" / "d2_three_cell_triangle.png"
    I = make_drive()

    print(
        f"D2 complete triangle K3: |w|={G}, I={I_DRIVE}, V0={V0}, "
        "D=0, K_ATP/gNa off; chain/Y not run"
    )

    tr_off, m_off = run_off(I)
    print(
        f"off: r=({m_off['r1']:.4f},{m_off['r2']:.4f},{m_off['r3']:.4f}) "
        f"ratio={m_off['ratio']:.3f} pattern={m_off['pattern']}"
    )

    tr_on, m_on = run_triangle(I, G)
    print(
        f"triangle g={G:.0f}: "
        f"r=({m_on['r1']:.4f},{m_on['r2']:.4f},{m_on['r3']:.4f}) "
        f"ratio={m_on['ratio']:.3f} pattern={m_on['pattern']} "
        f"Vmin={m_on['Vmin']:.1f} "
        f"Δa=({m_on['da1']:.3f},{m_on['da2']:.3f},{m_on['da3']:.3f})"
    )
    print(
        f"chain D1 g=8 ref: "
        f"r=({CHAIN_REF['r1']:.4f},{CHAIN_REF['r2']:.4f},{CHAIN_REF['r3']:.4f}) "
        f"ratio={CHAIN_REF['ratio']:.3f} LHL Vmin={CHAIN_REF['Vmin']:.1f}"
    )

    plot_results(
        tr_off=tr_off, tr_on=tr_on, m_off=m_off, m_on=m_on, out_path=out
    )
    print(f"saved: {out}")

    # Compare to chain
    flatter = m_on["ratio"] < CHAIN_REF["ratio"] - 0.15
    near_uniform = m_on["ratio"] < 1.08
    if near_uniform:
        cmp = "triangle flattens vs chain LHL (near-uniform)"
    elif flatter:
        cmp = "triangle weaker split than chain LHL"
    else:
        cmp = "triangle keeps comparable rate contrast (check pattern)"
    print(f"COMPARE: {cmp}")
    print("claim ceiling: topology contrast chain vs K3 only — not brain circuits / Y")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

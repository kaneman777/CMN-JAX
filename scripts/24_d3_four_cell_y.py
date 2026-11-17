"""Phase D3 — 4-cell Y / star (hub + 3 leaves), |w|=8.

Run from the repo root::

    python scripts/24_d3_four_cell_y.py

Star edges only (hub–leaf). No leaf–leaf. Not a 3-cell “Y” (that is a chain).
Same drive / V0 / D=0 / K_ATP-off as D1/D2. Asks whether degree asymmetry
(hub deg=3, leaf deg=1) alone yields a rate contrast vs triangle flattening.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jnpnp.models.four_cell import simulate_four_cell_star
from jnpnp.models.point_sim import DT, I_ext_step, merge_params

T_END = 500.0
T_ON = 20.0
T_OFF = 480.0
I_DRIVE = 9.0
TAU_SYN = 8.0
V0 = -65.0
A0 = 1.0
G = 8.0

# Reference rows (no recompute)
REF_CHAIN = "D1 chain g=8: LHL r≈(0.018,0.029,0.018) ratio=1.58"
REF_TRI = "D2 triangle g=8: uniform r≈0.076 ratio=1.00"


def make_drive() -> jnp.ndarray:
    t = jnp.arange(0.0, T_END + DT / 2, DT)
    return I_ext_step(t, T_ON, T_OFF, I_DRIVE)


def metrics(tr: dict) -> dict[str, float | str]:
    t = np.asarray(tr["t"])
    mask = (t >= T_ON) & (t < T_OFF)
    rH = float(np.asarray(tr["sH"])[mask].mean())
    rL1 = float(np.asarray(tr["sL1"])[mask].mean())
    rL2 = float(np.asarray(tr["sL2"])[mask].mean())
    rL3 = float(np.asarray(tr["sL3"])[mask].mean())
    leaf_mean = float(np.mean([rL1, rL2, rL3]))
    rates = np.array([rH, rL1, rL2, rL3])
    ratio_mm = float(rates.max() / max(rates.min(), 1e-9))
    ratio_hl = float(rH / max(leaf_mean, 1e-9))
    vmin = float(
        min(
            np.asarray(tr["VH"]).min(),
            np.asarray(tr["VL1"]).min(),
            np.asarray(tr["VL2"]).min(),
            np.asarray(tr["VL3"]).min(),
        )
    )
    # pattern (soft labels)
    if ratio_mm < 1.08:
        pattern = "flat"
    elif rH > leaf_mean * 1.08:
        pattern = "hub_high"
    elif rH < leaf_mean / 1.08:
        pattern = "hub_low"
    else:
        pattern = "other"
    return {
        "rH": rH,
        "rL1": rL1,
        "rL2": rL2,
        "rL3": rL3,
        "leaf_mean": leaf_mean,
        "ratio_maxmin": ratio_mm,
        "ratio_hub_leaf": ratio_hl,
        "pattern": pattern,
        "Vmin": vmin,
    }


def run(I: jnp.ndarray, g: float) -> tuple[dict, dict]:
    w = -float(g)
    tr = simulate_four_cell_star(
        I,
        w_hub_from_leaf=w,
        w_leaf_from_hub=w,
        V0_h=V0,
        V0_l1=V0,
        V0_l2=V0,
        V0_l3=V0,
        a0_h=A0,
        a0_l1=A0,
        a0_l2=A0,
        a0_l3=A0,
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
    t = np.asarray(tr_on["t"])
    mask = (t >= 80.0) & (t < 180.0)
    ax.plot(t[mask], np.asarray(tr_on["VH"])[mask], lw=0.8, label="H", color="C3")
    for k, c in (("VL1", "C0"), ("VL2", "C2"), ("VL3", "C1")):
        ax.plot(t[mask], np.asarray(tr_on[k])[mask], lw=0.65, label=k[1:], color=c)
    ax.set_ylabel("V [mV]")
    ax.set_title("Y on: V (hub vs leaves)")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[0, 1]
    ax.plot(tr_on["t"], tr_on["aH"], lw=1.0, label="aH", color="C3")
    ax.plot(tr_on["t"], tr_on["aL1"], lw=0.8, label="aL1", color="C0")
    ax.axvspan(T_ON, T_OFF, color="0.9", zorder=0)
    ax.set_ylabel("ATP a")
    ax.set_title("ATP (hub + one leaf)")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    labels = ["H", "L1", "L2", "L3"]
    xs = np.arange(4)
    off_r = [m_off["rH"], m_off["rL1"], m_off["rL2"], m_off["rL3"]]
    on_r = [m_on["rH"], m_on["rL1"], m_on["rL2"], m_on["rL3"]]
    ax.bar(xs - 0.2, off_r, 0.4, label="off")
    ax.bar(xs + 0.2, on_r, 0.4, label="Y on")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean soft-spike")
    ax.set_title(
        f"Rates  on pattern={m_on['pattern']}  "
        f"rH/leaf={m_on['ratio_hub_leaf']:.2f}"
    )
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.axis("off")
    txt = (
        f"topology: star Y (hub–leaf only)\n"
        f"|w|={G:.0f} both ways; no leaf–leaf\n"
        f"V0 identical; seed=no; D=0\n\n"
        f"off: H={m_off['rH']:.4f} L=({m_off['rL1']:.4f},"
        f"{m_off['rL2']:.4f},{m_off['rL3']:.4f})\n"
        f"Y:   H={m_on['rH']:.4f} L=({m_on['rL1']:.4f},"
        f"{m_on['rL2']:.4f},{m_on['rL3']:.4f})\n"
        f"     leaf_mean={m_on['leaf_mean']:.4f}\n"
        f"     max/min={m_on['ratio_maxmin']:.3f}  "
        f"H/leaf={m_on['ratio_hub_leaf']:.3f}\n"
        f"     Vmin={m_on['Vmin']:.1f}  pattern={m_on['pattern']}\n\n"
        f"{REF_CHAIN}\n{REF_TRI}\n"
    )
    ax.text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=8)

    fig.suptitle(
        rf"D3 four-cell star Y  $|w|$={G:.0f}  (degree asymmetry; no WTA claim)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "figures" / "d3_four_cell_y.png"
    I = make_drive()

    print(
        f"D3 four-cell star Y: |w|={G}, hub–leaf only, I={I_DRIVE}, "
        f"V0={V0}, D=0, K_ATP/gNa off"
    )
    print("not a 3-cell Y (=chain); leaf–leaf edges absent")

    tr_off, m_off = run(I, 0.0)
    print(
        f"off: H={m_off['rH']:.4f} L=({m_off['rL1']:.4f},{m_off['rL2']:.4f},"
        f"{m_off['rL3']:.4f}) ratio={m_off['ratio_maxmin']:.3f} "
        f"pattern={m_off['pattern']}"
    )

    tr_on, m_on = run(I, G)
    print(
        f"Y on: H={m_on['rH']:.4f} L=({m_on['rL1']:.4f},{m_on['rL2']:.4f},"
        f"{m_on['rL3']:.4f}) leaf_mean={m_on['leaf_mean']:.4f} "
        f"max/min={m_on['ratio_maxmin']:.3f} H/leaf={m_on['ratio_hub_leaf']:.3f} "
        f"Vmin={m_on['Vmin']:.1f} pattern={m_on['pattern']}"
    )
    print(f"ref: {REF_CHAIN}")
    print(f"ref: {REF_TRI}")

    plot_results(
        tr_off=tr_off, tr_on=tr_on, m_off=m_off, m_on=m_on, out_path=out
    )
    print(f"saved: {out}")

    if m_off["pattern"] == "flat" and m_on["pattern"] in ("hub_high", "hub_low"):
        note = "degree asymmetry alone yields hub vs leaf rate contrast"
    elif m_off["pattern"] == "flat" and m_on["pattern"] == "flat":
        note = "star stays flat under tested |w| (unlike chain LHL)"
    else:
        note = f"on pattern={m_on['pattern']} (see numbers; no WTA claim)"
    print(f"DEGREE: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

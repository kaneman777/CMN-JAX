"""Phase D1 hygiene — |w| sweep on the 3-cell linear chain (no triangle).

Run from the repo root::

    python scripts/21_d1_chain_w_sweep.py

Same protocol as ``20_d1_three_cell_chain.py``: topology 1—2—3 only, D=0,
K_ATP/gNa off, identical V0, fully symmetric mutual inhibition
``w12=w21=w23=w32=-g``. Finds the band where LHL (middle-high) role split
survives without deep hyperpolarization.

Y / triangle / ring are **not** implemented here.
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

# |w| = g sweep (fully symmetric)
G_SWEEP = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0)
VMIN_WARN = -120.0
VMIN_BAD = -150.0
RATIO_SPLIT = 1.3
UNIFORM_RATIO = 1.08


def make_drive() -> jnp.ndarray:
    t = jnp.arange(0.0, T_END + DT / 2, DT)
    return I_ext_step(t, T_ON, T_OFF, I_DRIVE)


def label_pattern(r1: float, r2: float, r3: float, ratio: float) -> str:
    if ratio < UNIFORM_RATIO:
        return "flat"
    ends = 0.5 * (r1 + r3)
    # LHL: low-high-low (middle high)
    if r2 > r1 and r2 > r3 and abs(r1 - r3) < 0.25 * max(r2, 1e-9):
        return "LHL"
    # HLH: high-low-high (middle low)
    if r2 < r1 and r2 < r3 and abs(r1 - r3) < 0.25 * max(ends, 1e-9):
        return "HLH"
    return "other"


def run_g(I: jnp.ndarray, g: float) -> dict[str, float | str]:
    w = -float(g)
    tr = simulate_three_cell_chain(
        I,
        w12=w,
        w21=w,
        w23=w,
        w32=w,
        V0_1=V0,
        V0_2=V0,
        V0_3=V0,
        a0_1=A0,
        a0_2=A0,
        a0_3=A0,
        tau_syn=TAU_SYN,
        params=merge_params(),
    )
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
    return {
        "g": float(g),
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "ratio": ratio,
        "pattern": label_pattern(r1, r2, r3, ratio),
        "Vmin": vmin,
    }


def pick_recommended(rows: list[dict]) -> dict | None:
    """Prefer LHL with ratio≥1.3 and Vmin > warn; nearest to D1 g=8."""
    good = [
        r
        for r in rows
        if r["pattern"] == "LHL"
        and r["ratio"] >= RATIO_SPLIT
        and r["Vmin"] > VMIN_WARN
        and r["g"] > 0
    ]
    if not good:
        good = [
            r
            for r in rows
            if r["pattern"] == "LHL"
            and r["Vmin"] > VMIN_BAD
            and r["g"] > 0
        ]
    if not good:
        return None
    return min(good, key=lambda r: (abs(r["g"] - 8.0), -r["ratio"]))


def plot_sweep(rows: list[dict], rec: dict | None, out_path: Path) -> None:
    gs = [r["g"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(gs, [r["r1"] for r in rows], "o-", label="r1", color="C0")
    ax.plot(gs, [r["r2"] for r in rows], "s-", label="r2 (mid)", color="C3")
    ax.plot(gs, [r["r3"] for r in rows], "^-", label="r3", color="C2")
    ax.axvline(8.0, color="0.5", ls=":", lw=0.9, label="D1 g=8")
    if rec is not None:
        ax.axvline(rec["g"], color="C1", ls="--", lw=0.9, label=f"rec g={rec['g']:.0f}")
    ax.set_xlabel("|w| = g")
    ax.set_ylabel("mean soft-spike")
    ax.set_title("Rates vs |w|")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(gs, [r["ratio"] for r in rows], "o-", color="C0", label="ratio")
    ax.axhline(RATIO_SPLIT, color="0.4", ls=":", lw=0.9, label="ratio=1.3")
    ax.axvline(8.0, color="0.5", ls=":", lw=0.9)
    for r in rows:
        ax.annotate(
            str(r["pattern"]),
            (r["g"], r["ratio"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=6,
        )
    ax2 = ax.twinx()
    ax2.plot(gs, [r["Vmin"] for r in rows], "x--", color="C3", label="Vmin")
    ax2.axhline(VMIN_WARN, color="C3", ls=":", lw=0.7)
    ax2.set_ylabel("Vmin [mV]", color="C3")
    ax.set_xlabel("|w| = g")
    ax.set_ylabel("max/min ratio")
    ax.set_title("ratio & Vmin (LHL band)")
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        r"D1 chain $|w|$ sweep  (1—2—3 only; no △/Y; D=0)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "figures" / "d1_chain_w_sweep.png"
    I = make_drive()

    print(
        "D1 |w| sweep on 1—2—3 (no triangle/Y): "
        f"g={G_SWEEP}, I={I_DRIVE}, V0={V0}, D=0, K_ATP/gNa off"
    )
    rows: list[dict] = []
    for g in G_SWEEP:
        row = run_g(I, float(g))
        rows.append(row)
        print(
            f"g={row['g']:5.1f}: r=({row['r1']:.4f},{row['r2']:.4f},{row['r3']:.4f}) "
            f"ratio={row['ratio']:.3f} pattern={row['pattern']} "
            f"Vmin={row['Vmin']:.1f}"
        )

    rec = pick_recommended(rows)
    # bands
    lhl = [r for r in rows if r["pattern"] == "LHL" and r["g"] > 0]
    fade = [
        r
        for r in rows
        if r["g"] > 0 and (r["pattern"] == "flat" or r["ratio"] < RATIO_SPLIT)
    ]
    deep = [r for r in rows if r["Vmin"] <= VMIN_WARN]

    plot_sweep(rows, rec, out)
    print(f"saved: {out}")

    if rec is not None:
        print(
            f"RECOMMENDED g={rec['g']:.0f} "
            f"(pattern={rec['pattern']}, ratio={rec['ratio']:.3f}, "
            f"Vmin={rec['Vmin']:.1f}) — for later triangle *comparison* only"
        )
    else:
        print("RECOMMENDED g: none clean (see table)")

    if lhl:
        gs = [r["g"] for r in lhl]
        print(f"LHL band (survives): g in {gs}")
    if fade:
        print(f"weak/fade band: g={[r['g'] for r in fade]}")
    if deep:
        print(f"deep-V warn (Vmin≤{VMIN_WARN}): g={[r['g'] for r in deep]}")
    print("triangle / Y: NOT implemented")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase B1 — 2-cell minimal net: weak synaptic coupling → firing asymmetry.

Run from the repo root::

    python scripts/10_b1_two_cell_asymm.py

E0: w12=w21=0 (symmetric rates). E1: weak mutual inhibition with mild
structural asymmetry (w12 ≠ w21) + tiny V0 seed. Learning off.
Success criterion ① only: clear rate asymmetry (ratio ≳ 1.5) vs off.

Claim ceiling: 2-cell synaptic coupling can seed firing asymmetry / slack —
not network intelligence, rehab, or brain rhythms.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jnpnp.models.point_sim import DT, I_ext_step
from jnpnp.models.two_cell import simulate_two_cell

# ---------------------------------------------------------------------------
# Protocol (learning off; K_e / K_ATP / gNa gate default off)
# ---------------------------------------------------------------------------
T_END = 500.0
T_ON = 20.0
T_OFF = 480.0
I_DRIVE = 9.0  # µA/cm² shared
TAU_SYN = 8.0  # ms synaptic filter

# Tiny initial V asymmetry (seed); a0 identical
V0_1 = -65.0
V0_2 = -64.2
A0 = 1.0

# E0
W_OFF = (0.0, 0.0)
# E1: mutual inhibition, structurally asymmetric (document as such)
W_ON = (-80.0, -10.0)  # w12, w21


def make_drive() -> tuple[jnp.ndarray, jnp.ndarray]:
    t = jnp.arange(0.0, T_END + DT / 2, DT)
    I = I_ext_step(t, T_ON, T_OFF, I_DRIVE)
    return t, I


def rate_metrics(tr: dict) -> dict[str, float]:
    t = np.asarray(tr["t"])
    s1 = np.asarray(tr["s1"])
    s2 = np.asarray(tr["s2"])
    mask = (t >= T_ON) & (t < T_OFF)
    r1 = float(s1[mask].mean())
    r2 = float(s2[mask].mean())
    da1 = float(tr["a1"][0] - tr["a1"][-1])
    da2 = float(tr["a2"][0] - tr["a2"][-1])
    ratio = max(r1, r2) / max(min(r1, r2), 1e-9)
    return {"r1": r1, "r2": r2, "ratio": ratio, "da1": da1, "da2": da2}


def run_case(I_drive: jnp.ndarray, w12: float, w21: float) -> tuple[dict, dict]:
    tr = simulate_two_cell(
        I_drive,
        w12=w12,
        w21=w21,
        V0_1=V0_1,
        V0_2=V0_2,
        a0_1=A0,
        a0_2=A0,
        tau_syn=TAU_SYN,
    )
    return tr, rate_metrics(tr)


def plot_results(off_tr: dict, on_tr: dict, off_m: dict, on_m: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(on_tr["t"], on_tr["V1"], lw=0.7, label="V1 on")
    ax.plot(on_tr["t"], on_tr["V2"], lw=0.7, label="V2 on")
    ax.axvspan(T_ON, T_OFF, color="0.9", zorder=0)
    ax.set_ylabel("V [mV]")
    ax.set_title("E1 voltages (coupling on)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(on_tr["t"], on_tr["a1"], lw=1.0, label="a1 on")
    ax.plot(on_tr["t"], on_tr["a2"], lw=1.0, label="a2 on")
    ax.plot(off_tr["t"], off_tr["a1"], lw=0.8, ls="--", color="0.5", label="a1 off")
    ax.set_ylabel("a")
    ax.set_xlabel("t [ms]")
    ax.set_title("ATP")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    labels = ["cell1", "cell2"]
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width / 2, [off_m["r1"], off_m["r2"]], width, label="off", color="C0")
    ax.bar(x + width / 2, [on_m["r1"], on_m["r2"]], width, label="on", color="C3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean soft-spike (drive window)")
    ax.set_title(
        f"Rates  off ratio={off_m['ratio']:.2f}  on ratio={on_m['ratio']:.2f}"
    )
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.bar(x - width / 2, [off_m["da1"], off_m["da2"]], width, label="off", color="C0")
    ax.bar(x + width / 2, [on_m["da1"], on_m["da2"]], width, label="on", color="C3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\Delta a$")
    ax.set_title("ATP drawdown")
    ax.legend(fontsize=8)

    fig.suptitle(
        rf"B1 two-cell: $I_{{\mathrm{{drive}}}}={I_DRIVE}$, "
        rf"$w_{{12}},w_{{21}}={W_ON}$ (struct. asym. inhib.), "
        rf"$V_0=({V0_1},{V0_2})$",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path.resolve()}")


def main() -> int:
    print(
        f"B1: shared I_drive={I_DRIVE}, tau_syn={TAU_SYN}, "
        f"V0=({V0_1},{V0_2}), a0=({A0},{A0}); K_e/K_ATP/gNa off"
    )
    print(f"seed: tiny V0 asymmetry; E1 weights structural asym. inhib. {W_ON}")
    _, I_drive = make_drive()

    print("--- E0 coupling OFF ---")
    off_tr, off_m = run_case(I_drive, *W_OFF)
    print(
        f"  r1={off_m['r1']:.4f} r2={off_m['r2']:.4f} ratio={off_m['ratio']:.3f} "
        f"Δa=({off_m['da1']:.3f},{off_m['da2']:.3f})"
    )

    print("--- E1 coupling ON ---")
    on_tr, on_m = run_case(I_drive, *W_ON)
    print(
        f"  r1={on_m['r1']:.4f} r2={on_m['r2']:.4f} ratio={on_m['ratio']:.3f} "
        f"Δa=({on_m['da1']:.3f},{on_m['da2']:.3f})"
    )

    out = Path("figures") / "b1_two_cell_asymm.png"
    plot_results(off_tr, on_tr, off_m, on_m, out)

    ok_ratio = on_m["ratio"] >= 1.5
    ok_vs_off = on_m["ratio"] > off_m["ratio"] + 0.2
    active = 2 if on_m["r2"] >= on_m["r1"] else 1
    da_active = on_m["da2"] if active == 2 else on_m["da1"]
    da_other = on_m["da1"] if active == 2 else on_m["da2"]
    atp_align = da_active >= da_other - 0.01
    print("--- success ① ---")
    print(f"  ratio≥1.5: {ok_ratio} ({on_m['ratio']:.3f})")
    print(f"  larger than off: {ok_vs_off} (off {off_m['ratio']:.3f})")
    print(f"  higher-rate cell has ≥ Δa: {atp_align} (active=cell{active})")
    if ok_ratio and ok_vs_off:
        print("OK: asymmetry / slack preview (not network intelligence)")
        status = "success"
    elif ok_vs_off:
        print("PARTIAL: some asymmetry vs off, below ratio target")
        status = "partial"
    else:
        print("NULL: no clear coupling-driven asymmetry")
        status = "null"
    print(f"outcome: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

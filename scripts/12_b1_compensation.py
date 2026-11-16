"""Phase B1-② — metabolic shoulder-takeover (compensation) probe.

Run from the repo root::

    python scripts/12_b1_compensation.py

Uses sweep-recommended w=(-15, -3). Shared I_drive / τ_syn / V0 seed as B1.
Per-cell a_max stress on high- vs low-firing cell; couple on vs off.
No learning, no diffusion, no ≥3 cells.

Claim ceiling: coupled partner may raise firing under partner ATP stress —
not rehab / recovery / brain injury compensation.
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

# Match B1 / w-sweep
T_END = 500.0
T_ON = 20.0
T_OFF = 480.0
I_DRIVE = 9.0
TAU_SYN = 8.0
V0_1 = -65.0
V0_2 = -64.2

# Sweep-recommended structural inhibition (NOT old (-80,-10))
W_ON = (-15.0, -3.0)
W_OFF = (0.0, 0.0)

# a_max grid: baseline + stress levels
A_MAX_GRID = (1.0, 0.4, 0.2)


def make_drive() -> jnp.ndarray:
    t = jnp.arange(0.0, T_END + DT / 2, DT)
    return I_ext_step(t, T_ON, T_OFF, I_DRIVE)


def metrics(tr: dict) -> dict[str, float]:
    t = np.asarray(tr["t"])
    s1 = np.asarray(tr["s1"])
    s2 = np.asarray(tr["s2"])
    mask = (t >= T_ON) & (t < T_OFF)
    r1 = float(s1[mask].mean())
    r2 = float(s2[mask].mean())
    da1 = float(tr["a1"][0] - tr["a1"][-1])
    da2 = float(tr["a2"][0] - tr["a2"][-1])
    vmin = float(min(np.asarray(tr["V1"]).min(), np.asarray(tr["V2"]).min()))
    return {
        "r1": r1,
        "r2": r2,
        "ratio": max(r1, r2) / max(min(r1, r2), 1e-9),
        "da1": da1,
        "da2": da2,
        "Vmin": vmin,
    }


def run_case(
    I_drive: jnp.ndarray,
    *,
    w12: float,
    w21: float,
    a_max_1: float,
    a_max_2: float,
) -> tuple[dict, dict]:
    tr = simulate_two_cell(
        I_drive,
        w12=w12,
        w21=w21,
        V0_1=V0_1,
        V0_2=V0_2,
        a_max_1=a_max_1,
        a_max_2=a_max_2,
        tau_syn=TAU_SYN,
    )
    return tr, metrics(tr)


def plot_results(
    *,
    base_tr: dict,
    stress_hi_tr: dict,
    rows_on: list[dict],
    rows_off: list[dict],
    base_m: dict,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(base_tr["t"], base_tr["V1"], lw=0.7, label="V1 base")
    ax.plot(base_tr["t"], base_tr["V2"], lw=0.7, label="V2 base")
    ax.axvspan(T_ON, T_OFF, color="0.9", zorder=0)
    ax.set_ylabel("V [mV]")
    ax.set_title(f"F0 baseline (on) r1={base_m['r1']:.4f} r2={base_m['r2']:.4f}")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(stress_hi_tr["t"], stress_hi_tr["V1"], lw=0.7, label="V1 a1↓")
    ax.plot(stress_hi_tr["t"], stress_hi_tr["V2"], lw=0.7, label="V2 a1↓")
    ax.axvspan(T_ON, T_OFF, color="0.9", zorder=0)
    ax.set_ylabel("V [mV]")
    ax.set_title("F1 stress high-firing cell (a_max1=0.2, on)")
    ax.legend(fontsize=7)

    # Bar: stress-a1 series on vs off — r1 and r2 vs a_max1
    ax = axes[1, 0]
    stress_a1_on = [r for r in rows_on if r["tag"].startswith("stress_a1")]
    stress_a1_off = [r for r in rows_off if r["tag"].startswith("stress_a1")]
    xs = np.arange(len(stress_a1_on))
    w = 0.18
    ax.bar(xs - 1.5 * w, [r["r1"] for r in stress_a1_on], w, label="r1 on", color="C0")
    ax.bar(xs - 0.5 * w, [r["r2"] for r in stress_a1_on], w, label="r2 on", color="C3")
    ax.bar(xs + 0.5 * w, [r["r1"] for r in stress_a1_off], w, label="r1 off", color="C0", alpha=0.35)
    ax.bar(xs + 1.5 * w, [r["r2"] for r in stress_a1_off], w, label="r2 off", color="C3", alpha=0.35)
    ax.axhline(base_m["r2"], color="C3", ls="--", lw=1.0, label=f"r2 base={base_m['r2']:.4f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"a1={r['a_max_1']}" for r in stress_a1_on])
    ax.set_ylabel("mean soft-spike")
    ax.set_title("Stress high-firing (a_max1↓): r on vs off")
    ax.legend(fontsize=6, ncol=2)

    # Compensation Δr2 = r2(stress) - r2(base) for a1↓; and Δr1 for a2↓
    ax = axes[1, 1]
    labels = []
    d_on = []
    d_off = []
    for a in (0.4, 0.2):
        on = next(r for r in rows_on if r["tag"] == f"stress_a1_{a}")
        off = next(r for r in rows_off if r["tag"] == f"stress_a1_{a}")
        labels.append(f"a1={a}\nΔr2")
        d_on.append(on["r2"] - base_m["r2"])
        d_off.append(off["r2"] - next(r for r in rows_off if r["tag"] == "base")["r2"])
    base_off = next(r for r in rows_off if r["tag"] == "base")
    for a in (0.4, 0.2):
        on = next(r for r in rows_on if r["tag"] == f"stress_a2_{a}")
        off = next(r for r in rows_off if r["tag"] == f"stress_a2_{a}")
        labels.append(f"a2={a}\nΔr1")
        d_on.append(on["r1"] - base_m["r1"])
        d_off.append(off["r1"] - base_off["r1"])
    xs = np.arange(len(labels))
    ax.bar(xs - 0.2, d_on, 0.4, label="couple on", color="C2")
    ax.bar(xs + 0.2, d_off, 0.4, label="couple off", color="0.6")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Δr partner vs own baseline")
    ax.set_title("Compensation Δr (partner rise)")
    ax.legend(fontsize=7)

    fig.suptitle(
        r"B1-② compensation  $w=(-15,-3)$  (K_e/K_ATP/gNa off)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "figures" / "b1_compensation.png"

    I = make_drive()
    rows_on: list[dict] = []
    rows_off: list[dict] = []
    traces: dict[str, dict] = {}

    print(
        f"B1-② compensation: W_ON={W_ON}, I_drive={I_DRIVE}, "
        f"tau_syn={TAU_SYN}, V0=({V0_1},{V0_2}) — gates off"
    )

    # F0 / F2 baseline
    for couple, w, bucket in (("on", W_ON, rows_on), ("off", W_OFF, rows_off)):
        tr, m = run_case(I, w12=w[0], w21=w[1], a_max_1=1.0, a_max_2=1.0)
        row = {"tag": "base", "couple": couple, "a_max_1": 1.0, "a_max_2": 1.0, **m}
        bucket.append(row)
        if couple == "on":
            traces["base"] = tr
        print(
            f"F0/{couple}: a=(1,1) r=({m['r1']:.4f},{m['r2']:.4f}) "
            f"ratio={m['ratio']:.3f} Δa=({m['da1']:.3f},{m['da2']:.3f}) "
            f"Vmin={m['Vmin']:.1f}  high={'cell1' if m['r1'] >= m['r2'] else 'cell2'}"
        )

    base_on = rows_on[0]
    high = "cell1" if base_on["r1"] >= base_on["r2"] else "cell2"
    print(f"--- baseline high-firing = {high} (fixed for F1 labels) ---")

    # F1 / F2 stress grid
    for a_stress in A_MAX_GRID:
        if a_stress >= 1.0:
            continue
        # Stress high-firing cell (cell1 expected)
        for couple, w, bucket in (("on", W_ON, rows_on), ("off", W_OFF, rows_off)):
            tr, m = run_case(
                I, w12=w[0], w21=w[1], a_max_1=a_stress, a_max_2=1.0
            )
            tag = f"stress_a1_{a_stress}"
            bucket.append(
                {
                    "tag": tag,
                    "couple": couple,
                    "a_max_1": a_stress,
                    "a_max_2": 1.0,
                    **m,
                }
            )
            if couple == "on" and a_stress == 0.2:
                traces["stress_a1_0.2"] = tr
            dr2 = m["r2"] - (base_on["r2"] if couple == "on" else rows_off[0]["r2"])
            print(
                f"F1-a1/{couple}: a=({a_stress},1) r=({m['r1']:.4f},{m['r2']:.4f}) "
                f"Δa=({m['da1']:.3f},{m['da2']:.3f}) Δr2={dr2:+.4f}"
            )

        # Stress low-firing cell (cell2 expected)
        for couple, w, bucket in (("on", W_ON, rows_on), ("off", W_OFF, rows_off)):
            tr, m = run_case(
                I, w12=w[0], w21=w[1], a_max_1=1.0, a_max_2=a_stress
            )
            tag = f"stress_a2_{a_stress}"
            bucket.append(
                {
                    "tag": tag,
                    "couple": couple,
                    "a_max_1": 1.0,
                    "a_max_2": a_stress,
                    **m,
                }
            )
            dr1 = m["r1"] - (base_on["r1"] if couple == "on" else rows_off[0]["r1"])
            print(
                f"F1-a2/{couple}: a=(1,{a_stress}) r=({m['r1']:.4f},{m['r2']:.4f}) "
                f"Δa=({m['da1']:.3f},{m['da2']:.3f}) Δr1={dr1:+.4f}"
            )

    plot_results(
        base_tr=traces["base"],
        stress_hi_tr=traces["stress_a1_0.2"],
        rows_on=rows_on,
        rows_off=rows_off,
        base_m=base_on,
        out_path=out,
    )
    print(f"saved: {out}")

    # Verdict helpers
    def row_on(tag: str) -> dict:
        return next(r for r in rows_on if r["tag"] == tag)

    def row_off(tag: str) -> dict:
        return next(r for r in rows_off if r["tag"] == tag)

    print("--- compensation check (partner Δr) ---")
    for a in (0.4, 0.2):
        on = row_on(f"stress_a1_{a}")
        off = row_off(f"stress_a1_{a}")
        d_on = on["r2"] - base_on["r2"]
        d_off = off["r2"] - rows_off[0]["r2"]
        print(
            f"  a1={a}: stressed r1 {base_on['r1']:.4f}→{on['r1']:.4f}; "
            f"partner Δr2 on={d_on:+.4f} off={d_off:+.4f}"
        )
    for a in (0.4, 0.2):
        on = row_on(f"stress_a2_{a}")
        off = row_off(f"stress_a2_{a}")
        d_on = on["r1"] - base_on["r1"]
        d_off = off["r1"] - rows_off[0]["r1"]
        print(
            f"  a2={a}: stressed r2 {base_on['r2']:.4f}→{on['r2']:.4f}; "
            f"partner Δr1 on={d_on:+.4f} off={d_off:+.4f}"
        )

    # Auto verdict (honest)
    a = 0.2
    on = row_on(f"stress_a1_{a}")
    off = row_off(f"stress_a1_{a}")
    r1_down = on["r1"] < base_on["r1"] - 1e-4
    d_on = on["r2"] - base_on["r2"]
    d_off = off["r2"] - rows_off[0]["r2"]
    partner_up = d_on > 1e-4
    couple_helps = d_on > d_off + 1e-4
    if r1_down and partner_up and couple_helps:
        verdict = "success"
        why = "stress high-firer ↓r1; partner ↑r2 more with coupling than without"
    elif r1_down and partner_up:
        verdict = "partial"
        why = "partner rises on, but off rise similar (coupling not decisive)"
    elif r1_down:
        verdict = "partial" if abs(d_on) < 1e-3 else "null"
        why = (
            "stressed cell slows but partner does not rise (or falls)"
            if not partner_up
            else "unclear"
        )
        if not partner_up:
            verdict = "null"
    else:
        verdict = "null"
        why = "stressed high-firer rate did not clearly fall"
    print(f"VERDICT: {verdict} — {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

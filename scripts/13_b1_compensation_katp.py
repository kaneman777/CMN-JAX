"""Phase B1-②′ — compensation retry with simplified K_ATP on both cells.

Run from the repo root::

    python scripts/13_b1_compensation_katp.py

Same protocol as ``12_b1_compensation.py`` (w=(-15,-3), shared drive / τ / V0),
but Phase-4-Bio simplified K_ATP enabled identically on both cells.
ATP→gNa phenomenological gate stays off (K_ATP only). Optional one-shot
K_ATP-off baseline reproduces ② null.

Claim ceiling: K_ATP-gated metabolic stress may show shoulder-takeover
preview — not rehab / injury compensation.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jnpnp.models.point_sim import DT, I_ext_step, merge_params
from jnpnp.models.two_cell import simulate_two_cell

# Match B1 / ②
T_END = 500.0
T_ON = 20.0
T_OFF = 480.0
I_DRIVE = 9.0
TAU_SYN = 8.0
V0_1 = -65.0
V0_2 = -64.2
W_ON = (-15.0, -3.0)
W_OFF = (0.0, 0.0)
A_STRESS = (0.4, 0.2)

# Phase 4-Bio C defaults (identical on both cells)
G_KATP_BAR = 0.3
A_KATP_HALF = 0.2
N_KATP = 2.0


def params_katp_on() -> dict[str, float]:
    return merge_params(
        g_KATP_bar=G_KATP_BAR,
        a_KATP_half=A_KATP_HALF,
        n_KATP=N_KATP,
        atp_gNa_floor=0.0,  # gNa phenomenological gate OFF
    )


def params_katp_off() -> dict[str, float]:
    return merge_params(g_KATP_bar=0.0, atp_gNa_floor=0.0)


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
    params: dict[str, float],
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
        params=params,
    )
    return tr, metrics(tr)


def plot_results(
    *,
    base_tr: dict,
    stress_tr: dict,
    rows_on: list[dict],
    rows_off: list[dict],
    base_on: dict,
    base_off: dict,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(base_tr["t"], base_tr["V1"], lw=0.7, label="V1")
    ax.plot(base_tr["t"], base_tr["V2"], lw=0.7, label="V2")
    ax.axvspan(T_ON, T_OFF, color="0.9", zorder=0)
    ax.set_ylabel("V [mV]")
    ax.set_title(
        f"G0 K_ATP on baseline  r=({base_on['r1']:.4f},{base_on['r2']:.4f})"
    )
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(stress_tr["t"], stress_tr["V1"], lw=0.7, label="V1 a1=0.2")
    ax.plot(stress_tr["t"], stress_tr["V2"], lw=0.7, label="V2 a1=0.2")
    ax.axvspan(T_ON, T_OFF, color="0.9", zorder=0)
    ax.set_ylabel("V [mV]")
    ax.set_title("G1 stress high-firer (a_max1=0.2, couple on)")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    on_a1 = [r for r in rows_on if r["tag"].startswith("a1_")]
    off_a1 = [r for r in rows_off if r["tag"].startswith("a1_")]
    # prepend baselines for bars
    series_on = [base_on] + on_a1
    series_off = [base_off] + off_a1
    xs = np.arange(len(series_on))
    w = 0.18
    labels = ["base"] + [f"a1={r['a_max_1']}" for r in on_a1]
    ax.bar(xs - 1.5 * w, [r["r1"] for r in series_on], w, label="r1 on", color="C0")
    ax.bar(xs - 0.5 * w, [r["r2"] for r in series_on], w, label="r2 on", color="C3")
    ax.bar(
        xs + 0.5 * w,
        [r["r1"] for r in series_off],
        w,
        label="r1 off",
        color="C0",
        alpha=0.35,
    )
    ax.bar(
        xs + 1.5 * w,
        [r["r2"] for r in series_off],
        w,
        label="r2 off",
        color="C3",
        alpha=0.35,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean soft-spike")
    ax.set_title("a_max1↓ rates: couple on vs off")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[1, 1]
    labels = []
    d_on = []
    d_off = []
    for a in A_STRESS:
        on = next(r for r in rows_on if r["tag"] == f"a1_{a}")
        off = next(r for r in rows_off if r["tag"] == f"a1_{a}")
        labels.append(f"a1={a}\nΔr2")
        d_on.append(on["r2"] - base_on["r2"])
        d_off.append(off["r2"] - base_off["r2"])
    for a in A_STRESS:
        on = next(r for r in rows_on if r["tag"] == f"a2_{a}")
        off = next(r for r in rows_off if r["tag"] == f"a2_{a}")
        labels.append(f"a2={a}\nΔr1")
        d_on.append(on["r1"] - base_on["r1"])
        d_off.append(off["r1"] - base_off["r1"])
    xs = np.arange(len(labels))
    ax.bar(xs - 0.2, d_on, 0.4, label="couple on", color="C2")
    ax.bar(xs + 0.2, d_off, 0.4, label="couple off", color="0.6")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Δr partner vs baseline")
    ax.set_title("Compensation Δr (partner)")
    ax.legend(fontsize=7)

    fig.suptitle(
        rf"B1-②′ K_ATP  $w=(-15,-3)$  "
        rf"$\bar g$={G_KATP_BAR}, $a_{{1/2}}$={A_KATP_HALF}  (gNa off)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "figures" / "b1_compensation_katp.png"
    I = make_drive()
    p_on = params_katp_on()
    p_off_ctrl = params_katp_off()

    print(
        f"B1-②′ K_ATP compensation: W_ON={W_ON}, I={I_DRIVE}, tau={TAU_SYN}, "
        f"V0=({V0_1},{V0_2})"
    )
    print(
        f"K_ATP: g_bar={G_KATP_BAR}, a_half={A_KATP_HALF}, n={N_KATP}; "
        f"atp_gNa_floor=0 (gNa off)"
    )

    # Optional ② reproduction (K_ATP off, couple on, a1=0.2)
    tr, m = run_case(
        I, w12=W_ON[0], w21=W_ON[1], a_max_1=0.2, a_max_2=1.0, params=p_off_ctrl
    )
    print(
        f"ctrl K_ATP-off a=(0.2,1) on: r=({m['r1']:.4f},{m['r2']:.4f}) "
        f"(②-like null check)"
    )

    rows_on: list[dict] = []
    rows_off: list[dict] = []
    traces: dict[str, dict] = {}

    # G0 baselines
    for couple, w, bucket, key in (
        ("on", W_ON, rows_on, "base_on"),
        ("off", W_OFF, rows_off, "base_off"),
    ):
        tr, m = run_case(
            I, w12=w[0], w21=w[1], a_max_1=1.0, a_max_2=1.0, params=p_on
        )
        row = {
            "tag": "base",
            "couple": couple,
            "a_max_1": 1.0,
            "a_max_2": 1.0,
            **m,
        }
        bucket.append(row)
        traces[key] = tr
        hi = "cell1" if m["r1"] >= m["r2"] else "cell2"
        print(
            f"G0/{couple}: a=(1,1) r=({m['r1']:.4f},{m['r2']:.4f}) "
            f"ratio={m['ratio']:.3f} Δa=({m['da1']:.3f},{m['da2']:.3f}) "
            f"Vmin={m['Vmin']:.1f} high={hi}"
        )

    base_on = rows_on[0]
    base_off = rows_off[0]
    high = "cell1" if base_on["r1"] >= base_on["r2"] else "cell2"
    print(f"--- G0 high-firing = {high} ---")

    # G1/G2: stress a_max1; G3: stress a_max2
    for a in A_STRESS:
        for couple, w, bucket in (("on", W_ON, rows_on), ("off", W_OFF, rows_off)):
            tr, m = run_case(
                I, w12=w[0], w21=w[1], a_max_1=a, a_max_2=1.0, params=p_on
            )
            base = base_on if couple == "on" else base_off
            dr2 = m["r2"] - base["r2"]
            bucket.append(
                {
                    "tag": f"a1_{a}",
                    "couple": couple,
                    "a_max_1": a,
                    "a_max_2": 1.0,
                    **m,
                }
            )
            if couple == "on" and a == 0.2:
                traces["stress_a1"] = tr
            print(
                f"G1-a1/{couple}: a=({a},1) r=({m['r1']:.4f},{m['r2']:.4f}) "
                f"Δa=({m['da1']:.3f},{m['da2']:.3f}) "
                f"Δr1={m['r1']-base['r1']:+.4f} Δr2={dr2:+.4f}"
            )

        for couple, w, bucket in (("on", W_ON, rows_on), ("off", W_OFF, rows_off)):
            tr, m = run_case(
                I, w12=w[0], w21=w[1], a_max_1=1.0, a_max_2=a, params=p_on
            )
            base = base_on if couple == "on" else base_off
            dr1 = m["r1"] - base["r1"]
            bucket.append(
                {
                    "tag": f"a2_{a}",
                    "couple": couple,
                    "a_max_1": 1.0,
                    "a_max_2": a,
                    **m,
                }
            )
            print(
                f"G3-a2/{couple}: a=(1,{a}) r=({m['r1']:.4f},{m['r2']:.4f}) "
                f"Δa=({m['da1']:.3f},{m['da2']:.3f}) "
                f"Δr1={dr1:+.4f} Δr2={m['r2']-base['r2']:+.4f}"
            )

    plot_results(
        base_tr=traces["base_on"],
        stress_tr=traces["stress_a1"],
        rows_on=rows_on,
        rows_off=rows_off,
        base_on=base_on,
        base_off=base_off,
        out_path=out,
    )
    print(f"saved: {out}")

    # Verdict on primary G1 a1=0.2
    a = 0.2
    on = next(r for r in rows_on if r["tag"] == f"a1_{a}")
    off = next(r for r in rows_off if r["tag"] == f"a1_{a}")
    r1_down = on["r1"] < base_on["r1"] - 5e-4
    d_on = on["r2"] - base_on["r2"]
    d_off = off["r2"] - base_off["r2"]
    partner_up = d_on > 5e-4
    couple_helps = d_on > d_off + 5e-4
    print("--- primary check a_max1=0.2 ---")
    print(
        f"  r1 {base_on['r1']:.4f}→{on['r1']:.4f}; "
        f"Δr2 on={d_on:+.4f} off={d_off:+.4f}"
    )
    if r1_down and partner_up and couple_helps:
        verdict, why = (
            "success",
            "r1↓ and partner r2↑ more with coupling than without",
        )
    elif r1_down and partner_up:
        verdict, why = (
            "partial",
            "r1↓ and partner↑ but off rise similar (coupling not decisive)",
        )
    elif r1_down:
        verdict, why = (
            "partial",
            "stressed r1 falls but partner does not clearly rise",
        )
    else:
        verdict, why = (
            "null",
            "stressed high-firer rate did not clearly fall "
            "(2-cell+drive differs from 4-Bio single-cell)",
        )
    print(f"VERDICT: {verdict} — {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

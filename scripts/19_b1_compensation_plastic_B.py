"""B1 compensation + Plan-B minimal plasticity (not Plan A).

Run from the repo root::

    python scripts/19_b1_compensation_plastic_B.py

Plan B: when cell1 is ATP-poor, weaken |w21| (cell1→cell2 inhibition) so
cell2 can rise — shoulder-takeover *preview* under K_ATP. Plan A (relax
w12 / protect the weak) is **not** implemented here.

u1 = clip(1 - a1 / a_ref, 0, 1) with a_ref=1 (healthy ATP scale).
|w21_eff| = |w21| * (1 - α u1); w12 fixed.

Claim ceiling: minimal plasticity + stress may show compensation preview —
not rehab / injury compensation / Plan A.
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

# Match B1 / ②′
T_END = 500.0
T_ON = 20.0
T_OFF = 480.0
I_DRIVE = 9.0
TAU_SYN = 8.0
V0_1 = -65.0
V0_2 = -64.2
W_ON = (-15.0, -3.0)  # w12, w21
W_OFF = (0.0, 0.0)
A_STRESS = (0.4, 0.2)

G_KATP_BAR = 0.3
A_KATP_HALF = 0.2
N_KATP = 2.0

# Plan B plasticity
ALPHA = 0.85
A_REF = 1.0  # healthy ATP reference for u1 (not a_max1)


def params_katp_on() -> dict[str, float]:
    return merge_params(
        g_KATP_bar=G_KATP_BAR,
        a_KATP_half=A_KATP_HALF,
        n_KATP=N_KATP,
        atp_gNa_floor=0.0,
    )


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
    w_eff = np.asarray(tr["w21_eff"])
    abs_w = np.abs(w_eff[mask])
    return {
        "r1": r1,
        "r2": r2,
        "ratio": max(r1, r2) / max(min(r1, r2), 1e-9),
        "da1": float(tr["a1"][0] - tr["a1"][-1]),
        "da2": float(tr["a2"][0] - tr["a2"][-1]),
        "a1_mean": float(np.asarray(tr["a1"])[mask].mean()),
        "abs_w21_mean": float(abs_w.mean()),
        "abs_w21_min": float(abs_w.min()),
        "Vmin": float(
            min(np.asarray(tr["V1"]).min(), np.asarray(tr["V2"]).min())
        ),
    }


def run_case(
    I_drive: jnp.ndarray,
    *,
    w12: float,
    w21: float,
    a_max_1: float,
    a_max_2: float,
    plastic_B: bool,
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
        plastic_B=plastic_B,
        plastic_alpha=ALPHA if plastic_B else 0.0,
        plastic_a_ref=A_REF,
    )
    return tr, metrics(tr)


def plot_results(
    *,
    p0: dict,
    rows: list[dict],
    traces: dict[str, dict],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    ax = axes[0, 0]
    tr = traces["P0"]
    ax.plot(tr["t"], tr["V1"], lw=0.7, label="V1")
    ax.plot(tr["t"], tr["V2"], lw=0.7, label="V2")
    ax.axvspan(T_ON, T_OFF, color="0.9", zorder=0)
    ax.set_ylabel("V [mV]")
    ax.set_title(f"P0 baseline  r=({p0['r1']:.4f},{p0['r2']:.4f})")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    tr = traces["P1_0.2"]
    ax.plot(tr["t"], tr["V1"], lw=0.7, label="V1")
    ax.plot(tr["t"], tr["V2"], lw=0.7, label="V2")
    ax.axvspan(T_ON, T_OFF, color="0.9", zorder=0)
    ax.set_ylabel("V [mV]")
    ax.set_title("P1 Plan-B on  a_max1=0.2")
    ax.legend(fontsize=7)

    # rates: P0 + P1/P2/P3 at each a
    ax = axes[1, 0]
    labels = ["P0"]
    r1s = [p0["r1"]]
    r2s = [p0["r2"]]
    for a in A_STRESS:
        for tag, name in (
            (f"P1_{a}", f"P1\na1={a}"),
            (f"P2_{a}", f"P2\na1={a}"),
            (f"P3_{a}", f"P3\na1={a}"),
        ):
            row = next(r for r in rows if r["phase"] == tag)
            labels.append(name)
            r1s.append(row["r1"])
            r2s.append(row["r2"])
    xs = np.arange(len(labels))
    ax.bar(xs - 0.2, r1s, 0.4, label="r1", color="C0")
    ax.bar(xs + 0.2, r2s, 0.4, label="r2", color="C3")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("mean soft-spike")
    ax.set_title("Rates (Plan B vs controls)")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    # Δr2 vs P0 for P1/P2/P3
    labs = []
    vals = []
    colors = []
    for a in A_STRESS:
        for tag, c in ((f"P1_{a}", "C2"), (f"P2_{a}", "C1"), (f"P3_{a}", "0.55")):
            row = next(r for r in rows if r["phase"] == tag)
            labs.append(f"{tag}\nΔr2")
            vals.append(row["r2"] - p0["r2"])
            colors.append(c)
    xs = np.arange(len(labs))
    ax.bar(xs, vals, color=colors)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel("Δr2 vs P0")
    ax.set_title("Partner rise (green=P1 PlanB)")

    fig.suptitle(
        rf"B1 Plan-B plastic  $w=(-15,-3)$  α={ALPHA}  "
        rf"$u_1=1-a_1/a_{{\mathrm{{ref}}}}$  K_ATP on  (not Plan A)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "figures" / "b1_compensation_plastic_B.png"
    I = make_drive()
    params = params_katp_on()

    print(
        "B1 Plan-B plasticity (NOT Plan A): "
        f"W_ON={W_ON}, α={ALPHA}, a_ref={A_REF}, "
        f"u1=clip(1-a1/a_ref,0,1); |w21| shrinks when cell1 ATP-poor; w12 fixed"
    )
    print(
        f"K_ATP: g_bar={G_KATP_BAR}, a_half={A_KATP_HALF}; gNa off; "
        f"drive I={I_DRIVE}, τ={TAU_SYN} (②′ match)"
    )

    rows: list[dict] = []
    traces: dict[str, dict] = {}

    # P0 baseline: a=(1,1), couple on, plastic off
    tr, m = run_case(
        I,
        w12=W_ON[0],
        w21=W_ON[1],
        a_max_1=1.0,
        a_max_2=1.0,
        plastic_B=False,
        params=params,
    )
    p0 = {"phase": "P0", "a_max_1": 1.0, "plastic": False, "couple": True, **m}
    rows.append(p0)
    traces["P0"] = tr
    hi = "cell1" if m["r1"] >= m["r2"] else "cell2"
    print(
        f"P0: a=(1,1) plastic=off couple=on "
        f"r=({m['r1']:.4f},{m['r2']:.4f}) |w21|={m['abs_w21_mean']:.3f} "
        f"high={hi}"
    )

    for a in A_STRESS:
        # P1: stress + Plan B on + couple on
        tr, m = run_case(
            I,
            w12=W_ON[0],
            w21=W_ON[1],
            a_max_1=a,
            a_max_2=1.0,
            plastic_B=True,
            params=params,
        )
        row = {
            "phase": f"P1_{a}",
            "a_max_1": a,
            "plastic": True,
            "couple": True,
            **m,
        }
        rows.append(row)
        traces[f"P1_{a}"] = tr
        print(
            f"P1 a1={a}: plastic=ON  r=({m['r1']:.4f},{m['r2']:.4f}) "
            f"Δr1={m['r1']-p0['r1']:+.4f} Δr2={m['r2']-p0['r2']:+.4f} "
            f"|w21|_mean={m['abs_w21_mean']:.3f} min={m['abs_w21_min']:.3f} "
            f"a1_mean={m['a1_mean']:.3f}"
        )

        # P2: same stress, plastic OFF, couple on (②′-like)
        tr, m = run_case(
            I,
            w12=W_ON[0],
            w21=W_ON[1],
            a_max_1=a,
            a_max_2=1.0,
            plastic_B=False,
            params=params,
        )
        row = {
            "phase": f"P2_{a}",
            "a_max_1": a,
            "plastic": False,
            "couple": True,
            **m,
        }
        rows.append(row)
        traces[f"P2_{a}"] = tr
        print(
            f"P2 a1={a}: plastic=off r=({m['r1']:.4f},{m['r2']:.4f}) "
            f"Δr1={m['r1']-p0['r1']:+.4f} Δr2={m['r2']-p0['r2']:+.4f} "
            f"|w21|={m['abs_w21_mean']:.3f}"
        )

        # P3: Plan B on but w=0
        tr, m = run_case(
            I,
            w12=W_OFF[0],
            w21=W_OFF[1],
            a_max_1=a,
            a_max_2=1.0,
            plastic_B=True,
            params=params,
        )
        row = {
            "phase": f"P3_{a}",
            "a_max_1": a,
            "plastic": True,
            "couple": False,
            **m,
        }
        rows.append(row)
        traces[f"P3_{a}"] = tr
        print(
            f"P3 a1={a}: couple=OFF  r=({m['r1']:.4f},{m['r2']:.4f}) "
            f"Δr1={m['r1']-p0['r1']:+.4f} Δr2={m['r2']-p0['r2']:+.4f}"
        )

    plot_results(p0=p0, rows=rows, traces=traces, out_path=out)
    print(f"saved: {out}")

    # Primary success check at a_max1=0.2
    a = 0.2
    p1 = next(r for r in rows if r["phase"] == f"P1_{a}")
    p2 = next(r for r in rows if r["phase"] == f"P2_{a}")
    p3 = next(r for r in rows if r["phase"] == f"P3_{a}")
    r1_down = p1["r1"] < p0["r1"] - 5e-4
    r2_up = p1["r2"] > p0["r2"] + 5e-4
    ctrl_weak = (p1["r2"] - p0["r2"]) > (p2["r2"] - p0["r2"]) + 5e-4
    no_couple = (p1["r2"] - p0["r2"]) > (p3["r2"] - p0["r2"]) + 5e-4
    print("--- primary a_max1=0.2 ---")
    print(
        f"  P0 r2={p0['r2']:.4f}; P1 Δr2={p1['r2']-p0['r2']:+.4f}; "
        f"P2 Δr2={p2['r2']-p0['r2']:+.4f}; P3 Δr2={p3['r2']-p0['r2']:+.4f}"
    )
    if r1_down and r2_up and ctrl_weak and no_couple:
        verdict, why = (
            "success",
            "P1: r1↓ and r2↑; P2/P3 lack the partner rise",
        )
    elif r1_down and r2_up and (ctrl_weak or no_couple):
        verdict, why = (
            "partial",
            "P1 shows r1↓+r2↑ but one control is weak/ambiguous",
        )
    elif r1_down and r2_up:
        verdict, why = (
            "partial",
            "P1 r1↓+r2↑ but controls also rise (plasticity not decisive)",
        )
    elif r1_down:
        verdict, why = (
            "partial",
            "r1↓ under stress but Plan-B partner rise absent/weak",
        )
    else:
        verdict, why = ("null", "stressed r1 did not clearly fall")
    print(f"VERDICT: {verdict} — {why}")
    print("claim: Plan-B preview only — not rehab; not Plan A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

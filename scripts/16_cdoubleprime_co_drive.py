"""Phase C′′ — re-score syn/diff/both under B1-style co-drive.

Run from the repo root::

    python scripts/16_cdoubleprime_co_drive.py

Same four conditions as C′, but both cells share I_drive in one window
(so synaptic push–pull can enter the score). Contrast: C′ cue→probe was
diff-dominated.

Claim ceiling: task dependence of train vs bus readouts —
not optimal hybrid brain communication.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jnpnp.models.point_sim import DT, I_ext_step, merge_params
from jnpnp.models.two_cell import simulate_two_cell_ke

# Match C′ weights / D; B1 co-drive
GAMMA_KE = 0.0005
TAU_K = 1000.0
K_BATH = 4.0
D_KE = 0.005
W_ON = (-15.0, -3.0)
TAU_SYN = 8.0

T_END = 500.0
T_ON = 20.0
T_OFF = 480.0
I_DRIVE = 9.0

# B1 tiny V0 seed (document)
V0_1 = -65.0
V0_2 = -64.2

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


def make_shared_drive() -> jnp.ndarray:
    t = jnp.arange(0.0, T_END + DT / 2, DT)
    return I_ext_step(t, T_ON, T_OFF, I_DRIVE)


def summarize(tr: dict, name: str, w12: float, w21: float, D: float) -> dict:
    t = np.asarray(tr["t"])
    s1 = np.asarray(tr["s1"])
    s2 = np.asarray(tr["s2"])
    ke1 = np.asarray(tr["K_e1"])
    ke2 = np.asarray(tr["K_e2"])
    drive = (t >= T_ON) & (t < T_OFF)
    r1 = float(s1[drive].mean())
    r2 = float(s2[drive].mean())
    ratio = max(r1, r2) / max(min(r1, r2), 1e-9)
    da1 = float(tr["a1"][0] - tr["a1"][-1])
    da2 = float(tr["a2"][0] - tr["a2"][-1])
    return {
        "name": name,
        "w12": w12,
        "w21": w21,
        "D": D,
        "r1": r1,
        "r2": r2,
        "ratio": ratio,
        "Ke1_max": float(ke1.max()),
        "Ke2_max": float(ke2.max()),
        "Ke1_end_drive": float(ke1[drive][-1]) if drive.any() else float("nan"),
        "Ke2_end_drive": float(ke2[drive][-1]) if drive.any() else float("nan"),
        "da1": da1,
        "da2": da2,
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
        ax.plot(tr["t"], tr["V1"], lw=0.55, color=colors[name], alpha=0.9, label=f"V1 {name}")
    ax.axvspan(T_ON, T_OFF, color="0.9", zorder=0)
    ax.set_ylabel("V1 [mV]")
    ax.set_title("V1 under co-drive")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[0, 1]
    for name in ("neither", "syn", "diff", "both"):
        tr = traces[name]
        ax.plot(tr["t"], tr["V2"], lw=0.55, color=colors[name], alpha=0.9, label=f"V2 {name}")
    ax.axvspan(T_ON, T_OFF, color="0.9", zorder=0)
    ax.set_ylabel("V2 [mV]")
    ax.set_title("V2 under co-drive")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[1, 0]
    names = [r["name"] for r in rows]
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [r["r1"] for r in rows], 0.4, label="r1", color="C0")
    ax.bar(xs + 0.2, [r["r2"] for r in rows], 0.4, label="r2", color="C3")
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean soft-spike (drive window)")
    ax.set_title("Rates (train-sensitive window)")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.bar(xs - 0.2, [r["Ke1_max"] for r in rows], 0.4, label="Ke1_max", color="C1")
    ax.bar(xs + 0.2, [r["Ke2_max"] for r in rows], 0.4, label="Ke2_max", color="C2")
    ax.axhline(K_BATH, color="0.4", ls=":", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel(r"$K_e$ [mM]")
    ax.set_title(r"$K_e$ peaks (bus)")
    ax.legend(fontsize=7)

    fig.suptitle(
        rf"C′′ co-drive  $w={W_ON}$, $D={D_KE}$, $I$={I_DRIVE} "
        rf"({T_ON:.0f}–{T_OFF:.0f} ms), both cells",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "figures" / "cdoubleprime_co_drive.png"
    params = ke_params()
    I = make_shared_drive()

    print(
        f"C′′ co-drive: W_ON={W_ON}, D={D_KE}, I_drive={I_DRIVE} "
        f"({T_ON}-{T_OFF}), tau_syn={TAU_SYN}, V0=({V0_1},{V0_2})"
    )
    print("protocol: B1-style shared drive (NOT cue→probe); K_ATP/gNa off; ion Ke on")

    traces: dict[str, dict] = {}
    rows: list[dict] = []
    for name, (w12, w21), D in CONDITIONS:
        tr = simulate_two_cell_ke(
            I,
            I,
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
            f"{name:7s}: r=({m['r1']:.4f},{m['r2']:.4f}) ratio={m['ratio']:.3f} "
            f"Ke_max=({m['Ke1_max']:.2f},{m['Ke2_max']:.2f}) "
            f"Δa=({m['da1']:.3f},{m['da2']:.3f})"
        )

    plot_results(traces, rows, out)
    print(f"saved: {out}")

    by = {r["name"]: r for r in rows}
    n, s, d, b = by["neither"], by["syn"], by["diff"], by["both"]

    def rate_dist(a: dict, c: dict) -> float:
        return abs(a["r1"] - c["r1"]) + abs(a["r2"] - c["r2"])

    syn_vs_n = rate_dist(s, n)
    diff_vs_n = rate_dist(d, n)
    both_vs_s = rate_dist(b, s)
    both_vs_d = rate_dist(b, d)
    syn_ratio_shift = abs(s["ratio"] - n["ratio"])
    diff_ke = abs(d["Ke1_max"] - n["Ke1_max"]) + abs(d["Ke2_max"] - n["Ke2_max"])

    print("--- contrasts ---")
    print(
        f"  |syn−neither|_r={syn_vs_n:.4f} |Δratio|={syn_ratio_shift:.3f} "
        f"(train)"
    )
    print(
        f"  |diff−neither|_r={diff_vs_n:.4f} |ΔKe|_maxsum={diff_ke:.3f} "
        f"(bus)"
    )
    print(f"  |both−syn|_r={both_vs_s:.4f} |both−diff|_r={both_vs_d:.4f}")

    train_visible = syn_vs_n > 0.002 or syn_ratio_shift > 0.15
    bus_visible = diff_ke > 0.3 or diff_vs_n > 0.002
    if train_visible and both_vs_s < both_vs_d:
        shape = "syn-leaning"
    elif train_visible and both_vs_d < both_vs_s:
        shape = "diff-leaning"
    elif train_visible:
        shape = "mixed"
    else:
        shape = "unclear"

    if train_visible and bus_visible:
        verdict = "success"
        why = (
            f"co-drive shows train (syn≠neither) and bus (diff≠neither); "
            f"both is {shape}"
        )
    elif train_visible:
        verdict = "partial"
        why = f"train visible under co-drive; bus weak on rates; both is {shape}"
    elif bus_visible:
        verdict = "partial"
        why = "bus visible; train still weak even under co-drive"
    else:
        verdict = "null"
        why = "neither syn nor diff clearly shifts co-drive scores"
    print(f"VERDICT: {verdict} — {why}")
    print(
        "C′ contrast: cue→probe was diff-dominated; "
        "co-drive is meant to put the train on the scoreboard"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase C′′′ — short co-drive: score train + bus in one exam.

Run from the repo root::

    python scripts/17_ctripleprime_short_codrive.py

Same neither/syn/diff/both as C′/C′′, but shared I_drive with a **short**
window so K_e does not slam into the clip ceiling (C′′ failure mode).
Sweep drive lengths; pick one window where both syn and diff leave a mark.

Claim ceiling: short co-drive can put synaptic and K_e-diffusion contrasts
on the same scoreboard — not optimal hybrid communication / brain replay.
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

GAMMA_KE = 0.0005
TAU_K = 1000.0
K_BATH = 4.0
KE_CLIP = 20.0
D_KE = 0.005
W_ON = (-15.0, -3.0)
TAU_SYN = 8.0
I_DRIVE = 9.0
T_ON = 20.0
# drive widths to sweep (ms); all << C′′ 460 ms window
DRIVE_WIDTHS = (60.0, 80.0, 100.0, 120.0, 150.0, 200.0)
TAIL_MS = 80.0  # post-drive quiet for Ke plot

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


def make_shared_drive(t_off: float, t_end: float) -> jnp.ndarray:
    t = jnp.arange(0.0, t_end + DT / 2, DT)
    return I_ext_step(t, T_ON, t_off, I_DRIVE)


def summarize(
    tr: dict, name: str, w12: float, w21: float, D: float, t_off: float
) -> dict:
    t = np.asarray(tr["t"])
    s1 = np.asarray(tr["s1"])
    s2 = np.asarray(tr["s2"])
    ke1 = np.asarray(tr["K_e1"])
    ke2 = np.asarray(tr["K_e2"])
    drive = (t >= T_ON) & (t < t_off)
    r1 = float(s1[drive].mean())
    r2 = float(s2[drive].mean())
    ratio = max(r1, r2) / max(min(r1, r2), 1e-9)
    ke1_max = float(ke1.max())
    ke2_max = float(ke2.max())
    return {
        "name": name,
        "w12": w12,
        "w21": w21,
        "D": D,
        "r1": r1,
        "r2": r2,
        "ratio": ratio,
        "Ke1_max": ke1_max,
        "Ke2_max": ke2_max,
        "Ke_clipped": ke1_max >= KE_CLIP - 0.05 or ke2_max >= KE_CLIP - 0.05,
        "da1": float(tr["a1"][0] - tr["a1"][-1]),
        "da2": float(tr["a2"][0] - tr["a2"][-1]),
    }


def run_four(params: dict, t_off: float) -> tuple[dict[str, dict], list[dict]]:
    t_end = t_off + TAIL_MS
    I = make_shared_drive(t_off, t_end)
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
        m = summarize(tr, name, w12, w21, D, t_off)
        traces[name] = tr
        rows.append(m)
    return traces, rows


def score_window(rows: list[dict]) -> dict:
    by = {r["name"]: r for r in rows}
    n, s, d, b = by["neither"], by["syn"], by["diff"], by["both"]
    train_r = abs(s["r1"] - n["r1"]) + abs(s["r2"] - n["r2"])
    train_ratio = abs(s["ratio"] - n["ratio"])
    bus_ke = abs(d["Ke1_max"] - n["Ke1_max"]) + abs(d["Ke2_max"] - n["Ke2_max"])
    bus_r = abs(d["r1"] - n["r1"]) + abs(d["r2"] - n["r2"])
    clipped = any(r["Ke_clipped"] for r in rows)
    train_ok = (not clipped) and (train_r > 0.0015 or train_ratio > 0.08)
    bus_ok = (not clipped) and (bus_ke > 0.25 or bus_r > 0.0015)
    both_vs_s = abs(b["r1"] - s["r1"]) + abs(b["r2"] - s["r2"])
    both_vs_d = abs(b["r1"] - d["r1"]) + abs(b["r2"] - d["r2"])
    both_ke_vs_d = abs(b["Ke1_max"] - d["Ke1_max"]) + abs(b["Ke2_max"] - d["Ke2_max"])
    # both not fully collapsed: has train-like rate shift OR keeps bus Ke near diff
    both_mixed = (
        (train_ok and abs(b["ratio"] - n["ratio"]) > 0.05)
        or (bus_ok and bus_ke > 0.25 and both_ke_vs_d < 0.5 * bus_ke + 0.2)
    )
    return {
        "train_r": train_r,
        "train_ratio": train_ratio,
        "bus_ke": bus_ke,
        "bus_r": bus_r,
        "clipped": clipped,
        "train_ok": train_ok,
        "bus_ok": bus_ok,
        "both_mixed": both_mixed,
        "both_vs_s": both_vs_s,
        "both_vs_d": both_vs_d,
        "quality": (
            (2 if train_ok else 0)
            + (2 if bus_ok else 0)
            + (1 if both_mixed else 0)
            - (3 if clipped else 0)
        ),
    }


def plot_results(
    traces: dict[str, dict],
    rows: list[dict],
    t_off: float,
    width: float,
    out_path: Path,
) -> None:
    colors = {
        "neither": "0.55",
        "syn": "C0",
        "diff": "C2",
        "both": "C3",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    ax = axes[0, 0]
    for name in ("neither", "syn", "diff", "both"):
        tr = traces[name]
        ax.plot(tr["t"], tr["V1"], lw=0.6, color=colors[name], label=f"V1 {name}")
    ax.axvspan(T_ON, t_off, color="0.9", zorder=0)
    ax.set_ylabel("V1 [mV]")
    ax.set_title(f"V1 (drive {width:.0f} ms)")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[0, 1]
    for name in ("neither", "syn", "diff", "both"):
        tr = traces[name]
        ax.plot(tr["t"], tr["K_e1"], lw=1.0, color=colors[name], label=f"Ke1 {name}")
        ax.plot(
            tr["t"],
            tr["K_e2"],
            lw=0.9,
            ls="--",
            color=colors[name],
            alpha=0.85,
            label=f"Ke2 {name}",
        )
    ax.axhline(KE_CLIP, color="r", ls=":", lw=0.9, label="clip")
    ax.axhline(K_BATH, color="0.4", ls=":", lw=0.7)
    ax.axvspan(T_ON, t_off, color="0.9", zorder=0)
    ax.set_ylabel(r"$K_e$ [mM]")
    ax.set_title(r"$K_e$ (solid cell1 / dashed cell2)")
    ax.legend(fontsize=5, ncol=2)

    ax = axes[1, 0]
    names = [r["name"] for r in rows]
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [r["r1"] for r in rows], 0.4, label="r1", color="C0")
    ax.bar(xs + 0.2, [r["r2"] for r in rows], 0.4, label="r2", color="C3")
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean soft-spike")
    ax.set_title("Rates (train)")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.bar(xs - 0.2, [r["Ke1_max"] for r in rows], 0.4, label="Ke1_max", color="C1")
    ax.bar(xs + 0.2, [r["Ke2_max"] for r in rows], 0.4, label="Ke2_max", color="C2")
    ax.axhline(KE_CLIP, color="r", ls=":", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel(r"$K_e$ max [mM]")
    ax.set_title("Ke peaks (bus; below clip?)")
    ax.legend(fontsize=7)

    fig.suptitle(
        rf"C′′′ short co-drive  width={width:.0f} ms  "
        rf"$w={W_ON}$, $D={D_KE}$, $I$={I_DRIVE}",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "figures" / "ctripleprime_short_codrive.png"
    params = ke_params()

    print(
        f"C′′′ short co-drive: W_ON={W_ON}, D={D_KE}, I={I_DRIVE}, "
        f"widths={DRIVE_WIDTHS}, Ke_clip={KE_CLIP}"
    )
    print("goal: train (syn≠neither) AND bus (diff≠neither, no Ke clip)")

    candidates: list[dict] = []
    for width in DRIVE_WIDTHS:
        t_off = T_ON + width
        traces, rows = run_four(params, t_off)
        sc = score_window(rows)
        by = {r["name"]: r for r in rows}
        print(
            f"width={width:5.0f}: "
            f"syn ratio={by['syn']['ratio']:.3f} "
            f"Ke_max neither={by['neither']['Ke1_max']:.2f} "
            f"diff={by['diff']['Ke1_max']:.2f}/{by['diff']['Ke2_max']:.2f} "
            f"clip={sc['clipped']} train_ok={sc['train_ok']} "
            f"bus_ok={sc['bus_ok']} Q={sc['quality']}"
        )
        for r in rows:
            print(
                f"  {r['name']:7s}: r=({r['r1']:.4f},{r['r2']:.4f}) "
                f"ratio={r['ratio']:.3f} Ke=({r['Ke1_max']:.2f},{r['Ke2_max']:.2f}) "
                f"clip={r['Ke_clipped']}"
            )
        candidates.append(
            {
                "width": width,
                "t_off": t_off,
                "traces": traces,
                "rows": rows,
                "score": sc,
            }
        )

    # Prefer train+bus; else strongest unclipped train (ratio shift)
    both = [c for c in candidates if c["score"]["train_ok"] and c["score"]["bus_ok"]]
    if both:
        best = max(
            both,
            key=lambda c: (c["score"]["quality"], c["score"]["train_ratio"]),
        )
    else:
        unclip = [c for c in candidates if not c["score"]["clipped"]]
        pool = unclip if unclip else candidates
        best = max(
            pool,
            key=lambda c: (
                c["score"]["quality"],
                c["score"]["train_ratio"],
                c["score"]["bus_ke"],
            ),
        )

    width = best["width"]
    t_off = best["t_off"]
    rows = best["rows"]
    traces = best["traces"]
    sc = best["score"]
    by = {r["name"]: r for r in rows}

    plot_results(traces, rows, t_off, width, out)
    print(f"saved: {out} (chosen width={width:.0f} ms, t_off={t_off:.0f})")

    # Note: under identical co-drive, both cells build the same Ke → D has no
    # gradient, so diff≈neither is expected (bus needs asymmetry; see C′).
    print(
        f"note: |Ke_diff−Ke_neither|="
        f"{abs(by['diff']['Ke1_max']-by['neither']['Ke1_max']):.3f} "
        f"(symmetric drive → no Ke gradient for D)"
    )

    if sc["train_ok"] and sc["bus_ok"]:
        verdict = "both_visible"
        why = "short co-drive: syn≠neither and diff≠neither without Ke clip"
    elif sc["train_ok"]:
        verdict = "train_only"
        why = (
            "train visible without Ke clip; bus null under symmetric co-drive "
            "(identical Ke → D inert; bus needs asymmetry as in C′)"
        )
    elif sc["bus_ok"]:
        verdict = "bus_only"
        why = "bus visible; train weak"
    else:
        verdict = "null"
        why = "neither channel clearly scored in tested widths"
    print(f"VERDICT: {verdict} — {why}")
    print(
        "C′/C′′/C′′′: cue→probe=bus; long co-drive=train+Ke clip; "
        "short co-drive=train without clip, bus still needs asymmetry"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase C′′′ asym — slightly asymmetric short co-drive (train + bus?).

Run from the repo root::

    python scripts/18_casym_short_codrive.py

Same neither/syn/diff/both scoreboard as ``17_ctripleprime_short_codrive.py``,
but shared time window with **asymmetric amplitudes**
``I1 = I_DRIVE``, ``I2 = I_DRIVE * (1 - eps)``.

Hypothesis: tiny drive asymmetry seeds a K_e gradient so ``bus_ok`` can fire
in a short window while ``train_ok`` remains. ``eps=0`` must reproduce C′′′
bus-null.

Claim ceiling: short asymmetric co-drive *can* put both contrasts on one
scoreboard — not optimal hybrid / brain dual-line / always-both.
Final story label is Ask-side; this script only prints numeric candidates.
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
# primary: fix width, sweep eps (0 included = C′′′ regression)
PRIMARY_WIDTH = 150.0
EPS_SWEEP = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
# fallback if primary never gets train+bus unclipped
FALLBACK_WIDTHS = (100.0, 120.0, 150.0, 200.0)
TAIL_MS = 80.0

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


def make_asymmetric_drives(
    t_off: float, t_end: float, eps: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    t = jnp.arange(0.0, t_end + DT / 2, DT)
    i1 = I_ext_step(t, T_ON, t_off, I_DRIVE)
    i2 = I_ext_step(t, T_ON, t_off, I_DRIVE * (1.0 - eps))
    return i1, i2


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
        "dKe_max": abs(ke1_max - ke2_max),
        "Ke_clipped": ke1_max >= KE_CLIP - 0.05 or ke2_max >= KE_CLIP - 0.05,
        "da1": float(tr["a1"][0] - tr["a1"][-1]),
        "da2": float(tr["a2"][0] - tr["a2"][-1]),
    }


def run_four(
    params: dict, t_off: float, eps: float
) -> tuple[dict[str, dict], list[dict]]:
    t_end = t_off + TAIL_MS
    I1, I2 = make_asymmetric_drives(t_off, t_end, eps)
    traces: dict[str, dict] = {}
    rows: list[dict] = []
    for name, (w12, w21), D in CONDITIONS:
        tr = simulate_two_cell_ke(
            I1,
            I2,
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
    """Same train/bus rules as scripts/17 (transplanted)."""
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
    both_mixed = (
        (train_ok and abs(b["ratio"] - n["ratio"]) > 0.05)
        or (bus_ok and bus_ke > 0.25 and both_ke_vs_d < 0.5 * bus_ke + 0.2)
    )
    dke_diff = d["dKe_max"]
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
        "dKe_diff": dke_diff,
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
    eps: float,
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
    ax.set_title(f"V1 (width={width:.0f} ms, eps={eps:.2f})")
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
        rf"C′′′ asym short co-drive  width={width:.0f} ms  eps={eps:.2f}  "
        rf"$I_2=I(1-\epsilon)$  $w={W_ON}$, $D={D_KE}$, $I$={I_DRIVE}",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def evaluate(
    params: dict, width: float, eps: float
) -> dict:
    t_off = T_ON + width
    traces, rows = run_four(params, t_off, eps)
    sc = score_window(rows)
    by = {r["name"]: r for r in rows}
    return {
        "width": width,
        "eps": eps,
        "t_off": t_off,
        "traces": traces,
        "rows": rows,
        "score": sc,
        "by": by,
    }


def pick_best(candidates: list[dict]) -> dict:
    """Prefer: unclipped + train_ok + bus_ok, then **smallest eps**, then Q.

    Documented selection rule for STATUS / Ask (not a uniqueness claim).
    """
    dual = [
        c
        for c in candidates
        if (not c["score"]["clipped"])
        and c["score"]["train_ok"]
        and c["score"]["bus_ok"]
    ]
    if dual:
        return min(
            dual,
            key=lambda c: (
                c["eps"],
                -c["score"]["quality"],
                -c["score"]["bus_ke"],
                -c["score"]["train_ratio"],
            ),
        )
    # partial: unclipped with either channel
    unclip = [c for c in candidates if not c["score"]["clipped"]]
    pool = unclip if unclip else candidates
    return max(
        pool,
        key=lambda c: (
            c["score"]["quality"],
            c["score"]["train_ok"],
            c["score"]["bus_ok"],
            c["score"]["bus_ke"],
            c["score"]["train_ratio"],
            # among equals prefer smaller eps (milder asymmetry)
            -c["eps"],
        ),
    )


def print_candidate(c: dict) -> None:
    sc = c["score"]
    by = c["by"]
    print(
        f"eps={c['eps']:.2f} width={c['width']:.0f}: "
        f"train_ok={sc['train_ok']} bus_ok={sc['bus_ok']} "
        f"clip={sc['clipped']} both_mixed={sc['both_mixed']} "
        f"Q={sc['quality']} "
        f"bus_ke={sc['bus_ke']:.3f} dKe_diff={sc['dKe_diff']:.3f} "
        f"|Ke_diff-neither|="
        f"{abs(by['diff']['Ke1_max']-by['neither']['Ke1_max']) + abs(by['diff']['Ke2_max']-by['neither']['Ke2_max']):.3f}"
    )
    for r in c["rows"]:
        print(
            f"  {r['name']:7s}: r=({r['r1']:.4f},{r['r2']:.4f}) "
            f"ratio={r['ratio']:.3f} Ke=({r['Ke1_max']:.2f},{r['Ke2_max']:.2f}) "
            f"|dKe|={r['dKe_max']:.3f} clip={r['Ke_clipped']}"
        )


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "figures" / "casym_short_codrive.png"
    params = ke_params()

    print(
        f"C′′′ asym: W_ON={W_ON}, D={D_KE}, I={I_DRIVE}, "
        f"primary_width={PRIMARY_WIDTH}, eps={EPS_SWEEP}, Ke_clip={KE_CLIP}"
    )
    print(
        "selection: prefer unclipped & train_ok & bus_ok at **minimal eps**; "
        "eps=0 must show bus_ok=False (C′′′ regression)"
    )

    candidates: list[dict] = []
    print("--- primary: width fixed, eps sweep ---")
    for eps in EPS_SWEEP:
        c = evaluate(params, PRIMARY_WIDTH, float(eps))
        print_candidate(c)
        candidates.append(c)

    best_primary = pick_best(candidates)
    need_fallback = not (
        (not best_primary["score"]["clipped"])
        and best_primary["score"]["train_ok"]
        and best_primary["score"]["bus_ok"]
    )

    if need_fallback:
        print("--- fallback: small width × eps grid ---")
        for width in FALLBACK_WIDTHS:
            for eps in EPS_SWEEP:
                if width == PRIMARY_WIDTH and eps in EPS_SWEEP:
                    # already evaluated
                    continue
                c = evaluate(params, float(width), float(eps))
                print_candidate(c)
                candidates.append(c)

    best = pick_best(candidates)
    # always keep eps=0 @ primary for STATUS contrast
    eps0 = next(
        c
        for c in candidates
        if c["eps"] == 0.0 and c["width"] == PRIMARY_WIDTH
    )

    plot_results(
        best["traces"],
        best["rows"],
        best["t_off"],
        best["width"],
        best["eps"],
        out,
    )
    print(f"saved: {out}")
    print(
        f"CHOSEN: eps={best['eps']:.2f} width={best['width']:.0f} "
        f"(rule: dual-ok min-eps, else max Q among unclipped)"
    )
    print(
        f"eps=0 control @ width={PRIMARY_WIDTH:.0f}: "
        f"train_ok={eps0['score']['train_ok']} "
        f"bus_ok={eps0['score']['bus_ok']} "
        f"clip={eps0['score']['clipped']} Q={eps0['score']['quality']}"
    )

    sc = best["score"]
    if sc["train_ok"] and sc["bus_ok"] and not sc["clipped"]:
        verdict = "success_candidate"
        why = "unclipped train_ok+bus_ok at eps>0; eps=0 bus null retained"
    elif sc["train_ok"] and not sc["clipped"]:
        verdict = "partial_train"
        why = "train visible; bus still weak / needs larger asymmetry or different window"
    elif sc["bus_ok"] and not sc["clipped"]:
        verdict = "partial_bus"
        why = "bus visible; train weak or broken by asymmetry"
    else:
        verdict = "null_or_clipped"
        why = "no clean dual scoreboard in tested (eps,width)"
    print(f"SELF_VERDICT (Ask decides final label): {verdict} — {why}")
    print(
        "claim ceiling: not optimal hybrid / not brain dual-line / "
        "not 'always both visible'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

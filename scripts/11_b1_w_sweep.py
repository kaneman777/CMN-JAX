"""Phase B1 reinforcement — sweep |w| of structural mutual inhibition.

Run from the repo root::

    python scripts/11_b1_w_sweep.py

Same drive / τ_syn / V0 seed as ``10_b1_two_cell_asymm.py``. Scales the
reference weights (-80, -10) by {1, 0.5, 0.25, 0.125, ...}. No a_max
shoulder-takeover (②). Saves ``figures/b1_w_sweep.png``.
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

# Match B1 main experiment
T_END = 500.0
T_ON = 20.0
T_OFF = 480.0
I_DRIVE = 9.0
TAU_SYN = 8.0
V0_1 = -65.0
V0_2 = -64.2
A0 = 1.0

W_REF = (-80.0, -10.0)  # B1 E1 baseline
# scale factors along ~8:1 ray (1 → 0.125)
SCALES = (1.0, 0.75, 0.5, 0.375, 0.25, 0.125)
# Extra pairs off the pure 8:1 ray (same protocol) to find healthier ① band
EXTRA_W = (
    (-50.0, -10.0),
    (-40.0, -10.0),
    (-20.0, -4.0),
    (-16.0, -4.0),
    (-15.0, -3.0),
)


def make_drive() -> jnp.ndarray:
    t = jnp.arange(0.0, T_END + DT / 2, DT)
    return I_ext_step(t, T_ON, T_OFF, I_DRIVE)


def eval_pair(I_drive: jnp.ndarray, w12: float, w21: float) -> dict[str, float]:
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
    t = np.asarray(tr["t"])
    s1 = np.asarray(tr["s1"])
    s2 = np.asarray(tr["s2"])
    v1 = np.asarray(tr["V1"])
    v2 = np.asarray(tr["V2"])
    mask = (t >= T_ON) & (t < T_OFF)
    r1 = float(s1[mask].mean())
    r2 = float(s2[mask].mean())
    da1 = float(tr["a1"][0] - tr["a1"][-1])
    da2 = float(tr["a2"][0] - tr["a2"][-1])
    ratio = max(r1, r2) / max(min(r1, r2), 1e-9)
    return {
        "w12": w12,
        "w21": w21,
        "scale": abs(w12 / W_REF[0]),
        "r1": r1,
        "r2": r2,
        "ratio": ratio,
        "da1": da1,
        "da2": da2,
        "V1_min": float(v1.min()),
        "V2_min": float(v2.min()),
        "ok_15": ratio >= 1.5,
    }


def plot_sweep(rows: list[dict], out_path: Path) -> None:
    ray = [r for r in rows if r.get("family") == "ray" and r["scale"] > 0]
    extra = [r for r in rows if r.get("family") == "extra"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)

    ax = axes[0]
    if ray:
        ax.plot(
            [r["scale"] for r in ray],
            [r["ratio"] for r in ray],
            "o-",
            color="C0",
            lw=1.5,
            label="8:1 ray",
        )
    for r in extra:
        ax.scatter(
            [abs(r["w12"] / W_REF[0])],
            [r["ratio"]],
            marker="D",
            color="C3",
            zorder=3,
            label="extra" if r is extra[0] else None,
        )
    ax.axhline(1.5, color="0.3", ls="--", lw=1.0, label="ratio=1.5")
    ax.invert_xaxis()
    ax.set_xlabel(r"scale $\approx|w_{12}|/80$")
    ax.set_ylabel("rate ratio")
    ax.set_title("Asymmetry vs |w|")
    ax.legend(fontsize=7)

    ax = axes[1]
    if ray:
        ax.plot([r["scale"] for r in ray], [r["r1"] for r in ray], "o-", color="C0", label="r1 ray")
        ax.plot([r["scale"] for r in ray], [r["r2"] for r in ray], "s--", color="C3", label="r2 ray")
    ax.invert_xaxis()
    ax.set_xlabel(r"scale $\approx|w_{12}|/80$")
    ax.set_ylabel("mean soft-spike")
    ax.set_title("Rates (8:1 ray)")
    ax.legend(fontsize=7)

    ax = axes[2]
    if ray:
        ax.plot(
            [r["scale"] for r in ray],
            [min(r["V1_min"], r["V2_min"]) for r in ray],
            "o-",
            color="C2",
            label="ray",
        )
    for r in extra:
        ax.scatter(
            [abs(r["w12"] / W_REF[0])],
            [min(r["V1_min"], r["V2_min"])],
            marker="D",
            color="C3",
            zorder=3,
            label="extra" if r is extra[0] else None,
        )
    ax.axhline(-100.0, color="0.5", ls=":", lw=0.8, label="-100 mV")
    ax.invert_xaxis()
    ax.set_xlabel(r"scale $\approx|w_{12}|/80$")
    ax.set_ylabel(r"$\min(V)$ [mV]")
    ax.set_title("Deepest hyperpolarization")
    ax.legend(fontsize=7)

    fig.suptitle(
        rf"B1 $|w|$ sweep: ref $w={W_REF}$, $I_{{\mathrm{{drive}}}}={I_DRIVE}$ "
        rf"(no ② $a_{{\max}}$)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path.resolve()}")


def pick_recommend(rows: list[dict]) -> dict | None:
    """Prefer ratio≥1.5 with V_min > -100; else > -120; minimize |w12|."""
    def score(r: dict) -> tuple:
        vmin = min(r["V1_min"], r["V2_min"])
        healthy = 0 if vmin > -100 else (1 if vmin > -120 else 2)
        return (healthy, abs(r["w12"]))

    cands = [r for r in rows if r["scale"] > 0 and r["ok_15"]]
    if not cands:
        return None
    return min(cands, key=score)


def main() -> int:
    print(
        f"B1 |w| sweep: ref={W_REF}, scales={SCALES}, "
        f"I_drive={I_DRIVE}, tau_syn={TAU_SYN}, V0=({V0_1},{V0_2}) — NO ②"
    )
    I_drive = make_drive()

    rows: list[dict] = []
    off = eval_pair(I_drive, 0.0, 0.0)
    off["scale"] = 0.0
    off["family"] = "off"
    rows.append(off)
    print(
        f"OFF: r=({off['r1']:.4f},{off['r2']:.4f}) ratio={off['ratio']:.3f} "
        f"Vmin=({off['V1_min']:.1f},{off['V2_min']:.1f})"
    )

    print("--- 8:1 ray ---")
    for s in SCALES:
        w12, w21 = W_REF[0] * s, W_REF[1] * s
        m = eval_pair(I_drive, w12, w21)
        m["family"] = "ray"
        rows.append(m)
        vmin = min(m["V1_min"], m["V2_min"])
        note = "  [deep]" if vmin < -100 else ("  [mild]" if vmin < -80 else "")
        print(
            f"scale={s:.3f} w=({w12:.2f},{w21:.2f}): "
            f"r=({m['r1']:.4f},{m['r2']:.4f}) ratio={m['ratio']:.3f} "
            f"Δa=({m['da1']:.3f},{m['da2']:.3f}) "
            f"Vmin={vmin:.1f} ok15={m['ok_15']}{note}"
        )

    print("--- extra pairs ---")
    for w12, w21 in EXTRA_W:
        m = eval_pair(I_drive, w12, w21)
        m["family"] = "extra"
        rows.append(m)
        vmin = min(m["V1_min"], m["V2_min"])
        note = "  [deep]" if vmin < -100 else ("  [mild]" if vmin < -80 else "")
        print(
            f"w=({w12:.2f},{w21:.2f}): "
            f"r=({m['r1']:.4f},{m['r2']:.4f}) ratio={m['ratio']:.3f} "
            f"Δa=({m['da1']:.3f},{m['da2']:.3f}) "
            f"Vmin={vmin:.1f} ok15={m['ok_15']}{note}"
        )

    out = Path("figures") / "b1_w_sweep.png"
    plot_sweep(rows, out)

    rec = pick_recommend(rows)
    print("--- recommend for future ② (not run here) ---")
    if rec is None:
        print("  no pair with ratio≥1.5; keep ref or retune")
    else:
        vmin = min(rec["V1_min"], rec["V2_min"])
        print(
            f"  recommended w=({rec['w12']:.2f},{rec['w21']:.2f}) "
            f"ratio={rec['ratio']:.3f} Vmin={vmin:.1f} "
            f"(family={rec.get('family')})"
        )
        print(
            "  reason: weakest |w12| among ratio≥1.5 with preferably Vmin>-100 "
            "(8:1 ray alone loses ① when scaled down; ref has deep hyperpol)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

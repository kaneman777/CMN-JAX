"""Phase 3c — RQ3 minimal: post-learning ATP supply stress (a_max sweep).

Run from the repo root::

    python scripts/06_rq3_atp_stress.py

C0: obtain selective w* with a short λ=0 retrain (Phase-3a task; gate off).
C1: freeze w*, enable optional ATP→gNa gate, sweep a_max, compare strong vs
weak/noise responses. Saves ``figures/rq3_atp_stress.png``.

Claim ceiling: supply drop simplifies / weakens responses first — not a
phase-transition proof of essential-feature extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from jnpnp.models.point_sim import DT, merge_params, soft_spike, simulate_driven

# ---------------------------------------------------------------------------
# Match Phase 3a stimulus geometry
# ---------------------------------------------------------------------------
T_END = 250.0
T_MID = 0.5 * T_END
PULSE_AMP = 15.0
PULSE_WIDTH = 6.0
PULSE_PERIOD = 12.0
V_TH = 0.0
K_SOFT = 0.5
S_HI = 0.10
S_LO = 0.03

# Frozen w* fallback (Phase 3a λ=0 STATUS); optional short retrain below.
W_STAR_DEFAULT = np.array([0.5315, 0.0918], dtype=np.float64)
RETRAIN_LAMBDA0 = True
RETRAIN_STEPS = 40
RETRAIN_LR = 0.05
W0 = jnp.array([0.8, 0.8], dtype=jnp.float64)

A_MAX_GRID = (1.0, 0.7, 0.4, 0.2)
# Near-threshold noise drive (N-half pulses); independent of w2*.
WEAK_NOISE_GAIN = 0.45
# Phenomenological ATP→Na excitability for RQ3 preview (0 = biophysical-only).
ATP_GNA_FLOOR = 0.35


def make_time_grid(t_end: float = T_END, dt: float = DT) -> jnp.ndarray:
    n = int(round(t_end / dt))
    return jnp.arange(n + 1, dtype=jnp.float64) * dt


def pulse_train(t_local: jnp.ndarray, amp: float = PULSE_AMP) -> jnp.ndarray:
    phase = jnp.mod(t_local, PULSE_PERIOD)
    return jnp.where(phase < PULSE_WIDTH, amp, 0.0)


def build_envelopes(t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    in_S = t < T_MID
    in_N = ~in_S
    I_sig = jnp.where(in_S, pulse_train(jnp.where(in_S, t, 0.0)), 0.0)
    I_noise = jnp.where(in_N, pulse_train(jnp.where(in_N, t - T_MID, 0.0)), 0.0)
    return I_sig, I_noise


def train_w_star() -> np.ndarray:
    """Short λ=0 Optax train (Phase 3a loss; no ATP–gNa gate)."""
    t = make_time_grid()
    I_sig, I_noise = build_envelopes(t)
    mask_S = (t < T_MID).astype(jnp.float64)
    mask_N = (t >= T_MID).astype(jnp.float64)

    def loss_fn(w):
        I_ext = w[0] * I_sig + w[1] * I_noise
        traj = simulate_driven(I_ext, fix_a=False, dt=DT, t_end=T_END)
        s = soft_spike(traj["V"], V_th=V_TH, k=K_SOFT)
        mean_S = jnp.sum(s * mask_S) / jnp.sum(mask_S)
        mean_N = jnp.sum(s * mask_N) / jnp.sum(mask_N)
        return (mean_S - S_HI) ** 2 + (mean_N - S_LO) ** 2

    w = jnp.array(W0)
    opt = optax.adam(RETRAIN_LR)
    state = opt.init(w)
    vag = jax.jit(jax.value_and_grad(loss_fn))
    print(f"C0: retraining w* (λ=0, gate off, {RETRAIN_STEPS} steps) ...")
    for step in range(RETRAIN_STEPS):
        loss, g = vag(w)
        if step % 10 == 0 or step == RETRAIN_STEPS - 1:
            print(f"  step {step:02d}: L={float(loss):.5g} w=({float(w[0]):.3f},{float(w[1]):.3f})")
        updates, state = opt.update(g, state, w)
        w = jnp.clip(optax.apply_updates(w, updates), 0.0, 2.5)
    return np.array([float(w[0]), float(w[1])], dtype=np.float64)


def eval_condition(
    I_ext: jnp.ndarray,
    a_max: float,
    window: str,
    *,
    atp_gNa_floor: float = ATP_GNA_FLOOR,
) -> dict:
    """window: 'S' | 'N' | 'all' for mean soft-spike period."""
    params = merge_params(a_max=a_max, atp_gNa_floor=atp_gNa_floor)
    traj = simulate_driven(I_ext, fix_a=False, dt=DT, t_end=T_END, params=params)
    t = np.asarray(traj["t"])
    s = np.asarray(soft_spike(traj["V"], V_th=V_TH, k=K_SOFT))
    if window == "S":
        mask = t < T_MID
    elif window == "N":
        mask = t >= T_MID
    else:
        mask = np.ones_like(t, dtype=bool)
    a = np.asarray(traj["a"])
    Ip = np.asarray(traj["I_pump"])
    return {
        "mean_s": float(s[mask].mean()),
        "mean_a": float(a.mean()),
        "delta_a": float(a[0] - a[-1]),
        "L_met": float(np.maximum(Ip, 0.0).mean()),
        "V": np.asarray(traj["V"]),
        "t": t,
        "a": a,
    }


def main() -> int:
    if RETRAIN_LAMBDA0:
        w_star = train_w_star()
    else:
        w_star = W_STAR_DEFAULT.copy()
        print(f"C0: using embedded w* = {w_star}")

    t = make_time_grid()
    I_sig, I_noise = build_envelopes(t)
    I_strong = w_star[0] * I_sig
    I_weak = WEAK_NOISE_GAIN * I_noise

    print("--- C1: a_max sweep (w frozen, ATP→gNa gate on) ---")
    print(f"w* = ({w_star[0]:.4f}, {w_star[1]:.4f})")
    print(f"a_max grid = {A_MAX_GRID}")
    print(f"atp_gNa_floor = {ATP_GNA_FLOOR}, weak_gain = {WEAK_NOISE_GAIN}")

    rows = []
    traces = {}
    for a_max in A_MAX_GRID:
        strong = eval_condition(I_strong, a_max, "S")
        weak = eval_condition(I_weak, a_max, "N")
        row = {
            "a_max": a_max,
            "strong_s": strong["mean_s"],
            "weak_s": weak["mean_s"],
            "strong_mean_a": strong["mean_a"],
            "weak_mean_a": weak["mean_a"],
            "strong_delta_a": strong["delta_a"],
            "weak_delta_a": weak["delta_a"],
            "strong_Lmet": strong["L_met"],
            "weak_Lmet": weak["L_met"],
        }
        rows.append(row)
        traces[a_max] = {"strong": strong, "weak": weak}
        print(
            f"a_max={a_max:.1f}: strong_s={row['strong_s']:.4f} weak_s={row['weak_s']:.4f} "
            f"mean_a_s={row['strong_mean_a']:.3f} Δa_s={row['strong_delta_a']:.3f} "
            f"Lmet_s={row['strong_Lmet']:.4f} Lmet_w={row['weak_Lmet']:.4f}"
        )

    # Biophysical-only null check at extremes (gate off)
    print("--- biophysical-only null (atp_gNa_floor=0) ---")
    for a_max in (1.0, 0.2):
        sb = eval_condition(I_strong, a_max, "S", atp_gNa_floor=0.0)
        wb = eval_condition(I_weak, a_max, "N", atp_gNa_floor=0.0)
        print(
            f"  gate=0 a_max={a_max:.1f}: strong_s={sb['mean_s']:.4f} weak_s={wb['mean_s']:.4f}"
        )

    out_path = Path("figures") / "rq3_atp_stress.png"
    plot_results(rows, traces, w_star, out_path)

    s0, s_lo = rows[0]["strong_s"], rows[-1]["strong_s"]
    w0, w_lo = rows[0]["weak_s"], rows[-1]["weak_s"]
    overall_drop = (s_lo < 0.9 * s0) or (w_lo < 0.9 * w0)
    rel_s = 1.0 - s_lo / max(s0, 1e-9)
    rel_w = 1.0 - w_lo / max(w0, 1e-9)
    weak_first = (rel_w > rel_s + 0.05) or (w_lo < 0.5 * w0 and s_lo > 0.5 * s0)
    strong_survives = s_lo > 0.02
    print("--- summary ---")
    print(
        f"overall response drop: {overall_drop} "
        f"(strong {s0:.4f}→{s_lo:.4f}, weak {w0:.4f}→{w_lo:.4f})"
    )
    print(f"weak falls more / first: {weak_first} (rel_drop strong={rel_s:.3f} weak={rel_w:.3f})")
    print(f"strong not fully silenced: {strong_survives}")
    print(f"saved: {out_path.resolve()}")
    if overall_drop and weak_first and strong_survives:
        print("OK: ATP-stress coarsening preview (not a phase-transition claim)")
    elif overall_drop:
        print("PARTIAL: response drops; weak-first or strong-survive incomplete — STATUS")
    else:
        print("WARN: preview weak — record honestly in STATUS")
    return 0


def plot_results(rows: list[dict], traces: dict, w_star: np.ndarray, out_path: Path) -> None:
    a_vals = [r["a_max"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(a_vals, [r["strong_s"] for r in rows], "o-", color="C0", lw=1.5, label="strong (S)")
    ax.plot(a_vals, [r["weak_s"] for r in rows], "s--", color="C3", lw=1.5, label="weak/noise (N)")
    ax.set_xlabel(r"$a_{\max}$")
    ax.set_ylabel("mean soft-spike")
    ax.invert_xaxis()
    ax.set_title("Response vs ATP supply")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(a_vals, [r["strong_mean_a"] for r in rows], "o-", color="C0", label="strong mean(a)")
    ax.plot(a_vals, [r["weak_mean_a"] for r in rows], "s--", color="C3", label="weak mean(a)")
    ax.set_xlabel(r"$a_{\max}$")
    ax.set_ylabel(r"mean $a$")
    ax.invert_xaxis()
    ax.set_title("ATP level")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for a_max, style in ((1.0, "-"), (0.4, "--"), (0.2, ":")):
        if a_max not in traces:
            continue
        tr = traces[a_max]["strong"]
        ax.plot(tr["t"], tr["V"], lw=0.8, ls=style, label=rf"strong $a_{{\max}}$={a_max}")
    ax.axvline(T_MID, color="0.5", ls=":", lw=0.7)
    ax.set_ylabel("V [mV]")
    ax.set_xlabel("t [ms]")
    ax.set_title("Strong-signal voltage")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    for a_max, style in ((1.0, "-"), (0.4, "--"), (0.2, ":")):
        if a_max not in traces:
            continue
        tr = traces[a_max]["weak"]
        ax.plot(tr["t"], tr["V"], lw=0.8, ls=style, label=rf"weak $a_{{\max}}$={a_max}")
    ax.axvline(T_MID, color="0.5", ls=":", lw=0.7)
    ax.set_ylabel("V [mV]")
    ax.set_xlabel("t [ms]")
    ax.set_title("Weak/noise voltage")
    ax.legend(fontsize=7)

    fig.suptitle(
        rf"Phase 3c RQ3: frozen $w^*=({w_star[0]:.3f},{w_star[1]:.3f})$, "
        rf"$a_{{\max}}$ sweep (ATP→gNa floor={ATP_GNA_FLOOR})",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

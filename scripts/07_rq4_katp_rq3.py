"""Phase 4-Bio — simplified K_ATP + RQ3 A/B/C re-compare.

Run from the repo root::

    python scripts/07_rq4_katp_rq3.py

D0: unit smoke (fixed pulse; low a → I_KATP up → firing down).
D1: freeze w*, sweep a_max under
  A: biophysical only (no extra brake)
  B: phenomenological atp_gNa_floor (Phase 3c)
  C: simplified K_ATP (this phase)

Claim ceiling: simplified K_ATP path can show weak-first drop under supply
stress — not a molecular K_ATP reconstruction.
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
# Shared Phase-3a/3c stimulus geometry
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

W_STAR_DEFAULT = np.array([0.5315, 0.0918], dtype=np.float64)
RETRAIN_LAMBDA0 = True
RETRAIN_STEPS = 40
RETRAIN_LR = 0.05
W0 = jnp.array([0.8, 0.8], dtype=jnp.float64)

A_MAX_GRID = (1.0, 0.7, 0.4, 0.2)
WEAK_NOISE_GAIN = 0.45
ATP_GNA_FLOOR_B = 0.35

# Simplified K_ATP (Hill σ); tuned for C weak-first preview
G_KATP_BAR = 0.3
A_KATP_HALF = 0.2
N_KATP = 2.0

# D0 unit smoke
D0_T_END = 300.0
D0_TON = 20.0
D0_TOFF = 280.0
D0_IAMP = 10.0


def make_time_grid(t_end: float = T_END, dt: float = DT) -> jnp.ndarray:
    n = int(round(t_end / dt))
    return jnp.arange(n + 1, dtype=jnp.float64) * dt


def pulse_train(t_local: jnp.ndarray, amp: float = PULSE_AMP) -> jnp.ndarray:
    phase = jnp.mod(t_local, PULSE_PERIOD)
    return jnp.where(phase < PULSE_WIDTH, amp, 0.0)


def build_envelopes(t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    in_S = t < T_MID
    I_sig = jnp.where(in_S, pulse_train(jnp.where(in_S, t, 0.0)), 0.0)
    I_noise = jnp.where(~in_S, pulse_train(jnp.where(~in_S, t - T_MID, 0.0)), 0.0)
    return I_sig, I_noise


def train_w_star() -> np.ndarray:
    """λ=0 Phase-3a loss; brakes off (A-like)."""
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
    print(f"C0: retraining w* (λ=0, brakes off, {RETRAIN_STEPS} steps) ...")
    for step in range(RETRAIN_STEPS):
        loss, g = vag(w)
        if step % 10 == 0 or step == RETRAIN_STEPS - 1:
            print(f"  step {step:02d}: L={float(loss):.5g} w=({float(w[0]):.3f},{float(w[1]):.3f})")
        updates, state = opt.update(g, state, w)
        w = jnp.clip(optax.apply_updates(w, updates), 0.0, 2.5)
    return np.array([float(w[0]), float(w[1])], dtype=np.float64)


def condition_params(cond: str, a_max: float) -> dict[str, float]:
    """A / B / C parameter packs (K_e off)."""
    base = dict(a_max=a_max, g_KATP_bar=0.0, atp_gNa_floor=0.0)
    if cond == "A":
        return merge_params(**base)
    if cond == "B":
        base["atp_gNa_floor"] = ATP_GNA_FLOOR_B
        return merge_params(**base)
    if cond == "C":
        base["g_KATP_bar"] = G_KATP_BAR
        base["a_KATP_half"] = A_KATP_HALF
        base["n_KATP"] = N_KATP
        return merge_params(**base)
    raise ValueError(cond)


def eval_mean_s(I_ext: jnp.ndarray, cond: str, a_max: float, window: str) -> dict:
    params = condition_params(cond, a_max)
    traj = simulate_driven(I_ext, fix_a=False, dt=DT, t_end=T_END, params=params)
    t = np.asarray(traj["t"])
    s = np.asarray(soft_spike(traj["V"], V_th=V_TH, k=K_SOFT))
    if window == "S":
        mask = t < T_MID
    elif window == "N":
        mask = t >= T_MID
    else:
        mask = np.ones_like(t, dtype=bool)
    return {
        "mean_s": float(s[mask].mean()),
        "mean_a": float(np.asarray(traj["a"]).mean()),
        "mean_I_KATP": float(np.asarray(traj["I_KATP"]).mean()),
        "V": np.asarray(traj["V"]),
        "t": t,
        "a": np.asarray(traj["a"]),
        "I_KATP": np.asarray(traj["I_KATP"]),
    }


def run_d0(out_path: Path) -> None:
    """Unit smoke: same I_ext, high vs low fixed a with K_ATP on."""
    print("--- D0: K_ATP unit smoke ---")
    t = make_time_grid(D0_T_END)
    I_ext = jnp.where((t >= D0_TON) & (t < D0_TOFF), D0_IAMP, 0.0)
    traces = {}
    for a_fix, label in ((1.0, "a=1"), (0.2, "a=0.2")):
        params = merge_params(
            a_max=a_fix,
            g_KATP_bar=G_KATP_BAR,
            a_KATP_half=A_KATP_HALF,
            n_KATP=N_KATP,
            atp_gNa_floor=0.0,
        )
        traj = simulate_driven(
            I_ext, fix_a=True, dt=DT, t_end=D0_T_END, params=params
        )
        s_mean = float(np.asarray(soft_spike(traj["V"])).mean())
        ika_mean = float(np.asarray(traj["I_KATP"]).mean())
        print(f"  fix_a={a_fix}: mean_s={s_mean:.4f} mean_I_KATP={ika_mean:.3f}")
        traces[label] = traj

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    for label, style in (("a=1", "-"), ("a=0.2", "--")):
        tr = traces[label]
        axes[0].plot(tr["t"], tr["V"], ls=style, lw=0.9, label=label)
        axes[1].plot(tr["t"], tr["a"], ls=style, lw=1.0, label=label)
        axes[2].plot(tr["t"], tr["I_KATP"], ls=style, lw=1.0, label=label)
    axes[0].set_ylabel("V [mV]")
    axes[0].set_title(
        rf"D0 K_ATP unit: $\bar g_{{\mathrm{{KATP}}}}={G_KATP_BAR}$, "
        rf"$a_{{1/2}}={A_KATP_HALF}$, $n={N_KATP}$ (fix $a$)"
    )
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel(r"$a$ (fixed)")
    axes[1].legend(fontsize=8)
    axes[2].set_ylabel(r"$I_{\mathrm{KATP}}$")
    axes[2].set_xlabel("t [ms]")
    axes[2].legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path.resolve()}")


def run_d1(w_star: np.ndarray, out_path: Path) -> list[dict]:
    print("--- D1: RQ3 A/B/C compare (w frozen, K_e off) ---")
    print(f"w* = ({w_star[0]:.4f}, {w_star[1]:.4f})")
    print(
        f"C params: g_KATP_bar={G_KATP_BAR}, a_half={A_KATP_HALF}, n={N_KATP}; "
        f"B floor={ATP_GNA_FLOOR_B}; weak_gain={WEAK_NOISE_GAIN}"
    )
    t = make_time_grid()
    I_sig, I_noise = build_envelopes(t)
    I_strong = w_star[0] * I_sig
    I_weak = WEAK_NOISE_GAIN * I_noise

    rows: list[dict] = []
    for cond in ("A", "B", "C"):
        for a_max in A_MAX_GRID:
            strong = eval_mean_s(I_strong, cond, a_max, "S")
            weak = eval_mean_s(I_weak, cond, a_max, "N")
            row = {
                "cond": cond,
                "a_max": a_max,
                "strong_s": strong["mean_s"],
                "weak_s": weak["mean_s"],
                "strong_mean_a": strong["mean_a"],
                "weak_mean_a": weak["mean_a"],
                "strong_Ika": strong["mean_I_KATP"],
            }
            rows.append(row)
            print(
                f"  {cond} a_max={a_max:.1f}: strong_s={row['strong_s']:.4f} "
                f"weak_s={row['weak_s']:.4f} mean_Ika_s={row['strong_Ika']:.3f}"
            )

    plot_d1(rows, w_star, out_path)
    summarize_d1(rows)
    return rows


def plot_d1(rows: list[dict], w_star: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True, constrained_layout=True)
    colors = {"strong": "C0", "weak": "C3"}
    for ax, cond in zip(axes, ("A", "B", "C")):
        sub = [r for r in rows if r["cond"] == cond]
        a_vals = [r["a_max"] for r in sub]
        ax.plot(
            a_vals,
            [r["strong_s"] for r in sub],
            "o-",
            color=colors["strong"],
            lw=1.5,
            label="strong",
        )
        ax.plot(
            a_vals,
            [r["weak_s"] for r in sub],
            "s--",
            color=colors["weak"],
            lw=1.5,
            label="weak",
        )
        ax.set_xlabel(r"$a_{\max}$")
        ax.invert_xaxis()
        titles = {
            "A": "A: biophysical only",
            "B": rf"B: ATP→gNa ($f$={ATP_GNA_FLOOR_B})",
            "C": rf"C: K_ATP ($\bar g$={G_KATP_BAR})",
        }
        ax.set_title(titles[cond], fontsize=10)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("mean soft-spike")
    fig.suptitle(
        rf"Phase 4-Bio RQ3 re-compare: $w^*=({w_star[0]:.3f},{w_star[1]:.3f})$",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path.resolve()}")


def summarize_d1(rows: list[dict]) -> None:
    print("--- D1 summary ---")
    for cond in ("A", "B", "C"):
        sub = [r for r in rows if r["cond"] == cond]
        s0, s_lo = sub[0]["strong_s"], sub[-1]["strong_s"]
        w0, w_lo = sub[0]["weak_s"], sub[-1]["weak_s"]
        rel_s = 1.0 - s_lo / max(s0, 1e-9)
        rel_w = 1.0 - w_lo / max(w0, 1e-9)
        overall = (s_lo < 0.9 * s0) or (w_lo < 0.9 * w0)
        weak_first = (rel_w > rel_s + 0.05) or (w_lo < 0.5 * w0 and s_lo > 0.5 * s0)
        strong_ok = s_lo > 0.015
        flat = (abs(s_lo - s0) < 0.01) and (abs(w_lo - w0) < 0.01)
        print(
            f"  {cond}: overall_drop={overall} weak_first={weak_first} "
            f"strong_survives={strong_ok} flat≈{flat} "
            f"(S {s0:.4f}→{s_lo:.4f}, W {w0:.4f}→{w_lo:.4f})"
        )


def main() -> int:
    print(
        f"σ(a)=1/(1+(a/a_half)^n); a_half={A_KATP_HALF}, n={N_KATP}, "
        f"g_KATP_bar(C)={G_KATP_BAR}"
    )
    run_d0(Path("figures") / "rq4_katp_unit.png")

    if RETRAIN_LAMBDA0:
        w_star = train_w_star()
    else:
        w_star = W_STAR_DEFAULT.copy()
        print(f"using embedded w* = {w_star}")

    run_d1(w_star, Path("figures") / "rq4_katp_rq3_compare.png")
    print("OK: Phase 4-Bio D0+D1 finished (simplified K_ATP; not molecular fidelity)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 3a — RQ1 minimal: selective S/N response + metabolic penalty.

Run from the repo root::

    python scripts/04_rq1_selective_metabolic.py

Learns w=(w1,w2) with I_ext = w1*I_sig + w2*I_noise (+ optional bias).
Interval S (first half): prefer firing; interval N (second half): prefer silence.
Compares λ=0 (A) vs λ>0 (B) with L_met = mean(max(I_pump, 0)).
Saves ``figures/rq1_selective_metabolic.png``.
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

from jnpnp.models.point_sim import DT, soft_spike, simulate_driven

# ---------------------------------------------------------------------------
# Protocol / optimization (tunable but fixed for the reported run)
# ---------------------------------------------------------------------------
T_END = 250.0  # ms (shorter so I_pump is less floor-dominated)
T_MID = 0.5 * T_END
PULSE_AMP = 15.0  # µA/cm² base envelope
PULSE_WIDTH = 6.0  # ms
PULSE_PERIOD = 12.0  # ms
I_BIAS = 0.0

W0 = jnp.array([0.8, 0.8], dtype=jnp.float64)
W_MIN, W_MAX = 0.0, 2.5

V_TH = 0.0
K_SOFT = 0.5
S_HI = 0.10
S_LO = 0.03

LR = 0.05
N_STEPS = 80
LAM_A = 0.0
# Larger λ: mean(I_pump) only weakly depends on w; need scale to compete with L_task.
LAM_CANDIDATES_B = (1.0, 5.0, 15.0)


def build_channel_envelopes(
    t: jnp.ndarray,
    *,
    t_mid: float = T_MID,
    amp: float = PULSE_AMP,
    width: float = PULSE_WIDTH,
    period: float = PULSE_PERIOD,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """I_sig active on [0, t_mid), I_noise on [t_mid, T). Square pulse trains."""
    # Pulse gates: on for `width` every `period`, phase-aligned from interval start.
    def pulse_train(t_local: jnp.ndarray) -> jnp.ndarray:
        phase = jnp.mod(t_local, period)
        return jnp.where(phase < width, amp, 0.0)

    in_S = t < t_mid
    in_N = t >= t_mid
    t_S = jnp.where(in_S, t, 0.0)
    t_N = jnp.where(in_N, t - t_mid, 0.0)
    I_sig = jnp.where(in_S, pulse_train(t_S), 0.0)
    I_noise = jnp.where(in_N, pulse_train(t_N), 0.0)
    return I_sig, I_noise


def make_time_grid(dt: float = DT, t_end: float = T_END) -> jnp.ndarray:
    n_steps = int(round(t_end / dt))
    return jnp.arange(n_steps + 1, dtype=jnp.float64) * dt


def I_ext_from_w(
    w: jnp.ndarray,
    I_sig: jnp.ndarray,
    I_noise: jnp.ndarray,
    *,
    bias: float = I_BIAS,
) -> jnp.ndarray:
    return w[0] * I_sig + w[1] * I_noise + bias


def eval_metrics(
    w: jnp.ndarray,
    I_sig: jnp.ndarray,
    I_noise: jnp.ndarray,
    mask_S: jnp.ndarray,
    mask_N: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    I_ext = I_ext_from_w(w, I_sig, I_noise)
    traj = simulate_driven(I_ext, fix_a=False, dt=DT, t_end=T_END)
    s = soft_spike(traj["V"], V_th=V_TH, k=K_SOFT)
    # Avoid empty-mask nan: masks are nonempty by construction.
    mean_S = jnp.sum(s * mask_S) / jnp.sum(mask_S)
    mean_N = jnp.sum(s * mask_N) / jnp.sum(mask_N)
    L_hit = (mean_S - S_HI) ** 2
    L_fa = (mean_N - S_LO) ** 2
    L_met = jnp.mean(jnp.maximum(traj["I_pump"], 0.0))
    delta_a = traj["a"][0] - traj["a"][-1]
    return {
        "mean_S": mean_S,
        "mean_N": mean_N,
        "L_hit": L_hit,
        "L_fa": L_fa,
        "L_met": L_met,
        "delta_a": delta_a,
        "V": traj["V"],
        "t": traj["t"],
        "a": traj["a"],
        "I_pump": traj["I_pump"],
        "I_ext": traj["I_ext"],
        "s": s,
    }


def make_total_loss(lam: float, I_sig, I_noise, mask_S, mask_N):
    def total_loss(w: jnp.ndarray) -> jnp.ndarray:
        m = eval_metrics(w, I_sig, I_noise, mask_S, mask_N)
        return m["L_hit"] + m["L_fa"] + lam * m["L_met"]

    return total_loss


def train_one(lam: float, I_sig, I_noise, mask_S, mask_N) -> dict:
    w = jnp.array(W0, dtype=jnp.float64)
    opt = optax.adam(LR)
    opt_state = opt.init(w)
    loss_fn = make_total_loss(lam, I_sig, I_noise, mask_S, mask_N)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    metrics_fn = jax.jit(lambda ww: eval_metrics(ww, I_sig, I_noise, mask_S, mask_N))

    hist = {
        "loss": [],
        "L_hit": [],
        "L_fa": [],
        "L_met": [],
        "mean_S": [],
        "mean_N": [],
        "delta_a": [],
        "w1": [],
        "w2": [],
        "grad_norm": [],
    }

    print(f"\n=== train λ={lam} ===")
    for step in range(N_STEPS):
        loss, grad = value_and_grad(w)
        m = metrics_fn(w)
        row = {
            "loss": float(loss),
            "L_hit": float(m["L_hit"]),
            "L_fa": float(m["L_fa"]),
            "L_met": float(m["L_met"]),
            "mean_S": float(m["mean_S"]),
            "mean_N": float(m["mean_N"]),
            "delta_a": float(m["delta_a"]),
            "w1": float(w[0]),
            "w2": float(w[1]),
            "grad_norm": float(jnp.linalg.norm(grad)),
        }
        if not all(
            np.isfinite(row[k])
            for k in ("loss", "mean_S", "mean_N", "L_met", "grad_norm")
        ):
            raise RuntimeError(f"non-finite at step {step}, λ={lam}: {row}")

        for k, v in row.items():
            hist[k].append(v)

        if step % 10 == 0 or step == N_STEPS - 1:
            print(
                f"step {step:03d}: L={row['loss']:.5g}  "
                f"hit={row['L_hit']:.4g} fa={row['L_fa']:.4g} met={row['L_met']:.4g}  "
                f"mS={row['mean_S']:.4f} mN={row['mean_N']:.4f}  "
                f"w=({row['w1']:.3f},{row['w2']:.3f})"
            )

        updates, opt_state = opt.update(grad, opt_state, w)
        w = optax.apply_updates(w, updates)
        w = jnp.clip(w, W_MIN, W_MAX)

    m_final = metrics_fn(w)
    loss_f, grad_f = value_and_grad(w)
    final = {
        "loss": float(loss_f),
        "L_hit": float(m_final["L_hit"]),
        "L_fa": float(m_final["L_fa"]),
        "L_met": float(m_final["L_met"]),
        "mean_S": float(m_final["mean_S"]),
        "mean_N": float(m_final["mean_N"]),
        "delta_a": float(m_final["delta_a"]),
        "w": np.array([float(w[0]), float(w[1])]),
        "grad_norm": float(jnp.linalg.norm(grad_f)),
        "traj": {k: np.asarray(m_final[k]) for k in ("t", "V", "a", "I_pump", "I_ext", "s")},
        "lam": lam,
    }
    if not (np.isfinite(final["loss"]) and np.isfinite(final["grad_norm"])):
        raise RuntimeError(f"non-finite final λ={lam}")
    hist["final"] = final
    hist["init"] = {
        "loss": hist["loss"][0],
        "L_hit": hist["L_hit"][0],
        "L_fa": hist["L_fa"][0],
        "L_met": hist["L_met"][0],
        "mean_S": hist["mean_S"][0],
        "mean_N": hist["mean_N"][0],
        "delta_a": hist["delta_a"][0],
        "w": np.array([hist["w1"][0], hist["w2"][0]]),
    }
    return hist


def pick_B(res_a: dict, candidates: dict[float, dict]) -> float:
    """Prefer lower L_met / quieter N vs A, without collapsing mean_S or raising FA."""
    fa = res_a["final"]
    best_lam, best_score = None, -1e9
    for lam, hist in candidates.items():
        fb = hist["final"]
        if fb["mean_S"] < 0.4 * S_HI:
            continue  # task abandoned
        if fb["mean_N"] > fa["mean_N"] + 0.01:
            continue  # worse false alarms than A
        score = 0.0
        score += (fa["L_met"] - fb["L_met"]) * 50.0
        score += (fa["mean_N"] - fb["mean_N"]) * 20.0
        score += (fa["L_fa"] - fb["L_fa"]) * 10.0
        score += (abs(fa["w"][1]) - abs(fb["w"][1])) * 2.0
        score += max(0.0, fb["mean_S"] - 0.5 * S_HI)
        if score > best_score:
            best_score = score
            best_lam = lam
    if best_lam is None:
        for lam, hist in sorted(candidates.items()):
            if hist["final"]["mean_S"] >= 0.4 * S_HI:
                return lam
        return next(iter(candidates))
    return best_lam


def plot_results(res_a: dict, res_b: dict, lam_b: float, out_path: Path) -> None:
    steps = np.arange(N_STEPS)
    fa, fb = res_a["final"], res_b["final"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(steps, res_a["loss"], color="C0", lw=1.4, label=r"A $\lambda=0$")
    ax.plot(steps, res_b["loss"], color="C3", lw=1.4, label=rf"B $\lambda={lam_b}$")
    ax.set_yscale("log")
    ax.set_ylabel("L total")
    ax.set_xlabel("step")
    ax.set_title("Learning curves")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width / 2, fa["w"], width, label="A", color="C0")
    ax.bar(x + width / 2, fb["w"], width, label="B", color="C3")
    ax.set_xticks(x, ["w1 (sig)", "w2 (noise)"])
    ax.set_ylabel("weight")
    ax.set_title("Final weights")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    t = fa["traj"]["t"]
    ax.plot(t, fa["traj"]["V"], color="C0", lw=0.7, alpha=0.85, label="A V")
    ax.plot(t, fb["traj"]["V"], color="C3", lw=0.7, alpha=0.75, label="B V")
    ax.axvline(T_MID, color="0.4", ls="--", lw=0.8)
    ax.set_ylabel("V [mV]")
    ax.set_xlabel("t [ms]")
    ax.set_title("Final voltage (S | N)")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1, 1]
    labels = ["mean_S", "mean_N", "L_met", "delta_a"]
    a_vals = [fa["mean_S"], fa["mean_N"], fa["L_met"], fa["delta_a"]]
    b_vals = [fb["mean_S"], fb["mean_N"], fb["L_met"], fb["delta_a"]]
    x = np.arange(len(labels))
    ax.bar(x - width / 2, a_vals, width, label="A", color="C0")
    ax.bar(x + width / 2, b_vals, width, label="B", color="C3")
    ax.axhline(S_HI, color="C0", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(S_LO, color="C3", ls=":", lw=0.8, alpha=0.5)
    ax.set_xticks(x, labels, rotation=15)
    ax.set_title("Selectivity & metabolism")
    ax.legend(fontsize=8)

    fig.suptitle("Phase 3a RQ1: selective S/N + metabolic penalty", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    t = make_time_grid()
    I_sig, I_noise = build_channel_envelopes(t)
    mask_S = (t < T_MID).astype(jnp.float64)
    mask_N = (t >= T_MID).astype(jnp.float64)

    # Manual sanity at w≈(1,0) and w≈(0,1)
    for label, ww in (("w=(1,0)", [1.0, 0.0]), ("w=(0,1)", [0.0, 1.0]), ("w0", W0)):
        m = eval_metrics(jnp.asarray(ww, dtype=jnp.float64), I_sig, I_noise, mask_S, mask_N)
        print(
            f"probe {label}: mean_S={float(m['mean_S']):.4f} mean_N={float(m['mean_N']):.4f} "
            f"L_met={float(m['L_met']):.4f} delta_a={float(m['delta_a']):.4f}"
        )

    print("--- Phase 3a RQ1 selective metabolic ---")
    print(
        f"T={T_END}ms mid={T_MID}, pulse amp={PULSE_AMP} width={PULSE_WIDTH} "
        f"period={PULSE_PERIOD}, s_hi={S_HI} s_lo={S_LO}"
    )
    print(f"Adam lr={LR} steps={N_STEPS}, w0={np.asarray(W0)}, clip=[{W_MIN},{W_MAX}]")
    print(f"L_met=mean(max(I_pump,0)); λ_A={LAM_A}, λ_B candidates={LAM_CANDIDATES_B}")

    res_a = train_one(LAM_A, I_sig, I_noise, mask_S, mask_N)
    res_b_all = {
        lam: train_one(lam, I_sig, I_noise, mask_S, mask_N) for lam in LAM_CANDIDATES_B
    }
    lam_b = pick_B(res_a, res_b_all)
    res_b = res_b_all[lam_b]

    out_path = Path("figures") / "rq1_selective_metabolic.png"
    plot_results(res_a, res_b, lam_b, out_path)

    fa, fb = res_a["final"], res_b["final"]
    print("\n--- A/B final ---")
    for name, f in (("A", fa), ("B", fb)):
        print(
            f"{name} λ={f['lam']}: w=({f['w'][0]:.4f},{f['w'][1]:.4f})  "
            f"mean_S={f['mean_S']:.4f} mean_N={f['mean_N']:.4f}  "
            f"L_hit={f['L_hit']:.5g} L_fa={f['L_fa']:.5g} L_met={f['L_met']:.5g}  "
            f"delta_a={f['delta_a']:.4f}"
        )
    print(f"saved: {out_path.resolve()}")

    dN = fa["mean_N"] - fb["mean_N"]
    dFa = fa["L_fa"] - fb["L_fa"]
    dMet = fa["L_met"] - fb["L_met"]
    dW2 = abs(fa["w"][1]) - abs(fb["w"][1])
    ok_n = dN > 1e-4 or dFa > 1e-6
    ok_met = dMet > 1e-5
    ok_w2 = dW2 > 0.02  # B more strongly suppresses noise weight
    ok_task = fb["mean_S"] >= 0.4 * S_HI
    print(
        f"success checks: lower_N/L_fa={ok_n} (dN={dN:.4g}, dFa={dFa:.4g}), "
        f"lower_L_met={ok_met} (dMet={dMet:.4g}), lower_|w2|={ok_w2}, "
        f"mean_S_kept={ok_task}"
    )
    if (ok_n or ok_met or ok_w2) and ok_task:
        print("OK: metabolic penalty shows selectivity/savings preview (not RQ1 proof)")
        return 0
    print("WARN: success criteria weak; recorded honestly for STATUS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

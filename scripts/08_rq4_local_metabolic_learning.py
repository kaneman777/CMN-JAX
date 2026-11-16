"""Phase 4-Learn — local ATP 3-factor rule vs Adam (minimal).

Run from the repo root::

    python scripts/08_rq4_local_metabolic_learning.py

Same S/N selective task as Phase 3a. Compares:
  - Adam: L = L_hit + L_fa (+ optional small λ_met)
  - Local: Δw ∝ <e_i * g(a)> with optional minimal task assist if pure local collapses

Claim ceiling: local ATP-gated learning preview or honest null — not a BPTT-free proof.
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
from jnpnp.numerics.local_metabolic import local_delta_w, task_assist_delta

# ---------------------------------------------------------------------------
# Phase 3a-matched protocol
# ---------------------------------------------------------------------------
T_END = 250.0
T_MID = 0.5 * T_END
PULSE_AMP = 15.0
PULSE_WIDTH = 6.0
PULSE_PERIOD = 12.0

W0 = jnp.array([0.8, 0.8], dtype=jnp.float64)
W_MIN, W_MAX = 0.0, 2.5

V_TH = 0.0
K_SOFT = 0.5
S_HI = 0.10
S_LO = 0.03

N_STEPS = 80
ADAM_LR = 0.05
ADAM_LAM = 0.0  # match 3a-A; small λ optional

# Local rule — pure local succeeded (selectivity); assist kept off
TAU_E = 20.0  # ms
I_REF = 15.0  # eligibility input scale (ė uses I/I_ref)
USE_TASK_ASSIST = False
ETA_ASSIST = 0.15
LOCAL_VARIANTS = (
    {"name": "Local", "a_crit": 0.45, "beta": 0.15, "eta": 0.25},
    {"name": "Local_hiCrit", "a_crit": 0.55, "beta": 0.12, "eta": 0.25},
)


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


def eval_traj(w: jnp.ndarray, I_sig: jnp.ndarray, I_noise: jnp.ndarray) -> dict:
    I_ext = w[0] * I_sig + w[1] * I_noise
    traj = simulate_driven(I_ext, fix_a=False, dt=DT, t_end=T_END)
    s = soft_spike(traj["V"], V_th=V_TH, k=K_SOFT)
    t = traj["t"]
    mask_S = (t < T_MID).astype(jnp.float64)
    mask_N = (t >= T_MID).astype(jnp.float64)
    mean_S = jnp.sum(s * mask_S) / jnp.sum(mask_S)
    mean_N = jnp.sum(s * mask_N) / jnp.sum(mask_N)
    L_met = jnp.mean(jnp.maximum(traj["I_pump"], 0.0))
    delta_a = traj["a"][0] - traj["a"][-1]
    return {
        "mean_S": mean_S,
        "mean_N": mean_N,
        "L_hit": (mean_S - S_HI) ** 2,
        "L_fa": (mean_N - S_LO) ** 2,
        "L_met": L_met,
        "delta_a": delta_a,
        "s": s,
        "a": traj["a"],
        "V": traj["V"],
        "t": t,
    }


def train_adam(I_sig, I_noise) -> dict:
    """Phase-3a-matched Adam (jitted loss + metrics; λ default 0)."""
    print(f"--- Adam (lr={ADAM_LR}, λ={ADAM_LAM}, steps={N_STEPS}) ---")
    t = make_time_grid()
    mask_S = (t < T_MID).astype(jnp.float64)
    mask_N = (t >= T_MID).astype(jnp.float64)

    def loss_fn(w):
        I_ext = w[0] * I_sig + w[1] * I_noise
        traj = simulate_driven(I_ext, fix_a=False, dt=DT, t_end=T_END)
        s = soft_spike(traj["V"], V_th=V_TH, k=K_SOFT)
        mean_S = jnp.sum(s * mask_S) / jnp.sum(mask_S)
        mean_N = jnp.sum(s * mask_N) / jnp.sum(mask_N)
        L_met = jnp.mean(jnp.maximum(traj["I_pump"], 0.0))
        return (mean_S - S_HI) ** 2 + (mean_N - S_LO) ** 2 + ADAM_LAM * L_met

    metrics_fn = jax.jit(lambda ww: eval_traj(ww, I_sig, I_noise))
    w = jnp.array(W0)
    opt = optax.adam(ADAM_LR)
    state = opt.init(w)
    vag = jax.jit(jax.value_and_grad(loss_fn))
    hist = {k: [] for k in ("w1", "w2", "mean_S", "mean_N", "L_met", "delta_a", "loss")}
    for step in range(N_STEPS):
        loss, g = vag(w)
        m = metrics_fn(w)
        hist["w1"].append(float(w[0]))
        hist["w2"].append(float(w[1]))
        hist["mean_S"].append(float(m["mean_S"]))
        hist["mean_N"].append(float(m["mean_N"]))
        hist["L_met"].append(float(m["L_met"]))
        hist["delta_a"].append(float(m["delta_a"]))
        hist["loss"].append(float(loss))
        if step % 20 == 0 or step == N_STEPS - 1:
            print(
                f"  step {step:03d}: L={float(loss):.4g} w=({float(w[0]):.3f},{float(w[1]):.3f}) "
                f"S={float(m['mean_S']):.4f} N={float(m['mean_N']):.4f}"
            )
        updates, state = opt.update(g, state, w)
        w = jnp.clip(optax.apply_updates(w, updates), W_MIN, W_MAX)
    final = metrics_fn(w)
    return {"name": "Adam", "w": np.array([float(w[0]), float(w[1])]), "hist": hist, "final": final}


def train_local(I_sig, I_noise, *, name: str, a_crit: float, beta: float, eta: float) -> dict:
    assist_note = f"+assist η={ETA_ASSIST}" if USE_TASK_ASSIST else "pure local"
    print(
        f"--- {name} (η={eta}, a_crit={a_crit}, β={beta}, τ_e={TAU_E}, {assist_note}) ---"
    )
    w = jnp.array(W0)
    hist = {k: [] for k in ("w1", "w2", "mean_S", "mean_N", "L_met", "delta_a", "loss")}
    for step in range(N_STEPS):
        m = eval_traj(w, I_sig, I_noise)
        loss = float(m["L_hit"] + m["L_fa"])
        hist["w1"].append(float(w[0]))
        hist["w2"].append(float(w[1]))
        hist["mean_S"].append(float(m["mean_S"]))
        hist["mean_N"].append(float(m["mean_N"]))
        hist["L_met"].append(float(m["L_met"]))
        hist["delta_a"].append(float(m["delta_a"]))
        hist["loss"].append(loss)
        if step % 20 == 0 or step == N_STEPS - 1:
            print(
                f"  step {step:03d}: L={loss:.4g} w=({float(w[0]):.3f},{float(w[1]):.3f}) "
                f"S={float(m['mean_S']):.4f} N={float(m['mean_N']):.4f}"
            )
        dw = local_delta_w(
            I_sig,
            I_noise,
            m["s"],
            m["a"],
            dt=DT,
            tau_e=TAU_E,
            a_crit=a_crit,
            beta=beta,
            eta=eta,
            M=1.0,
            I_ref=I_REF,
        )
        if USE_TASK_ASSIST:
            dw = dw + task_assist_delta(
                m["mean_S"],
                m["mean_N"],
                s_hi=S_HI,
                s_lo=S_LO,
                eta_assist=ETA_ASSIST,
            )
        w = jnp.clip(w + dw, W_MIN, W_MAX)
    final = eval_traj(w, I_sig, I_noise)
    return {
        "name": name,
        "w": np.array([float(w[0]), float(w[1])]),
        "hist": hist,
        "final": final,
        "a_crit": a_crit,
        "beta": beta,
        "eta": eta,
        "assist": USE_TASK_ASSIST,
    }


def plot_results(runs: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    steps = np.arange(N_STEPS)

    ax = axes[0, 0]
    for run, ls in zip(runs, ("-", "--", ":")):
        ax.plot(steps, run["hist"]["w1"], ls=ls, color="C0", label=f"{run['name']} w1")
        ax.plot(steps, run["hist"]["w2"], ls=ls, color="C3", label=f"{run['name']} w2")
    ax.set_xlabel("step")
    ax.set_ylabel("w")
    ax.set_title("Weight trajectories")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    for run, ls in zip(runs, ("-", "--", ":")):
        ax.plot(steps, run["hist"]["mean_S"], ls=ls, color="C0", label=f"{run['name']} S")
        ax.plot(steps, run["hist"]["mean_N"], ls=ls, color="C3", label=f"{run['name']} N")
    ax.axhline(S_HI, color="C0", ls=":", lw=0.7)
    ax.axhline(S_LO, color="C3", ls=":", lw=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("mean soft-spike")
    ax.set_title("mean_S / mean_N")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 0]
    for run, ls in zip(runs, ("-", "--", ":")):
        ax.plot(steps, run["hist"]["delta_a"], ls=ls, label=run["name"])
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\Delta a$")
    ax.set_title("ATP drawdown")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for run, ls in zip(runs, ("-", "--", ":")):
        ax.plot(steps, run["hist"]["L_met"], ls=ls, label=run["name"])
    ax.set_xlabel("step")
    ax.set_ylabel(r"mean $\max(I_{\mathrm{pump}},0)$")
    ax.set_title("L_met")
    ax.legend(fontsize=8)

    assist = "hybrid local+assist" if USE_TASK_ASSIST else "pure local"
    fig.suptitle(f"Phase 4-Learn: Local ({assist}) vs Adam", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path.resolve()}")


def classify_outcome(local_run: dict, adam_run: dict) -> str:
    lw = local_run["w"]
    lf = local_run["final"]
    af = adam_run["final"]
    # Collapse: both channels silent or both saturated fire with no S/N split
    mean_S, mean_N = float(lf["mean_S"]), float(lf["mean_N"])
    if mean_S < 0.01 and mean_N < 0.01:
        return "collapse"
    selective = (lw[1] < 0.85 * lw[0]) or (mean_N < 0.7 * mean_S and mean_S > 0.03)
    save_more = float(lf["delta_a"]) < 0.95 * float(af["delta_a"]) and mean_S > 0.02
    if selective or save_more:
        return "preview"
    return "null"


def main() -> int:
    t = make_time_grid()
    I_sig, I_noise = build_envelopes(t)
    print(
        f"e: ė=-e/τ_e + I_i*s; g=tanh((a-a_crit)/β); "
        f"assist={'ON' if USE_TASK_ASSIST else 'OFF'}"
    )

    # Probe pure local briefly if assist is on — still run configured mode as main
    adam = train_adam(I_sig, I_noise)
    locals_ = [
        train_local(I_sig, I_noise, name=v["name"], a_crit=v["a_crit"], beta=v["beta"], eta=v["eta"])
        for v in LOCAL_VARIANTS
    ]
    runs = [adam] + locals_
    out = Path("figures") / "rq4_local_vs_adam.png"
    plot_results(runs, out)

    print("--- final table ---")
    print(f"{'cond':16s} {'w1':>7s} {'w2':>7s} {'mean_S':>8s} {'mean_N':>8s} {'Δa':>7s} {'L_met':>7s}")
    for run in runs:
        f = run["final"]
        print(
            f"{run['name']:16s} {run['w'][0]:7.3f} {run['w'][1]:7.3f} "
            f"{float(f['mean_S']):8.4f} {float(f['mean_N']):8.4f} "
            f"{float(f['delta_a']):7.3f} {float(f['L_met']):7.4f}"
        )
    outcome = classify_outcome(locals_[0], adam)
    print(f"outcome (primary Local vs Adam): {outcome}")
    print(
        f"mode: {'hybrid (local + minimal task assist)' if USE_TASK_ASSIST else 'pure local'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

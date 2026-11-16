"""Phase 4-Learn A1 — Local + interval-S firing-rate homeostasis vs Adam / Local.

Run from the repo root::

    python scripts/09_rq4_local_homeo.py

Not pure local: Δw = Δw_met + Δw_homeo with
  Δw1_homeo = -η_h * (mean_S - s_target), Δw2_homeo = 0.

Compares Adam / Local (homeo off) / Local+homeo. Saves
``figures/rq4_local_homeo.png``.

Claim ceiling: homeostasis can curb gain runaway while keeping selectivity
preview — not cortical homeostasis reproduction or Adam-free proof.
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
from jnpnp.numerics.local_metabolic import firing_rate_homeo_delta, local_delta_w

# ---------------------------------------------------------------------------
# Phase 3a / 08-matched protocol
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
ADAM_LAM = 0.0

TAU_E = 20.0
I_REF = 15.0
A_CRIT = 0.45
BETA_G = 0.15
ETA_LOCAL = 0.25

# A1 homeostasis (near Adam final mean_S ≈ 0.08)
# η_h must be large enough to counter Δw_met on w1 (else clip returns).
S_TARGET = 0.08
ETA_H = 8.0


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
    }


def empty_hist() -> dict:
    return {k: [] for k in ("w1", "w2", "mean_S", "mean_N", "L_met", "delta_a", "loss")}


def record(hist: dict, w, m, loss: float) -> None:
    hist["w1"].append(float(w[0]))
    hist["w2"].append(float(w[1]))
    hist["mean_S"].append(float(m["mean_S"]))
    hist["mean_N"].append(float(m["mean_N"]))
    hist["L_met"].append(float(m["L_met"]))
    hist["delta_a"].append(float(m["delta_a"]))
    hist["loss"].append(loss)


def train_adam(I_sig, I_noise) -> dict:
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
    hist = empty_hist()
    for step in range(N_STEPS):
        loss, g = vag(w)
        m = metrics_fn(w)
        record(hist, w, m, float(loss))
        if step % 20 == 0 or step == N_STEPS - 1:
            print(
                f"  step {step:03d}: L={float(loss):.4g} w=({float(w[0]):.3f},{float(w[1]):.3f}) "
                f"S={float(m['mean_S']):.4f} N={float(m['mean_N']):.4f}"
            )
        updates, state = opt.update(g, state, w)
        w = jnp.clip(optax.apply_updates(w, updates), W_MIN, W_MAX)
    final = metrics_fn(w)
    return {"name": "Adam", "w": np.array([float(w[0]), float(w[1])]), "hist": hist, "final": final}


def train_local(
    I_sig,
    I_noise,
    *,
    name: str,
    use_homeo: bool,
    eta_h: float = ETA_H,
    s_target: float = S_TARGET,
) -> dict:
    mode = f"+homeo η_h={eta_h} s*={s_target}" if use_homeo else "homeo OFF (pure met)"
    print(
        f"--- {name} (η={ETA_LOCAL}, a_crit={A_CRIT}, β={BETA_G}, {mode}) ---"
    )
    w = jnp.array(W0)
    hist = empty_hist()
    for step in range(N_STEPS):
        m = eval_traj(w, I_sig, I_noise)
        loss = float(m["L_hit"] + m["L_fa"])
        record(hist, w, m, loss)
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
            a_crit=A_CRIT,
            beta=BETA_G,
            eta=ETA_LOCAL,
            M=1.0,
            I_ref=I_REF,
        )
        if use_homeo:
            dw = dw + firing_rate_homeo_delta(
                m["mean_S"], s_target=s_target, eta_h=eta_h
            )
        w = jnp.clip(w + dw, W_MIN, W_MAX)
    final = eval_traj(w, I_sig, I_noise)
    return {
        "name": name,
        "w": np.array([float(w[0]), float(w[1])]),
        "hist": hist,
        "final": final,
        "homeo": use_homeo,
    }


def plot_results(runs: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    steps = np.arange(N_STEPS)
    styles = ("-", "--", ":")

    ax = axes[0, 0]
    for run, ls in zip(runs, styles):
        ax.plot(steps, run["hist"]["w1"], ls=ls, color="C0", label=f"{run['name']} w1")
        ax.plot(steps, run["hist"]["w2"], ls=ls, color="C3", label=f"{run['name']} w2")
    ax.axhline(W_MAX, color="0.6", ls=":", lw=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("w")
    ax.set_title("Weight trajectories")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    for run, ls in zip(runs, styles):
        ax.plot(steps, run["hist"]["mean_S"], ls=ls, color="C0", label=f"{run['name']} S")
        ax.plot(steps, run["hist"]["mean_N"], ls=ls, color="C3", label=f"{run['name']} N")
    ax.axhline(S_TARGET, color="C0", ls=":", lw=0.8, label="s_target")
    ax.set_xlabel("step")
    ax.set_ylabel("mean soft-spike")
    ax.set_title("mean_S / mean_N")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 0]
    for run, ls in zip(runs, styles):
        ax.plot(steps, run["hist"]["delta_a"], ls=ls, label=run["name"])
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\Delta a$")
    ax.set_title("ATP drawdown")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for run, ls in zip(runs, styles):
        ax.plot(steps, run["hist"]["L_met"], ls=ls, label=run["name"])
    ax.set_xlabel("step")
    ax.set_ylabel(r"mean $\max(I_{\mathrm{pump}},0)$")
    ax.set_title("L_met")
    ax.legend(fontsize=8)

    fig.suptitle(
        rf"Phase 4-Learn A1: Local+homeo ($s^*={S_TARGET}$, $\eta_h={ETA_H}$) vs Adam / Local",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path.resolve()}")


def check_success(adam: dict, local: dict, homeo: dict) -> None:
    hw, hf = homeo["w"], homeo["final"]
    lf = local["final"]
    af = adam["final"]
    w1_ok = float(hw[0]) < W_MAX - 0.05
    n_ok = float(hf["mean_N"]) < 0.02
    s_near = abs(float(hf["mean_S"]) - S_TARGET) < 0.04
    da_better = float(hf["delta_a"]) < float(lf["delta_a"]) - 0.005
    lmet_better = float(hf["L_met"]) < float(lf["L_met"]) - 0.0005
    print("--- success checks (Local+homeo) ---")
    print(f"  w1 not at clip: {w1_ok} (w1={hw[0]:.3f}, max={W_MAX})")
    print(f"  mean_N low: {n_ok} (N={float(hf['mean_N']):.4f})")
    print(f"  mean_S near s*: {s_near} (S={float(hf['mean_S']):.4f}, s*={S_TARGET})")
    print(
        f"  Δa vs Local improved: {da_better} "
        f"(homeo {float(hf['delta_a']):.3f} vs Local {float(lf['delta_a']):.3f}; "
        f"Adam {float(af['delta_a']):.3f})"
    )
    print(
        f"  L_met vs Local improved: {lmet_better} "
        f"(homeo {float(hf['L_met']):.4f} vs Local {float(lf['L_met']):.4f}; "
        f"Adam {float(af['L_met']):.4f})"
    )


def main() -> int:
    print(
        f"Local+homeo (NOT pure local): Δw=Δw_met+Δw_homeo; "
        f"s_target={S_TARGET}, η_h={ETA_H}, η_met={ETA_LOCAL}"
    )
    t = make_time_grid()
    I_sig, I_noise = build_envelopes(t)

    adam = train_adam(I_sig, I_noise)
    local = train_local(I_sig, I_noise, name="Local", use_homeo=False)
    homeo = train_local(I_sig, I_noise, name="Local+homeo", use_homeo=True)

    runs = [adam, local, homeo]
    out = Path("figures") / "rq4_local_homeo.png"
    plot_results(runs, out)

    print("--- final table ---")
    print(f"{'cond':14s} {'w1':>7s} {'w2':>7s} {'mean_S':>8s} {'mean_N':>8s} {'Δa':>7s} {'L_met':>7s}")
    for run in runs:
        f = run["final"]
        print(
            f"{run['name']:14s} {run['w'][0]:7.3f} {run['w'][1]:7.3f} "
            f"{float(f['mean_S']):8.4f} {float(f['mean_N']):8.4f} "
            f"{float(f['delta_a']):7.3f} {float(f['L_met']):7.4f}"
        )
    check_success(adam, local, homeo)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 2b: Optax update of stim_gain to lower the Phase-2a dummy loss.

Run from the repo root::

    python scripts/02_train_stim_gain.py

Updates scalar ``stim_gain`` (I_ext = stim_gain * I0) with Adam so that
L = (mean_t sigmoid(k*(V-V_th)) - target)^2 decreases. No networks, surrogate
gradients, or Diffrax. Saves ``figures/train_stim_gain.png``.
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

from jnpnp.models.point_sim import soft_spike, simulate_trajectory

STIM_GAIN0 = 1.0
# Initial mean_s ≈ 0.039 at stim_gain=1; target far enough for a clear drop in L.
TARGET = 0.08
V_TH = 0.0
K_SOFT = 0.5
LR = 0.05
N_STEPS = 50
# Keep stim_gain in a safe band for HH excitability.
GAIN_MIN = 0.2
GAIN_MAX = 2.5


def mean_soft_spike(stim_gain: jnp.ndarray) -> jnp.ndarray:
    traj = simulate_trajectory(stim_gain, fix_a=False)
    return jnp.mean(soft_spike(traj["V"], V_th=V_TH, k=K_SOFT))


def loss_fn(stim_gain: jnp.ndarray) -> jnp.ndarray:
    return (mean_soft_spike(stim_gain) - TARGET) ** 2


def main() -> int:
    theta = jnp.asarray(STIM_GAIN0, dtype=jnp.float64)
    opt = optax.adam(LR)
    opt_state = opt.init(theta)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    mean_s_fn = jax.jit(mean_soft_spike)

    hist_loss: list[float] = []
    hist_gain: list[float] = []
    hist_mean: list[float] = []
    hist_grad: list[float] = []

    print("--- Phase 2b train stim_gain ---")
    print(f"target={TARGET}, lr={LR}, steps={N_STEPS}, adam")
    print(f"V_th={V_TH}, k={K_SOFT}, gain clip=[{GAIN_MIN}, {GAIN_MAX}]")

    for step in range(N_STEPS):
        loss, grad = value_and_grad(theta)
        mean_s = mean_s_fn(theta)
        loss_f = float(loss)
        grad_f = float(grad)
        mean_f = float(mean_s)
        gain_f = float(theta)

        if not (np.isfinite(loss_f) and np.isfinite(grad_f) and np.isfinite(mean_f)):
            print(f"FAIL at step {step}: non-finite loss/grad/mean_s")
            print(f"  loss={loss_f}, grad={grad_f}, mean_s={mean_f}, stim_gain={gain_f}")
            return 1

        hist_loss.append(loss_f)
        hist_gain.append(gain_f)
        hist_mean.append(mean_f)
        hist_grad.append(grad_f)

        if step % 5 == 0 or step == N_STEPS - 1:
            print(
                f"step {step:03d}: loss={loss_f:.6g}  mean_s={mean_f:.6g}  "
                f"stim_gain={gain_f:.6g}  grad={grad_f:.6g}"
            )

        updates, opt_state = opt.update(grad, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        theta = jnp.clip(theta, GAIN_MIN, GAIN_MAX)

    # Final metrics after last update
    loss_final, grad_final = value_and_grad(theta)
    mean_final = float(mean_s_fn(theta))
    loss_final_f = float(loss_final)
    grad_final_f = float(grad_final)
    gain_final = float(theta)

    if not (np.isfinite(loss_final_f) and np.isfinite(grad_final_f)):
        print("FAIL: final loss/grad not finite")
        return 1

    out_path = Path("figures") / "train_stim_gain.png"
    _plot_curves(hist_loss, hist_gain, hist_mean, out_path)

    print("--- summary ---")
    print(f"saved: {out_path.resolve()}")
    print(
        f"init:  loss={hist_loss[0]:.6g}  mean_s={hist_mean[0]:.6g}  "
        f"stim_gain={hist_gain[0]:.6g}"
    )
    print(
        f"final: loss={loss_final_f:.6g}  mean_s={mean_final:.6g}  "
        f"stim_gain={gain_final:.6g}  grad={grad_final_f:.6g}"
    )
    ratio = loss_final_f / max(hist_loss[0], 1e-16)
    print(f"loss_final / loss_init = {ratio:.4g}")

    moved = abs(mean_final - TARGET) < abs(hist_mean[0] - TARGET)
    halved = loss_final_f <= 0.5 * hist_loss[0]
    if halved or moved:
        print("OK: loss decreased and/or mean_s moved toward target")
        return 0
    print("WARN: loss did not clearly decrease; try lr/target")
    return 1


def _plot_curves(
    losses: list[float],
    gains: list[float],
    means: list[float],
    out_path: Path,
) -> None:
    steps = np.arange(len(losses))
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, constrained_layout=True)

    axes[0].plot(steps, losses, color="C0", lw=1.5)
    axes[0].set_ylabel("loss")
    axes[0].set_yscale("log")
    axes[0].set_title(r"Phase 2b: Optax train $\mathrm{stim\_gain}$")

    axes[1].plot(steps, gains, color="C1", lw=1.5)
    axes[1].set_ylabel("stim_gain")

    axes[2].plot(steps, means, color="C2", lw=1.5, label="mean_s")
    axes[2].axhline(TARGET, color="0.3", ls="--", lw=1.0, label=f"target={TARGET}")
    axes[2].set_ylabel("mean_s")
    axes[2].set_xlabel("step")
    axes[2].legend(loc="best", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

"""Phase 2a: jax.grad smoke on the Phase-1 point neuron (no Optax / no training loop).

Run from the repo root::

    python scripts/01_grad_smoke.py

Differentiates a scalar ``stim_gain`` (I_ext = stim_gain * I0) through the fixed-step
scan. Dummy loss: L = (mean_t sigmoid(k*(V-V_th)) - target)^2.
"""

from __future__ import annotations

import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jnpnp.models.point_sim import (
    dummy_soft_spike_loss,
    soft_spike,
    simulate_trajectory,
)

# θ = stim_gain (scalar); I_ext_effective = stim_gain * I0
STIM_GAIN0 = 1.0
TARGET = 0.05
V_TH = 0.0
K_SOFT = 0.5
# Spike timing makes large-ε FD noisy; 1e-6 matches AD on this trajectory.
FD_EPS = 1e-6


def loss_fn(stim_gain: jnp.ndarray) -> jnp.ndarray:
    return dummy_soft_spike_loss(
        stim_gain,
        target=TARGET,
        V_th=V_TH,
        k=K_SOFT,
        fix_a=False,
    )


def finite_diff_grad(stim_gain: float, eps: float = FD_EPS) -> float:
    """Central difference dL/dθ ≈ (L(θ+ε) - L(θ-ε)) / (2ε)."""
    lp = float(loss_fn(jnp.asarray(stim_gain + eps, dtype=jnp.float64)))
    lm = float(loss_fn(jnp.asarray(stim_gain - eps, dtype=jnp.float64)))
    return (lp - lm) / (2.0 * eps)


def main() -> int:
    theta = jnp.asarray(STIM_GAIN0, dtype=jnp.float64)
    value_and_grad = jax.value_and_grad(loss_fn)
    loss, grad = value_and_grad(theta)
    loss_f = float(loss)
    grad_f = float(grad)

    traj = simulate_trajectory(theta, fix_a=False)
    mean_s = float(jnp.mean(soft_spike(traj["V"], V_th=V_TH, k=K_SOFT)))

    print("--- Phase 2a grad smoke ---")
    print(f"theta = stim_gain = {STIM_GAIN0}")
    print(f"target = {TARGET}, V_th = {V_TH}, k = {K_SOFT}")
    print(f"mean_s = {mean_s:.6g}")
    print(f"loss = {loss_f:.6g}")
    print(f"grad = {grad_f:.6g}")

    ok_loss = bool(jnp.isfinite(loss))
    ok_grad = bool(jnp.isfinite(grad))
    print(f"finite(loss) = {ok_loss}")
    print(f"finite(grad) = {ok_grad}")

    print(f"finite-diff eps = {FD_EPS}")
    grad_fd = finite_diff_grad(STIM_GAIN0, FD_EPS)
    denom = max(abs(grad_f), abs(grad_fd), 1e-12)
    rel = abs(grad_f - grad_fd) / denom
    print(f"grad_fd = {grad_fd:.6g}")
    print(f"rel_err(|AD-FD|/max) = {rel:.6g}")

    if not (ok_loss and ok_grad):
        print("FAIL: loss or grad is not finite")
        return 1
    if rel > 0.5:
        print("WARN: AD vs FD relative error looks large; investigate if needed")
    else:
        print("OK: loss/grad finite; FD check in a reasonable ballpark")
    return 0


if __name__ == "__main__":
    sys.exit(main())

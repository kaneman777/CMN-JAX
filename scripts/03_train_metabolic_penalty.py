"""Phase 2c: toy Optax training with metabolic penalty (RQ1 preview).

Run from the repo root::

    python scripts/03_train_metabolic_penalty.py

Compares λ_met=0 (A, Phase-2b-like) vs λ_met>0 (B)::

    L = (mean_s - target)^2 + λ_met * mean_t(1 - a)

``mean(I_pump)`` was probed first but is nearly flat in ``stim_gain`` on this
toy; ``mean(1-a)`` is the fixed L_met used here so a tradeoff is visible.
θ = stim_gain only. Saves ``figures/train_metabolic_penalty.png``.
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
TARGET = 0.08
V_TH = 0.0
K_SOFT = 0.5
LR = 0.05
N_STEPS = 50
GAIN_MIN = 0.2
GAIN_MAX = 2.5

# L_met = mean(1-a). λ=0 is A; nonzero = B candidates (≤3 values total).
LAM_CANDIDATES = (0.0, 0.01, 0.02)
# Preferred B if contrast is acceptable.
LAM_B_PREFERRED = 0.02


def trajectory_metrics(
    stim_gain: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """mean_s, L_met=mean(1-a), mean(I_pump), mean(a)."""
    traj = simulate_trajectory(stim_gain, fix_a=False)
    mean_s = jnp.mean(soft_spike(traj["V"], V_th=V_TH, k=K_SOFT))
    mean_a = jnp.mean(traj["a"])
    l_met = jnp.mean(1.0 - traj["a"])
    mean_ip = jnp.mean(jnp.maximum(traj["I_pump"], 0.0))
    return mean_s, l_met, mean_ip, mean_a


def loss_parts(
    stim_gain: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mean_s, l_met, mean_ip, mean_a = trajectory_metrics(stim_gain)
    l_task = (mean_s - TARGET) ** 2
    return l_task, l_met, mean_s, mean_ip, mean_a


def make_total_loss(lam: float):
    def total_loss(stim_gain: jnp.ndarray) -> jnp.ndarray:
        l_task, l_met, _, _, _ = loss_parts(stim_gain)
        return l_task + lam * l_met

    return total_loss


def train_one(lam: float) -> dict:
    theta = jnp.asarray(STIM_GAIN0, dtype=jnp.float64)
    opt = optax.adam(LR)
    opt_state = opt.init(theta)
    loss_fn = make_total_loss(lam)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    parts_fn = jax.jit(loss_parts)

    hist: dict[str, list] = {
        "loss": [],
        "L_task": [],
        "L_met": [],
        "mean_s": [],
        "mean_Ip": [],
        "mean_a": [],
        "stim_gain": [],
        "grad": [],
    }

    print(f"\n=== train λ_met={lam} ===")
    for step in range(N_STEPS):
        loss, grad = value_and_grad(theta)
        l_task, l_met, mean_s, mean_ip, mean_a = parts_fn(theta)
        row = {
            "loss": float(loss),
            "L_task": float(l_task),
            "L_met": float(l_met),
            "mean_s": float(mean_s),
            "mean_Ip": float(mean_ip),
            "mean_a": float(mean_a),
            "stim_gain": float(theta),
            "grad": float(grad),
        }
        if not all(np.isfinite(row[k]) for k in ("loss", "grad", "mean_s", "L_met")):
            raise RuntimeError(f"non-finite at step {step}, λ={lam}: {row}")

        for k, v in row.items():
            hist[k].append(v)

        if step % 10 == 0 or step == N_STEPS - 1:
            print(
                f"step {step:03d}: L={row['loss']:.6g}  L_task={row['L_task']:.6g}  "
                f"L_met={row['L_met']:.6g}  mean_s={row['mean_s']:.6g}  "
                f"gain={row['stim_gain']:.6g}"
            )

        updates, opt_state = opt.update(grad, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        theta = jnp.clip(theta, GAIN_MIN, GAIN_MAX)

    l_task, l_met, mean_s, mean_ip, mean_a = parts_fn(theta)
    loss_f, grad_f = value_and_grad(theta)
    final = {
        "loss": float(loss_f),
        "L_task": float(l_task),
        "L_met": float(l_met),
        "mean_s": float(mean_s),
        "mean_Ip": float(mean_ip),
        "mean_a": float(mean_a),
        "stim_gain": float(theta),
        "grad": float(grad_f),
        "lam": lam,
    }
    if not (np.isfinite(final["loss"]) and np.isfinite(final["grad"])):
        raise RuntimeError(f"non-finite final for λ={lam}: {final}")
    hist["final"] = final
    hist["init"] = {
        "loss": hist["loss"][0],
        "L_task": hist["L_task"][0],
        "L_met": hist["L_met"][0],
        "mean_s": hist["mean_s"][0],
        "mean_Ip": hist["mean_Ip"][0],
        "mean_a": hist["mean_a"][0],
        "stim_gain": hist["stim_gain"][0],
    }
    return hist


def pick_B(results: dict[float, dict]) -> float:
    base = results[0.0]["final"]
    # Prefer LAM_B_PREFERRED when it shows a clear downward shift in gain or mean_s.
    pref = LAM_B_PREFERRED
    if pref in results and pref != 0.0:
        f = results[pref]["final"]
        if (f["stim_gain"] < base["stim_gain"] - 0.05) or (
            f["mean_s"] < base["mean_s"] - 0.002
        ):
            return pref
    best_lam = pref if pref in results and pref != 0.0 else None
    best_score = -1.0
    for lam, hist in results.items():
        if lam == 0.0:
            continue
        fin = hist["final"]
        # Reward lower gain / lower mean_s vs A (metabolic suppression direction).
        score = (base["stim_gain"] - fin["stim_gain"]) + 5.0 * (
            base["mean_s"] - fin["mean_s"]
        )
        if score > best_score:
            best_score = score
            best_lam = lam
    assert best_lam is not None
    return best_lam


def plot_compare(res_a: dict, res_b: dict, lam_b: float, out_path: Path) -> None:
    steps = np.arange(N_STEPS)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True, constrained_layout=True)

    axes[0].plot(steps, res_a["loss"], color="C0", lw=1.5, label=r"A: $\lambda=0$")
    axes[0].plot(steps, res_b["loss"], color="C3", lw=1.5, label=rf"B: $\lambda={lam_b}$")
    axes[0].set_ylabel("L total")
    axes[0].set_yscale("log")
    axes[0].set_title(r"Phase 2c: $L=L_{\mathrm{task}}+\lambda_{\mathrm{met}}\,\mathrm{mean}(1-a)$")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(steps, res_a["stim_gain"], color="C0", lw=1.5, label="A")
    axes[1].plot(steps, res_b["stim_gain"], color="C3", lw=1.5, label="B")
    axes[1].set_ylabel("stim_gain")
    axes[1].legend(loc="best", fontsize=8)

    axes[2].plot(steps, res_a["mean_s"], color="C0", lw=1.5, label="A mean_s")
    axes[2].plot(steps, res_b["mean_s"], color="C3", lw=1.5, label="B mean_s")
    axes[2].axhline(TARGET, color="0.35", ls="--", lw=1.0, label=f"target={TARGET}")
    axes[2].set_ylabel("mean_s")
    axes[2].set_xlabel("step")
    axes[2].legend(loc="best", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_table(results: dict[float, dict], lam_b: float) -> None:
    print("\n--- A/B summary (init -> final) ---")
    header = (
        f"{'cond':<8} {'lam':>6} {'loss':>24} {'L_task':>24} {'L_met':>16} "
        f"{'mean_s':>14} {'gain':>12} {'mean_Ip':>10} {'mean_a':>8}"
    )
    print(header)
    for name, lam in (("A", 0.0), ("B", lam_b)):
        h = results[lam]
        i, f = h["init"], h["final"]
        print(
            f"{name:<8} {lam:6.3g} "
            f"{i['loss']:10.4g}->{f['loss']:10.4g}  "
            f"{i['L_task']:10.4g}->{f['L_task']:10.4g}  "
            f"{i['L_met']:6.4f}->{f['L_met']:6.4f}  "
            f"{i['mean_s']:6.4f}->{f['mean_s']:6.4f}  "
            f"{i['stim_gain']:5.3f}->{f['stim_gain']:5.3f}  "
            f"{f['mean_Ip']:10.4f}  {f['mean_a']:8.4f}"
        )


def main() -> int:
    print("--- Phase 2c metabolic penalty ---")
    print("L_met = mean(1 - a)  [mean(I_pump) flat in stim_gain on this toy]")
    print(f"target={TARGET}, lr={LR}, steps={N_STEPS}, adam, λ candidates={LAM_CANDIDATES}")

    results: dict[float, dict] = {}
    for lam in LAM_CANDIDATES:
        results[lam] = train_one(lam)

    lam_b = pick_B(results)
    out_path = Path("figures") / "train_metabolic_penalty.png"
    plot_compare(results[0.0], results[lam_b], lam_b, out_path)
    print_table(results, lam_b)

    fa, fb = results[0.0]["final"], results[lam_b]["final"]
    print(f"\nsaved: {out_path.resolve()}")
    print(f"primary B: λ_met={lam_b}")
    print(
        f"Δfinal: mean_s(B-A)={fb['mean_s']-fa['mean_s']:+.5f}, "
        f"gain(B-A)={fb['stim_gain']-fa['stim_gain']:+.5f}, "
        f"L_met(B-A)={fb['L_met']-fa['L_met']:+.5f}"
    )
    for lam, h in results.items():
        if lam in (0.0, lam_b):
            continue
        f = h["final"]
        print(
            f"extra λ={lam}: mean_s={f['mean_s']:.5f}, gain={f['stim_gain']:.5f}, "
            f"L_met={f['L_met']:.5f}, L_task={f['L_task']:.5g}"
        )
    print("OK: A/B finished finite; figure written")
    return 0


if __name__ == "__main__":
    sys.exit(main())

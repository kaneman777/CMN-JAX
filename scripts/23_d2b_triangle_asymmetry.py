"""Phase D2b — tiny asymmetry on the complete triangle: does sync break?

Run from the repo root::

    python scripts/23_d2b_triangle_asymmetry.py

Base: D2 K3, |w|=8, D=0, K_ATP/gNa off, same drive / τ as D2.
Perturbations (one factor at a time; fixed, reproducible):
  - V0: (−65, −65+ε, −65), ε ∈ {0.1, 0.5, 1.0}
  - w: only |w13|=8+δ (δ ∈ {0.1, 0.5}), other edges −8
Control: ε=0, δ=0 (D2-like).

Reports rates + simple sync proxies (lag-0 corr, mean |Vi−Vj|).
Does **not** claim winner-take-all or confirmed alternation.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jnpnp.models.point_sim import DT, I_ext_step, merge_params
from jnpnp.models.three_cell import simulate_three_cell_chain

T_END = 500.0
T_ON = 20.0
T_OFF = 480.0
I_DRIVE = 9.0
TAU_SYN = 8.0
V0_BASE = -65.0
A0 = 1.0
G = 8.0

EPS_LIST = (0.1, 0.5, 1.0)
DELTA_LIST = (0.1, 0.5)


def make_drive() -> jnp.ndarray:
    t = jnp.arange(0.0, T_END + DT / 2, DT)
    return I_ext_step(t, T_ON, T_OFF, I_DRIVE)


def sync_metrics(tr: dict) -> dict[str, float]:
    """Lag-0 correlations and mean absolute voltage gaps on the drive window."""
    t = np.asarray(tr["t"])
    mask = (t >= T_ON) & (t < T_OFF)
    v1 = np.asarray(tr["V1"])[mask]
    v2 = np.asarray(tr["V2"])[mask]
    v3 = np.asarray(tr["V3"])[mask]

    def corr0(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 1.0 if np.allclose(a, b) else 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def peak_lag_ms(a: np.ndarray, b: np.ndarray, max_lag_steps: int = 80) -> float:
        """Peak cross-corr lag in ms (positive ⇒ b lags a)."""
        a0 = a - a.mean()
        b0 = b - b.mean()
        if a0.std() < 1e-12 or b0.std() < 1e-12:
            return 0.0
        best_lag = 0
        best_c = -np.inf
        for lag in range(-max_lag_steps, max_lag_steps + 1):
            if lag >= 0:
                aa, bb = a0[: len(a0) - lag or None], b0[lag:]
            else:
                aa, bb = a0[-lag:], b0[: len(b0) + lag or None]
            if len(aa) < 10:
                continue
            c = float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-15))
            if c > best_c:
                best_c = c
                best_lag = lag
        return best_lag * DT

    mad12 = float(np.mean(np.abs(v1 - v2)))
    mad13 = float(np.mean(np.abs(v1 - v3)))
    mad23 = float(np.mean(np.abs(v2 - v3)))
    return {
        "corr12": corr0(v1, v2),
        "corr13": corr0(v1, v3),
        "corr23": corr0(v2, v3),
        "corr_mean": float(np.mean([corr0(v1, v2), corr0(v1, v3), corr0(v2, v3)])),
        "mad_mean": float(np.mean([mad12, mad13, mad23])),
        "lag12_ms": peak_lag_ms(v1, v2),
        "lag13_ms": peak_lag_ms(v1, v3),
        "lag23_ms": peak_lag_ms(v2, v3),
    }


def rate_metrics(tr: dict) -> dict[str, float]:
    t = np.asarray(tr["t"])
    mask = (t >= T_ON) & (t < T_OFF)
    r1 = float(np.asarray(tr["s1"])[mask].mean())
    r2 = float(np.asarray(tr["s2"])[mask].mean())
    r3 = float(np.asarray(tr["s3"])[mask].mean())
    rates = np.array([r1, r2, r3])
    return {
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "ratio": float(rates.max() / max(rates.min(), 1e-9)),
        "Vmin": float(
            min(
                np.asarray(tr["V1"]).min(),
                np.asarray(tr["V2"]).min(),
                np.asarray(tr["V3"]).min(),
            )
        ),
    }


def run_case(
    I: jnp.ndarray,
    *,
    v0: tuple[float, float, float],
    w13: float,
    w31: float | None = None,
    tag: str,
) -> dict:
    w = -G
    if w31 is None:
        w31 = w13
    tr = simulate_three_cell_chain(
        I,
        w12=w,
        w21=w,
        w23=w,
        w32=w,
        w13=w13,
        w31=w31,
        V0_1=v0[0],
        V0_2=v0[1],
        V0_3=v0[2],
        a0_1=A0,
        a0_2=A0,
        a0_3=A0,
        tau_syn=TAU_SYN,
        params=merge_params(),
    )
    m = {**rate_metrics(tr), **sync_metrics(tr), "tag": tag, "tr": tr}
    return m


def observe_label(m: dict) -> str:
    """Factual bucket only — not WTA / alternation claims."""
    sync_tight = m["corr_mean"] > 0.995 and m["mad_mean"] < 0.05
    rate_split = m["ratio"] >= 1.08
    phase_shift = max(abs(m["lag12_ms"]), abs(m["lag13_ms"]), abs(m["lag23_ms"])) > 0.5
    if sync_tight and not rate_split:
        return "sync_maintained"
    if rate_split and m["mad_mean"] > 0.5:
        return "rate_split"
    if phase_shift or m["mad_mean"] > 0.05:
        return "small_phase_or_gap"
    return "near_sync"


def plot_results(rows: list[dict], out_path: Path) -> None:
    # pick control + one mid V0 + one mid w for traces
    ctrl = next(r for r in rows if r["tag"] == "ctrl")
    v05 = next(r for r in rows if r["tag"] == "V0_eps=0.5")
    w05 = next(r for r in rows if r["tag"] == "w13_delta=0.5")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    def panel_v(ax, m: dict, title: str) -> None:
        tr = m["tr"]
        # zoom drive mid window for readability
        t = np.asarray(tr["t"])
        mask = (t >= 100.0) & (t < 200.0)
        for k, c in (("V1", "C0"), ("V2", "C3"), ("V3", "C2")):
            ax.plot(t[mask], np.asarray(tr[k])[mask], lw=0.7, color=c, label=k)
        ax.set_ylabel("V [mV]")
        ax.set_title(title)
        ax.legend(fontsize=6)

    panel_v(
        axes[0, 0],
        ctrl,
        f"ctrl  corr={ctrl['corr_mean']:.4f} mad={ctrl['mad_mean']:.3f}",
    )
    panel_v(
        axes[0, 1],
        v05,
        f"V0 ε=0.5  corr={v05['corr_mean']:.4f} mad={v05['mad_mean']:.3f}",
    )
    panel_v(
        axes[1, 0],
        w05,
        f"w13 δ=0.5  corr={w05['corr_mean']:.4f} mad={w05['mad_mean']:.3f}",
    )

    ax = axes[1, 1]
    tags = [r["tag"] for r in rows]
    xs = np.arange(len(tags))
    ax.bar(xs - 0.2, [r["corr_mean"] for r in rows], 0.4, label="corr_mean")
    ax2 = ax.twinx()
    ax2.bar(
        xs + 0.2,
        [r["ratio"] for r in rows],
        0.4,
        color="C3",
        alpha=0.7,
        label="ratio",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(tags, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("lag0 corr mean")
    ax2.set_ylabel("rate ratio")
    ax.set_title("sync vs rate contrast")
    ax.legend(fontsize=6, loc="upper left")
    ax2.legend(fontsize=6, loc="upper right")

    fig.suptitle(
        rf"D2b triangle tiny asymmetry  $|w|$={G:.0f}  "
        r"(no WTA/alternation claim)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "figures" / "d2b_triangle_asymmetry.png"
    I = make_drive()

    print(
        f"D2b triangle asymmetry: base |w|={G}, drive={I_DRIVE}, "
        f"ε={EPS_LIST}, δ={DELTA_LIST}; Y/chain-sweep off"
    )

    rows: list[dict] = []

    # control
    m = run_case(I, v0=(V0_BASE, V0_BASE, V0_BASE), w13=-G, tag="ctrl")
    m["obs"] = observe_label(m)
    rows.append(m)
    print(
        f"ctrl: r=({m['r1']:.4f},{m['r2']:.4f},{m['r3']:.4f}) "
        f"ratio={m['ratio']:.4f} Vmin={m['Vmin']:.1f} "
        f"corr={m['corr_mean']:.6f} mad={m['mad_mean']:.4f} "
        f"lag12={m['lag12_ms']:.2f}ms obs={m['obs']}"
    )

    for eps in EPS_LIST:
        tag = f"V0_eps={eps}"
        m = run_case(
            I,
            v0=(V0_BASE, V0_BASE + eps, V0_BASE),
            w13=-G,
            tag=tag,
        )
        m["obs"] = observe_label(m)
        rows.append(m)
        print(
            f"{tag}: r=({m['r1']:.4f},{m['r2']:.4f},{m['r3']:.4f}) "
            f"ratio={m['ratio']:.4f} Vmin={m['Vmin']:.1f} "
            f"corr={m['corr_mean']:.6f} mad={m['mad_mean']:.4f} "
            f"lag12={m['lag12_ms']:.2f}ms obs={m['obs']}"
        )

    for delta in DELTA_LIST:
        tag = f"w13_delta={delta}"
        # only one directed edge: |w13|=8+δ; w31 stays −8
        m = run_case(
            I,
            v0=(V0_BASE, V0_BASE, V0_BASE),
            w13=-(G + delta),
            w31=-G,
            tag=tag,
        )
        m["obs"] = observe_label(m)
        rows.append(m)
        print(
            f"{tag}: r=({m['r1']:.4f},{m['r2']:.4f},{m['r3']:.4f}) "
            f"ratio={m['ratio']:.4f} Vmin={m['Vmin']:.1f} "
            f"corr={m['corr_mean']:.6f} mad={m['mad_mean']:.4f} "
            f"lag13={m['lag13_ms']:.2f}ms obs={m['obs']}"
        )

    plot_results(rows, out)
    print(f"saved: {out}")

    # robustness one-liner from observations
    broken = [r for r in rows if r["tag"] != "ctrl" and r["obs"] != "sync_maintained"]
    if not broken:
        verdict = "sync looks robust to tested tiny V0/w perturbations"
    elif all(r["obs"] == "small_phase_or_gap" for r in broken):
        verdict = "sync slightly softens (small gap/phase) but no clear rate split"
    else:
        verdict = "at least one perturbation clearly moves away from lockstep sync"
    print(f"ROBUSTNESS: {verdict}")
    print("no WTA / alternation claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

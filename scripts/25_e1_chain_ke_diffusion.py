"""Phase E1 — 3-cell chain + edge-local Ke diffusion (no triangle bus).

Run from the repo root::

    python scripts/25_e1_chain_ke_diffusion.py

Topology 1—2—3 only. |w|=8 mutual inhibition (D1 recommended g).
Ke diffusion only on edges 1↔2 and 2↔3 (never 1–3 / K3 bus).
Asks whether the bus thins / breaks / preserves path LHL (facts only).
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jnpnp.models.point_sim import DT, I_ext_step, merge_params
from jnpnp.models.three_cell import (
    simulate_three_cell_chain,
    simulate_three_cell_chain_ke,
)

T_END = 500.0
T_ON = 20.0
T_OFF = 480.0
I_DRIVE = 9.0
TAU_SYN = 8.0
V0 = -65.0
A0 = 1.0
G = 8.0
W = (-G, -G, -G, -G)  # w12,w21,w23,w32

# B2 / C′ Ke scales
GAMMA_KE = 0.0005
TAU_K = 1000.0
K_BATH = 4.0
KE_CLIP = 20.0
# representative D* (B2 used 0.005); short hygiene list if needed
D_STAR = 0.005
D_SWEEP = (0.002, 0.005, 0.01)  # pick best unclipped LHL-preserving if D* clips hard

D1_REF = (0.0182, 0.0287, 0.0182, 1.580)


def ke_params() -> dict[str, float]:
    return merge_params(
        K_e_bath=K_BATH,
        K_o=K_BATH,
        gamma_Ke=GAMMA_KE,
        tau_K=TAU_K,
        g_KATP_bar=0.0,
        atp_gNa_floor=0.0,
    )


def make_drive(t_off: float = T_OFF, t_end: float = T_END) -> jnp.ndarray:
    t = jnp.arange(0.0, t_end + DT / 2, DT)
    return I_ext_step(t, T_ON, t_off, I_DRIVE)


def label_pattern(r1: float, r2: float, r3: float, ratio: float) -> str:
    if ratio < 1.08:
        return "flat"
    if r2 > r1 and r2 > r3 and abs(r1 - r3) < 0.25 * max(r2, 1e-9):
        return "LHL"
    return "other"


def metrics(tr: dict, *, has_ke: bool) -> dict:
    t = np.asarray(tr["t"])
    mask = (t >= T_ON) & (t < min(T_OFF, float(t[-1]) + 1e-9))
    r1 = float(np.asarray(tr["s1"])[mask].mean())
    r2 = float(np.asarray(tr["s2"])[mask].mean())
    r3 = float(np.asarray(tr["s3"])[mask].mean())
    rates = np.array([r1, r2, r3])
    ratio = float(rates.max() / max(rates.min(), 1e-9))
    vmin = float(
        min(
            np.asarray(tr["V1"]).min(),
            np.asarray(tr["V2"]).min(),
            np.asarray(tr["V3"]).min(),
        )
    )
    out = {
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "ratio": ratio,
        "pattern": label_pattern(r1, r2, r3, ratio),
        "Vmin": vmin,
        "Ke_clipped": False,
        "Ke_max": (float("nan"),) * 3,
        "Ke_clip_t": None,
    }
    if has_ke:
        ke1 = np.asarray(tr["K_e1"])
        ke2 = np.asarray(tr["K_e2"])
        ke3 = np.asarray(tr["K_e3"])
        out["Ke_max"] = (float(ke1.max()), float(ke2.max()), float(ke3.max()))
        clipped = (
            ke1.max() >= KE_CLIP - 0.05
            or ke2.max() >= KE_CLIP - 0.05
            or ke3.max() >= KE_CLIP - 0.05
        )
        out["Ke_clipped"] = bool(clipped)
        if clipped:
            hit = np.where(
                (ke1 >= KE_CLIP - 0.05)
                | (ke2 >= KE_CLIP - 0.05)
                | (ke3 >= KE_CLIP - 0.05)
            )[0]
            out["Ke_clip_t"] = float(t[hit[0]]) if len(hit) else None
    return out


def run_syn_no_ke(I: jnp.ndarray) -> tuple[dict, dict]:
    tr = simulate_three_cell_chain(
        I,
        w12=W[0],
        w21=W[1],
        w23=W[2],
        w32=W[3],
        V0_1=V0,
        V0_2=V0,
        V0_3=V0,
        a0_1=A0,
        a0_2=A0,
        a0_3=A0,
        tau_syn=TAU_SYN,
        params=merge_params(),
    )
    return tr, metrics(tr, has_ke=False)


def run_ke(
    I: jnp.ndarray,
    *,
    D: float,
    syn: bool,
    ion_on: bool,
) -> tuple[dict, dict]:
    w = W if syn else (0.0, 0.0, 0.0, 0.0)
    tr = simulate_three_cell_chain_ke(
        I,
        w12=w[0],
        w21=w[1],
        w23=w[2],
        w32=w[3],
        D_ke=D,
        ion_on=ion_on,
        V0_1=V0,
        V0_2=V0,
        V0_3=V0,
        a0_1=A0,
        a0_2=A0,
        a0_3=A0,
        tau_syn=TAU_SYN,
        params=ke_params(),
    )
    return tr, metrics(tr, has_ke=True)


def plot_results(
    *,
    rows: list[dict],
    traces: dict[str, dict],
    d_star: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    ax = axes[0, 0]
    tr = traces["D0"]
    for k, c in (("V1", "C0"), ("V2", "C3"), ("V3", "C2")):
        ax.plot(tr["t"], tr[k], lw=0.6, color=c, label=k)
    ax.axvspan(T_ON, T_OFF, color="0.9", zorder=0)
    ax.set_ylabel("V [mV]")
    ax.set_title("D=0 (ion clamp): V")
    ax.legend(fontsize=6)

    ax = axes[0, 1]
    tr = traces["Dstar"]
    for k, c in (("V1", "C0"), ("V2", "C3"), ("V3", "C2")):
        ax.plot(tr["t"], tr[k], lw=0.6, color=c, label=k)
    ax.axvspan(T_ON, T_OFF, color="0.9", zorder=0)
    ax.set_ylabel("V [mV]")
    ax.set_title(f"D={d_star}: V")
    ax.legend(fontsize=6)

    ax = axes[1, 0]
    tr = traces["Dstar"]
    for k, c in (("K_e1", "C0"), ("K_e2", "C3"), ("K_e3", "C2")):
        ax.plot(tr["t"], tr[k], lw=1.0, color=c, label=k)
    ax.axhline(KE_CLIP, color="r", ls=":", lw=0.9)
    ax.axhline(K_BATH, color="0.4", ls=":", lw=0.7)
    ax.set_ylabel(r"$K_e$ [mM]")
    ax.set_title(f"D={d_star}: Ke (chain edges only)")
    ax.legend(fontsize=6)

    ax = axes[1, 1]
    names = [r["name"] for r in rows]
    xs = np.arange(len(names))
    ax.bar(xs - 0.25, [r["r1"] for r in rows], 0.25, label="r1")
    ax.bar(xs, [r["r2"] for r in rows], 0.25, label="r2")
    ax.bar(xs + 0.25, [r["r3"] for r in rows], 0.25, label="r3")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("mean soft-spike")
    ax.set_title("Rates")
    ax.legend(fontsize=6)

    fig.suptitle(
        rf"E1 chain + Ke  $|w|$={G:.0f}  "
        rf"$\gamma_{{\mathrm{{Ke}}}}$={GAMMA_KE}, $\tau_K$={TAU_K}  "
        r"(no triangle bus)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "figures" / "e1_chain_ke_diffusion.png"
    I = make_drive()

    print(
        f"E1 chain+Ke: |w|={G}, edges 12/23 only, "
        f"γ_Ke={GAMMA_KE}, τ_K={TAU_K}, bath={K_BATH}, "
        f"D* candidate={D_STAR}, drive={T_ON}-{T_OFF}"
    )
    print("no triangle Ke / no 1–3 diffusion")

    rows: list[dict] = []
    traces: dict[str, dict] = {}

    # D1 synaptic reference (no Ke state)
    tr, m = run_syn_no_ke(I)
    row = {"name": "D1_syn", **m}
    rows.append(row)
    print(
        f"D1_syn (no Ke): r=({m['r1']:.4f},{m['r2']:.4f},{m['r3']:.4f}) "
        f"ratio={m['ratio']:.3f} {m['pattern']} Vmin={m['Vmin']:.1f} "
        f"(ref LHL {D1_REF})"
    )

    # D=0 with Ke API but bath clamp → should match D1
    tr, m = run_ke(I, D=0.0, syn=True, ion_on=False)
    row = {"name": "D=0", **m}
    rows.append(row)
    traces["D0"] = tr
    print(
        f"D=0 ion_off: r=({m['r1']:.4f},{m['r2']:.4f},{m['r3']:.4f}) "
        f"ratio={m['ratio']:.3f} {m['pattern']} Vmin={m['Vmin']:.1f}"
    )

    # pick D*: prefer unclipped; keep LHL if possible
    candidates: list[tuple[float, dict, dict]] = []
    for d in D_SWEEP:
        tr_d, m_d = run_ke(I, D=float(d), syn=True, ion_on=True)
        candidates.append((float(d), tr_d, m_d))
        print(
            f"sweep D={d}: r=({m_d['r1']:.4f},{m_d['r2']:.4f},{m_d['r3']:.4f}) "
            f"ratio={m_d['ratio']:.3f} {m_d['pattern']} "
            f"Ke_max={m_d['Ke_max']} clip={m_d['Ke_clipped']} "
            f"clip_t={m_d['Ke_clip_t']}"
        )

    unclip_lhl = [
        c for c in candidates if (not c[2]["Ke_clipped"]) and c[2]["pattern"] == "LHL"
    ]
    unclip = [c for c in candidates if not c[2]["Ke_clipped"]]
    if unclip_lhl:
        d_star, tr_star, m_star = min(unclip_lhl, key=lambda c: abs(c[0] - D_STAR))
    elif unclip:
        d_star, tr_star, m_star = min(unclip, key=lambda c: abs(c[0] - D_STAR))
    else:
        d_star, tr_star, m_star = min(candidates, key=lambda c: abs(c[0] - D_STAR))

    row = {"name": f"D={d_star}", **m_star}
    rows.append(row)
    traces["Dstar"] = tr_star
    print(
        f"CHOSEN D*={d_star}: r=({m_star['r1']:.4f},{m_star['r2']:.4f},{m_star['r3']:.4f}) "
        f"ratio={m_star['ratio']:.3f} {m_star['pattern']} "
        f"Ke_max={m_star['Ke_max']} clip={m_star['Ke_clipped']}"
    )

    # optional bus-only
    tr_b, m_b = run_ke(I, D=d_star, syn=False, ion_on=True)
    row = {"name": "bus_only", **m_b}
    rows.append(row)
    print(
        f"bus_only D={d_star}: r=({m_b['r1']:.4f},{m_b['r2']:.4f},{m_b['r3']:.4f}) "
        f"ratio={m_b['ratio']:.3f} {m_b['pattern']} clip={m_b['Ke_clipped']}"
    )

    plot_results(rows=rows, traces=traces, d_star=d_star, out_path=out)
    print(f"saved: {out}")

    m0 = next(r for r in rows if r["name"] == "D=0")
    if m0["pattern"] == "LHL" and m_star["pattern"] == "LHL":
        fate = "LHL survived with edge Ke diffusion (may be thinner/stronger — see ratio)"
    elif m0["pattern"] == "LHL" and m_star["pattern"] != "LHL":
        fate = "LHL thinned/broke under D>0 (see pattern)"
    else:
        fate = "D=0 LHL check unexpected — inspect params"
    print(f"LHL FATE: {fate}")
    print("no claim: diffusion explains roles / optimal hybrid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

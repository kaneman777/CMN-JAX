"""Phase E1b — short-drive hygiene for bus_only Ke clip on the chain.

Run from the repo root::

    python scripts/26_e1b_chain_ke_short.py

Same 1—2—3 topology / γ_Ke / τ_K / D*=0.005 as E1. Does **not** overturn
E1's long-window “thin LHL” conclusion — only checks whether a short drive
lets bus_only (w=0, D=D*) score **without Ke clip**, with short syn±D controls.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jnpnp.models.point_sim import DT, I_ext_step, merge_params
from jnpnp.models.three_cell import simulate_three_cell_chain_ke

T_ON = 20.0
I_DRIVE = 9.0
TAU_SYN = 8.0
V0 = -65.0
A0 = 1.0
G = 8.0
W = (-G, -G, -G, -G)
TAIL_MS = 80.0

GAMMA_KE = 0.0005
TAU_K = 1000.0
K_BATH = 4.0
KE_CLIP = 20.0
D_STAR = 0.005

# C′′′-like short widths (ms)
WIDTHS = (60.0, 80.0, 100.0, 120.0, 150.0, 200.0)


def ke_params() -> dict[str, float]:
    return merge_params(
        K_e_bath=K_BATH,
        K_o=K_BATH,
        gamma_Ke=GAMMA_KE,
        tau_K=TAU_K,
        g_KATP_bar=0.0,
        atp_gNa_floor=0.0,
    )


def make_drive(t_off: float) -> tuple[jnp.ndarray, float]:
    t_end = t_off + TAIL_MS
    t = jnp.arange(0.0, t_end + DT / 2, DT)
    return I_ext_step(t, T_ON, t_off, I_DRIVE), t_end


def label_pattern(r1: float, r2: float, r3: float, ratio: float) -> str:
    if ratio < 1.08:
        return "flat"
    if r2 > r1 and r2 > r3 and abs(r1 - r3) < 0.25 * max(r2, 1e-9):
        return "LHL"
    return "other"


def metrics(tr: dict, t_off: float) -> dict:
    t = np.asarray(tr["t"])
    mask = (t >= T_ON) & (t < t_off)
    r1 = float(np.asarray(tr["s1"])[mask].mean())
    r2 = float(np.asarray(tr["s2"])[mask].mean())
    r3 = float(np.asarray(tr["s3"])[mask].mean())
    rates = np.array([r1, r2, r3])
    ratio = float(rates.max() / max(rates.min(), 1e-9))
    ke1 = np.asarray(tr["K_e1"])
    ke2 = np.asarray(tr["K_e2"])
    ke3 = np.asarray(tr["K_e3"])
    ke_max = (float(ke1.max()), float(ke2.max()), float(ke3.max()))
    clipped = any(k >= KE_CLIP - 0.05 for k in ke_max)
    clip_t = None
    if clipped:
        hit = np.where(
            (ke1 >= KE_CLIP - 0.05)
            | (ke2 >= KE_CLIP - 0.05)
            | (ke3 >= KE_CLIP - 0.05)
        )[0]
        clip_t = float(t[hit[0]]) if len(hit) else None
    return {
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "ratio": ratio,
        "pattern": label_pattern(r1, r2, r3, ratio),
        "Vmin": float(
            min(
                np.asarray(tr["V1"]).min(),
                np.asarray(tr["V2"]).min(),
                np.asarray(tr["V3"]).min(),
            )
        ),
        "Ke_max": ke_max,
        "Ke_clipped": clipped,
        "Ke_clip_t": clip_t,
    }


def run(
    I: jnp.ndarray,
    *,
    D: float,
    syn: bool,
    ion_on: bool,
    t_off: float,
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
    return tr, metrics(tr, t_off)


def plot_results(
    *,
    width: float,
    t_off: float,
    bus: dict,
    d0: dict,
    syn_d: dict,
    tr_bus: dict,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    ax = axes[0, 0]
    for k, c in (("V1", "C0"), ("V2", "C3"), ("V3", "C2")):
        ax.plot(tr_bus["t"], tr_bus[k], lw=0.65, color=c, label=k)
    ax.axvspan(T_ON, t_off, color="0.9", zorder=0)
    ax.set_ylabel("V [mV]")
    ax.set_title(f"bus_only width={width:.0f} ms")
    ax.legend(fontsize=6)

    ax = axes[0, 1]
    for k, c in (("K_e1", "C0"), ("K_e2", "C3"), ("K_e3", "C2")):
        ax.plot(tr_bus["t"], tr_bus[k], lw=1.0, color=c, label=k)
    ax.axhline(KE_CLIP, color="r", ls=":", lw=0.9)
    ax.axhline(K_BATH, color="0.4", ls=":", lw=0.7)
    ax.axvspan(T_ON, t_off, color="0.9", zorder=0)
    ax.set_ylabel(r"$K_e$ [mM]")
    ax.set_title(
        f"bus_only Ke  max={bus['Ke_max'][0]:.2f}  clip={bus['Ke_clipped']}"
    )
    ax.legend(fontsize=6)

    ax = axes[1, 0]
    rows = [("bus", bus), ("syn D=0", d0), (f"syn D={D_STAR}", syn_d)]
    xs = np.arange(len(rows))
    ax.bar(xs - 0.25, [r[1]["r1"] for r in rows], 0.25, label="r1")
    ax.bar(xs, [r[1]["r2"] for r in rows], 0.25, label="r2")
    ax.bar(xs + 0.25, [r[1]["r3"] for r in rows], 0.25, label="r3")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("mean soft-spike")
    ax.set_title("Short-window rates")
    ax.legend(fontsize=6)

    ax = axes[1, 1]
    ax.axis("off")
    txt = (
        f"T_on={T_ON}, T_off={t_off}, width={width:.0f} ms, tail={TAIL_MS}\n"
        f"D*={D_STAR}, γ_Ke={GAMMA_KE}, τ_K={TAU_K}\n\n"
        f"bus_only: r=({bus['r1']:.4f},{bus['r2']:.4f},{bus['r3']:.4f})\n"
        f"  ratio={bus['ratio']:.3f} {bus['pattern']} "
        f"clip={bus['Ke_clipped']} Ke_max={bus['Ke_max']}\n"
        f"syn D=0: r=({d0['r1']:.4f},{d0['r2']:.4f},{d0['r3']:.4f})\n"
        f"  ratio={d0['ratio']:.3f} {d0['pattern']}\n"
        f"syn+D:   r=({syn_d['r1']:.4f},{syn_d['r2']:.4f},{syn_d['r3']:.4f})\n"
        f"  ratio={syn_d['ratio']:.3f} {syn_d['pattern']} "
        f"clip={syn_d['Ke_clipped']}\n"
    )
    ax.text(0.02, 0.98, txt, va="top", family="monospace", fontsize=8)

    fig.suptitle(
        rf"E1b short-drive Ke hygiene  D*={D_STAR}  "
        r"(bus_only unclip goal; no role claim)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "figures" / "e1b_chain_ke_short.png"

    print(
        f"E1b short bus_only hygiene: D*={D_STAR}, widths={WIDTHS}, "
        f"T_on={T_ON}, γ_Ke={GAMMA_KE}, τ_K={TAU_K}"
    )
    print("E1 long-window thin-LHL conclusion is not overturned here")

    bus_rows: list[dict] = []
    for width in WIDTHS:
        t_off = T_ON + width
        I, _ = make_drive(t_off)
        tr, m = run(I, D=D_STAR, syn=False, ion_on=True, t_off=t_off)
        bus_rows.append({"width": width, "t_off": t_off, "tr": tr, **m})
        print(
            f"bus width={width:5.0f}: r=({m['r1']:.4f},{m['r2']:.4f},{m['r3']:.4f}) "
            f"ratio={m['ratio']:.3f} {m['pattern']} "
            f"Ke_max={tuple(round(x, 2) for x in m['Ke_max'])} "
            f"clip={m['Ke_clipped']} clip_t={m['Ke_clip_t']}"
        )

    unclip = [r for r in bus_rows if not r["Ke_clipped"]]
    if unclip:
        # prefer longest unclipped window (more signal) then closest to 150
        chosen = max(unclip, key=lambda r: (r["width"], -abs(r["width"] - 150.0)))
    else:
        chosen = min(bus_rows, key=lambda r: max(r["Ke_max"]))
        print("WARNING: no unclipped bus_only width in sweep; using lowest Ke_max")

    width = chosen["width"]
    t_off = chosen["t_off"]
    I, _ = make_drive(t_off)

    # controls on same short window
    tr0, d0 = run(I, D=0.0, syn=True, ion_on=False, t_off=t_off)
    trd, syn_d = run(I, D=D_STAR, syn=True, ion_on=True, t_off=t_off)

    print(
        f"CHOSEN width={width:.0f} ms (T_off={t_off:.0f}): "
        f"bus clip={chosen['Ke_clipped']} pattern={chosen['pattern']}"
    )
    print(
        f"  bus_only: r=({chosen['r1']:.4f},{chosen['r2']:.4f},{chosen['r3']:.4f}) "
        f"ratio={chosen['ratio']:.3f} Ke_max={chosen['Ke_max']}"
    )
    print(
        f"  syn D=0:  r=({d0['r1']:.4f},{d0['r2']:.4f},{d0['r3']:.4f}) "
        f"ratio={d0['ratio']:.3f} {d0['pattern']}"
    )
    print(
        f"  syn+D:    r=({syn_d['r1']:.4f},{syn_d['r2']:.4f},{syn_d['r3']:.4f}) "
        f"ratio={syn_d['ratio']:.3f} {syn_d['pattern']} clip={syn_d['Ke_clipped']}"
    )

    plot_results(
        width=width,
        t_off=t_off,
        bus=chosen,
        d0=d0,
        syn_d=syn_d,
        tr_bus=chosen["tr"],
        out_path=out,
    )
    print(f"saved: {out}")

    if not chosen["Ke_clipped"]:
        note = (
            f"short drive width={width:.0f} ms: bus_only unclipped and "
            f"{chosen['pattern']} (ratio={chosen['ratio']:.3f})"
        )
    else:
        note = "no unclipped bus_only window found in tested widths"
    print(f"HYGIENE: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

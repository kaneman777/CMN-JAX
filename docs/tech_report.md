# CMN-JAX Tech Report

- **Status:** **draft freeze** 2026-11-17 (Option S; no new experiments)
- **Language:** English body; Japanese Abstract working draft kept for parity
- **Claims ceiling:** toy-level **previews** with Ask labels. Not brain reproduction, emergence proof, Adam replacement, or optimal hybrid communication.
- **Primary sources:** [`STATUS.md`](../STATUS.md), [`results.md`](results.md), [`story.md`](story.md)
- **Numbers:** prefer 2026-11-17 local re-runs when they differ slightly from older STATUS tables; **Ask judgments are not upgraded/downgraded**

Wishlist / North star: [`vision.md`](vision.md) — **not results**.

---

## Abstract

Machine-learning ANNs and biophysical KNP forward problems / metabolic SNNs have long lived on separate shelves. The former are easy to train but thin on energy and ions; the latter are high-dimensional, so organizing principles are hard to read. A controlled **minimal-coupling** frame that links the two through finite energy, ion dynamics, and local plasticity has been scarce. This work builds a differentiable **toy**: point Hodgkin–Huxley with Na⁺ accumulation, pump, and normalized ATP, plus optional local \(K_e\), simplified \(K_{\mathrm{ATP}}\), and a local learning rule. Ablation-style controls separate **previews** of how metabolic constraints and synapse / \(K_e\) diffusion affect computational dynamics. We mapped single-cell selectivity, traces, and supply stress; two-cell task-dependent scoring; three-cell topology; and edge-wise diffusion. Which line appears on the scoreboard depends on task and timescale (e.g. short co-drive syn ratio ≈ 1.13, with bus null under symmetric drive). On a path, mid-high LHL appears for \(g \approx 6\)–\(8\) (ratio ≈ 1.58) and thins under edge \(K_e\) diffusion (≈ 1.08), while bus-only does not create roles. Functional compensation under metabolic stress, large-scale emergence, and replacing Adam-style global gradients remain unmet.

<details>
<summary>Abstract — agreed Japanese working draft</summary>

機械学習の ANN と、生物物理の KNP 順問題・代謝 SNN は長く別棚だった。前者は学習しやすいがエネルギー・イオンが薄く、後者は高次元で構成原理が見えにくい。両者を有限エネルギー・イオン動態・局所可塑性の**最小結合**でつなぐ対照枠は不足していた。本研究は点 Hodgkin–Huxley に Na⁺ 蓄積・ポンプ・正規化 ATP、および任意の局所 \(K_e\)・簡約 \(K_{\mathrm{ATP}}\)・局所学習則を載せた微分可能トイを構築し、代謝制約とシナプス／\(K_e\) 拡散が計算ダイナミクスに与える**予兆**を対照実験で切り分ける。1細胞の選択・痕跡・供給ストレス、2細胞の課題依存採点、3細胞トポロジ、辺沿い拡散までを地図化した。課題の時間スケールで採点に載る回線が変わり（例: 短同時で syn ratio≈1.13、対称ではバス null）、path では \(g\)≈6–8 で中央高 LHL（ratio≈1.58）が出て辺拡散で薄まる（≈1.08）一方、バス単独は役割を作らない。代謝ストレス時の機能的肩代わり、大規模創発、Adam 的大域勾配の置き換えは未達である。

</details>

---

## 1. Purpose

Agreed goals for this prototype:

1. Implement a **toy** point neuron with metabolic / ion constraints in JAX (forward + differentiable).
2. Under minimal interventions, separate **previews** of selectivity, slow \(K_e\) traces, and supply-stress coarsening with ablation-style controls.
3. With a few cells, **map** synapse (“train”) vs \(K_e\) diffusion (“bus”), and role patterns that depend on task and topology.
4. **Out of scope as goals:** finished functional compensation, large-scale emergence, or replacing Adam / global gradients (record unmet / not claimed).

Affiliation / preliminary notice lives in [`NOTICE`](../NOTICE) and the README — not repeated as an Abstract closer.

---

## 2. Model summary

Full equations and defaults: [`model.md`](model.md). Implementations: `hh_point.py`, `ions_atp.py`, `point_sim.py`, `two_cell.py`, `three_cell.py`. **Not** a closed continuum PNP/KNP; injected \(I_{\mathrm{ext}}\), clips, and point / few-cell approximations.

### 2.1 Point cell (membrane + metabolism)

Outward-positive currents. Core ODE sketch:

\[
C_m\dot V
=
-I_{\mathrm{Na}}-I_{\mathrm{K}}-I_{\mathrm{L}}-I_{\mathrm{pump}}-I_{\mathrm{KATP}}
+I_{\mathrm{ext}}(t)+I_{\mathrm{syn}}(t)
\]

- Classical HH gates \((m,h,n)\) and Nernst \(E_{\mathrm{Na}}\) (dynamic via \(\mathrm{Na}_i\)); \(E_{\mathrm{K}}\) fixed unless local \(K_e\) is on.
- Pump (Ellingsrud-style toy) depends on \(a\), \(\mathrm{Na}_i\), and extracellular K, with Hill-like saturation in Na/K.
- Intracellular Na and normalized ATP:

\[
\dot{\mathrm{Na}}_i=-\gamma_{\mathrm{Na}}(I_{\mathrm{Na}}+3I_{\mathrm{pump}}),\qquad
\dot a=\eta_{\mathrm{prod}}(a_{\max}-a)-\eta_{\mathrm{cost}}\max(I_{\mathrm{pump}},0)
\]

### 2.2 Optional fast metabolic brakes

- **Phenomenological ATP→gNa** (RQ3): \(I_{\mathrm{Na}}\leftarrow I_{\mathrm{Na}}\bigl(f+(1-f)a\bigr)\) when enabled.
- **Simplified \(K_{\mathrm{ATP}}\)** (Phase 4-Bio; default \(\bar g=0\)): Hill open fraction

\[
\sigma(a)=\frac{1}{1+(a/a_{1/2})^{n}},\qquad
I_{\mathrm{KATP}}=\bar g_{\mathrm{KATP}}\,\sigma(a)\,(V-E_{\mathrm{K}})
\]

with typical \(a_{1/2}=0.2\), \(n=2\). Toy brake only — not Kir6.x/SUR fidelity.

### 2.3 Dual lines (train / bus)

**Train (fast synaptic current).** Soft spike \(s_j\) filtered by \(\tau_{\mathrm{syn}}\):

\[
\dot{\bar s}_j=-\bar s_j/\tau_{\mathrm{syn}}+s_j,\qquad
I_{\mathrm{syn},i}=w_{i\leftarrow j}\,\bar s_j
\]

**Bus (slow local \(K_e\)).** Narrow cleft pool plus optional discrete diffusion between neighbors (not a 1D tissue PDE):

\[
\dot K_{e,i}
=
\gamma_{Ke}(I_{\mathrm{K},i}-2I_{\mathrm{pump},i})
-\frac{K_{e,i}-K_{e,\mathrm{bath}}}{\tau_K}
+D\,(K_{e,j}-K_{e,i})
\]

Three-cell E1 uses the same edge form on path edges 1—2 and 2—3 only (no triangle Ke bus).

### 2.4 Topology (wiring, not continuum geometry)

Sparse synaptic weights \(w_{ij}\) and an explicit edge set for \(D\) implement path, \(K_3\), or star. Equivalent to a graph Laplacian on those edges, but the code does **not** introduce a general topology-matrix API — edges are wired per experiment.

### 2.5 Numerics

| Item | Choice |
|------|--------|
| Time | Fixed \(\Delta t\) (smoke default \(0.025\,\mathrm{ms}\)) |
| \(V\), \(\mathrm{Na}_i\), \(a\), \(K_e\) | Forward Euler (+ clips) |
| HH gates | Exponential Euler |
| Loop | JAX `lax.scan`; `jax_enable_x64=True` |
| Diffrax / adaptive ODE solvers | Not required on the mainline |

### 2.6 Notation (short)

| Symbol | Meaning |
|--------|---------|
| \(V,m,h,n\) | Membrane potential and HH gates |
| \(\mathrm{Na}_i,a,a_{\max}\) | Internal Na; normalized ATP; supply ceiling |
| \(K_e,D,\tau_K\) | Local extracellular K; discrete diffusion rate; bath recovery |
| \(I_{\mathrm{pump}},I_{\mathrm{KATP}},\sigma(a)\) | Pump; simplified K_ATP; Hill open fraction |
| \(I_{\mathrm{syn}},w_{i\leftarrow j},\tau_{\mathrm{syn}}\) | Synaptic drive; weight; synaptic filter time |
| \(s,\bar s\) | Soft spike; filtered synaptic activation |

Optional local ATP three-factor learning / homeo (\(\dot e_i\), \(g(a)\), \(\Delta w\)): [`model.md`](model.md) §8d — supporting results, not required to read the C′/D/E cores.

---

## 3. Experiment map

| Stage | What we ask | Outcome (Ask label) |
|-------|-------------|---------------------|
| Toy core | Can a point HH neuron with ATP/ion constraints run forward and through `jax.grad` / toy Optax? | OK (Phase 1–2) |
| RQ1–3 | Do selectivity, slow \(K_e\) traces, and supply-stress coarsening appear under minimal interventions? | Preview; RQ3 pump-only path is **null** (needs a fast brake) |
| Train vs bus | Can a two-cell synapse (“train”) and two-point \(K_e\) diffusion (“bus”) each show a minimal effect? | Both demos OK; structural asymmetry for train; compensation remains weak / partial |
| Task scoring | On the same toy, which line shows up in the scoreboard under cue–probe vs co-drive / short / asymmetric drive? | Task- and timescale-dependent (C′ family); not an optimal hybrid claim |
| 3-cell topology | At matched \|w\|, does path differ from complete \(K_3\)? | Path → LHL (structural); \(K_3\) → uniform, sync mostly robust |
| Path + edge \(K_e\) | Does edge-wise \(K_e\) diffusion erase path LHL? | LHL remains but thins (~1.58→~1.08); bus_only stays flat (short window unclipped) |

*Footnote:* D3 four-cell star is off-mainline (flat). Plan-A compensation not run. Vision items are out of scope.

---

## 4. Results

**Reading note (Option S):** Main benchmarks are the C′ family, topology (D), and path+edge \(K_e\) (E). RQ / local-learning blocks are supporting. Ask labels are frozen.

### 4.0 Supporting benchmarks (RQ / brakes / local rule)

Supporting only — not the Option S headline. Ask labels frozen at **preview** / **null** / **partial** as recorded in STATUS.

| Block | Script / figure | Ask outcome (1 line) | Ceiling |
|-------|-----------------|----------------------|---------|
| RQ1 | `04` · [`rq1_selective_metabolic.png`](../figures/rq1_selective_metabolic.png) | λ>0 trims noise weight; mild metabolic savings; task not abandoned | Not sparse-coding emergence |
| RQ2 | `05` · [`rq2_ke_trace.png`](../figures/rq2_ke_trace.png), [`rq2_delay_probe.png`](../figures/rq2_delay_probe.png) | Cue–delay–probe separation ≈ +0.034 (ion-on) vs ≈0 (ion-off) | Not working memory |
| RQ3 | `06` · [`rq3_atp_stress.png`](../figures/rq3_atp_stress.png) | With ATP→gNa gate: weak-first coarsening; **pump-only = null** on this scale | Not phase transition / forgetting |
| 4-Bio | `07` · [`rq4_katp_*.png`](../figures/rq4_katp_rq3_compare.png) | A flat null; B (gNa) and C (Hill \(K_{\mathrm{ATP}}\)) weak-first | Not molecular K_ATP |
| 4-Learn | `08` · [`rq4_local_vs_adam.png`](../figures/rq4_local_vs_adam.png) | Pure local: selectivity preview (N↓); **savings favor Adam**; **w1 hits clip** | Not Adam/BPTT replacement |
| A1 homeo | `09` · [`rq4_local_homeo.png`](../figures/rq4_local_homeo.png) | w1 off clip; selection kept; **not pure local**; savings still short of Adam | Not cortical homeostasis |

Local 2026-11-17 re-run finals (same Ask reading): Adam \(w\approx(0.560,0.091)\); Local \(w=(2.500,0.243)\); Local+homeo \(w\approx(0.661,0.047)\), mean_S≈0.099, mean_N≈0.009.

**Parts demos (also supporting).** B1 train asymmetry **success** under *structural* \(w\) asymmetry (ratio ≈ 1.73; symmetric coupling alone did not split). B2 bus **success** (\(D>0\) raises neighbor Ke and probe response). Figures: [`b1_two_cell_asymm.png`](../figures/b1_two_cell_asymm.png), [`b2_ke_diffusion.png`](../figures/b2_ke_diffusion.png).

### 4.1 Core — Train / bus coexistence and task scoring (C′ family)

**Lead.** Scripts `15`–`18` (entry coexistence: `15_cprime_syn_plus_ke.py`). Same toy can carry synaptic current and \(K_e\) diffusion together; which line dominates the **scoreboard** depends on the protocol.

#### What we can say

- Synapse and \(K_e\) diffusion can coexist in one toy (C′ / `15`).
- **In this contrast set:** cue→probe scores the **bus** (diffusion) as the main effect (C′; both≈diff); longer co-drive puts the **train** (synaptic current) on the board (C′′; both≈syn), with Ke often **clipped** on the long window. Even when shortened, **symmetric** co-drive is train-only: the bus stays hard to see because no Ke gradient forms (C′′′).
- Bus invisible ≠ bus principle-invalid (long-drive Ke saturation; zero gradient under symmetric drive).
- Mild drive asymmetry can open a short-window bus Ke gap, but scoring remains train-leaning (C′′′ asym; **partial success**).
- “Trunk = current; diffusion matters in asymmetric / slow-window settings” is a map-consistent **intuition**, not a proof.

#### Four-way scoreboard (C′ family)

| Protocol | Dominant on scoreboard | Kernel |
|----------|------------------------|--------|
| C′ cue→probe | **Bus** | Activity asymmetry → Ke gradient |
| C′′ long co-drive | **Train** (bus clipped) | Simultaneous push–pull; Ke saturation |
| C′′′ short co-drive | **Train only** | No clip; **symmetric drive → bus null** |
| C′′′ asym | **Train + thin bus** | Asymmetry opens a bus mouth; both≈syn |

#### What we cannot say

- A general law for which line wins on arbitrary tasks; optimal hybrid operation; brain dual-line communication.
- “Short drive always shows both”; both-condition with equal train and bus strength; etc.

#### Figures

- [`figures/cprime_syn_plus_ke.png`](../figures/cprime_syn_plus_ke.png)
- [`figures/cdoubleprime_co_drive.png`](../figures/cdoubleprime_co_drive.png)
- [`figures/ctripleprime_short_codrive.png`](../figures/ctripleprime_short_codrive.png)
- [`figures/casym_short_codrive.png`](../figures/casym_short_codrive.png)
- Optional parts demos: [`figures/b1_two_cell_asymm.png`](../figures/b1_two_cell_asymm.png), [`figures/b2_ke_diffusion.png`](../figures/b2_ke_diffusion.png)

### 4.2 Core — Topology (D)

**Lead.** Scripts `20`–`23` (off-mainline star: `24`). Connected simple undirected three-cell graphs are essentially **path** vs complete \(K_3\). Same \|w\|, different role patterns.

#### What we can say

- At matched \|w\|, topology changes the rate pattern: **path → LHL** (mid high, ends low; ratio ≈ 1.58 at \(g=8\)), **\(K_3\) → uniform** (ratio = 1) with synchrony.
- Path LHL is a **structural** effect (chain degree asymmetry with identical \(V_0\) and symmetric \(w\)), not a proof of spontaneous symmetry breaking.
- Clear LHL sits in a coupling window \(g \approx 6\)–\(8\) (recommended \(g=8\)); weaker \(g\) gives a thin LHL; too-strong \(g\) flattens then can invert (HLH).
- On \(K_3\), tiny \(V_0\) / one-edge \(w\) perturbations keep sync **mostly robust**; rate splits stay near absent. Under these conditions we do **not** see clear alternation / WTA or a clean rate bifurcation (small phase/gap softens sync slightly under \(w\) perturbation — not a blanket “no vibration” claim).

#### What we cannot say

- Brain circuits, optimal topology, alternation / WTA, or a general symmetry-breaking theorem.

#### Figures

- [`figures/d1_three_cell_chain.png`](../figures/d1_three_cell_chain.png)
- [`figures/d1_chain_w_sweep.png`](../figures/d1_chain_w_sweep.png)
- [`figures/d2_three_cell_triangle.png`](../figures/d2_three_cell_triangle.png)
- [`figures/d2b_triangle_asymmetry.png`](../figures/d2b_triangle_asymmetry.png)
- Footnote only: [`figures/d3_four_cell_y.png`](../figures/d3_four_cell_y.png) (four-cell star stays **flat**; degree alone ≠ path-scale LHL; off-mainline)

### 4.3 Core — Path + edge \(K_e\) (E1 / E1b)

**Lead.** Scripts `25`–`26`. Path 1—2—3 with **edge-wise** \(K_e\) only (no triangle bus / no 1–3 diffusion). Ask-closed as a long-drive + short-drive hygiene pair.

#### What we can say

- With \(D=0\), path LHL reproduces (ratio ≈ 1.58).
- With synapse + edge \(D^\*>0\) (\(D^*=0.005\)), LHL **remains but thins** (≈ 1.08) on both long and short windows (E1b does not overturn E1).
- **bus_only** does not create roles — rates stay **flat**; long-drive bus_only can hit Ke clip, while a short window (E1b, width = 200 ms) shows **unclipped** flat bus_only.

#### What we cannot say

- Optimal hybrid operation; triangle + \(D\); that diffusion “explains” synaptic role differentiation.

#### Figures

- [`figures/e1_chain_ke_diffusion.png`](../figures/e1_chain_ke_diffusion.png)
- [`figures/e1b_chain_ke_short.png`](../figures/e1b_chain_ke_short.png)

### 4.4 Optional — Compensation limit

Not a headline result. Scripts `12`, `13`, `19`. Weights \(w=(-15,-3)\).

| Step | Ask label | Fact |
|------|-----------|------|
| ② gate off (pump+ATP only) | **null** | \(a_{\max1}\downarrow\) does not drop stressed rate; partner Δr≈0 |
| ②′ + simplified \(K_{\mathrm{ATP}}\) | **partial success** | Stressed r1 falls clearly; partner r2 does **not** rise (slight drop) |
| Plan-B plasticity (\|w21\| shrinks when cell1 ATP-poor; w12 fixed) | **partial success** | r1↓ yes; r2↑ no; P1≈P2 — after metabolic silence, \(I_{\mathrm{syn},2}\propto w_{21}s_1\approx 0\), so Plan-B has little to act on |
| Plan A (protect the weak) | **not run** | Different phenomenon; do not infer from Plan-B |

Figures: [`b1_compensation.png`](../figures/b1_compensation.png), [`b1_compensation_katp.png`](../figures/b1_compensation_katp.png), [`b1_compensation_plastic_B.png`](../figures/b1_compensation_plastic_B.png).

**Cannot say:** rehab impossible; compensation impossible in principle; Plan A would fail.

---

## 5. What we can / cannot claim (global)

Merged from [`story.md`](story.md) §3 and Ask closes for C′/D/E. No judgment upgrades.

### Can say

- Metabolic / ion constraints fit in one differentiable point-neuron toy (forward + `jax.grad` / toy Optax).
- RQ1 selectivity–savings, RQ2 slow \(K_e\) trace, RQ3 supply-stress coarsening are **previews** with controls; RQ3 / compensation slowdown needs a **fast brake** (phenomenological gNa or simplified \(K_{\mathrm{ATP}}\)); pump-only is **null** on this scale.
- Two-cell **train** (synapse) and **bus** (\(K_e\) diffusion) each have a minimal demo; both can coexist; **which line scores depends on task / timescale / symmetry** (C′ family) — not a claim of optimal hybrid use.
- At matched \|w\|, **path → LHL** vs **\(K_3\) → uniform** (structural topology effect); edge \(K_e\) **thins but does not erase** path LHL; bus_only stays flat.
- Fixed mutual inhibition is not enough for functional **compensation** previews (null / partial only).
- Reproducible prototype map — a research **seed**, not a finished theory.

### Cannot say

- Reproduction of brain coding, memory, forgetting, energy saving, or rehab.
- Proof of emergence or phase transitions.
- Replacement of Adam / backprop.
- Tissue diffusion, ephaptic coupling, EEG, or continuum 1D PNP.
- Synapses unnecessary; bus always dominant; optimal dual-line communication.
- Spontaneous role split from fully symmetric coupling; finished shoulder-compensation; alternation / WTA; general topology laws; triangle+\(D\) results (not run).

Vision / North star items remain in [`vision.md`](vision.md) only.

---

## 6. Reproducibility

| Item | Detail |
|------|--------|
| Env (logged) | Windows; Python ≥3.10 (dev used 3.13); `.venv`; JAX CPU `0.12` with `jax_enable_x64=True` |
| Install | `pip install -e ".[dev]"` — see [`README.md`](../README.md) |
| DOI | Concept [`10.5281/zenodo.24283139`](https://doi.org/10.5281/zenodo.24283139); Version v0.1.1 [`10.5281/zenodo.24283140`](https://doi.org/10.5281/zenodo.24283140) |
| Repo | https://github.com/kaneman777/CMN-JAX |
| Figures | Written under `figures/` by the scripts below |

**Mainline commands** (Option S cores bold):

```bash
python scripts/00_forward_smoke.py
python scripts/04_rq1_selective_metabolic.py
python scripts/05_rq2_ke_memory.py
python scripts/06_rq3_atp_stress.py
python scripts/07_rq4_katp_rq3.py
python scripts/08_rq4_local_metabolic_learning.py
python scripts/09_rq4_local_homeo.py
python scripts/10_b1_two_cell_asymm.py
python scripts/14_b2_ke_diffusion.py
python scripts/15_cprime_syn_plus_ke.py
python scripts/16_cdoubleprime_co_drive.py
python scripts/17_ctripleprime_short_codrive.py
python scripts/18_casym_short_codrive.py
python scripts/20_d1_three_cell_chain.py
python scripts/21_d1_chain_w_sweep.py
python scripts/22_d2_three_cell_triangle.py
python scripts/23_d2b_triangle_asymmetry.py
python scripts/25_e1_chain_ke_diffusion.py
python scripts/26_e1b_chain_ke_short.py
```

Optional: `12`/`13`/`19` (compensation), `24` (D3 off-mainline). Full list in README Quickstart. Dated logs: [`STATUS.md`](../STATUS.md).

---

## 7. Known doc gaps

- [`model.md`](model.md) / [`experiments.md`](experiments.md) last bulk update predate multi-cell D/E; STATUS logs remain thicker for those protocols.
- Older STATUS tables for some retrain runs (e.g. RQ1/RQ3 \(w^*\), Phase 2b/2c gains) can differ slightly from 2026-11-17 local re-runs; **directions and Ask labels unchanged**.

---

## Drafting checklist (author + agent)

| Section | State |
|---------|--------|
| Abstract EN + JA parity | **draft freeze** |
| §1 Purpose | Agreed |
| §2 Model summary | Agreed |
| §3 Experiment map | Agreed |
| §4.0–4.4 Results | Filled (Ask labels frozen) |
| §5 Global can/cannot | Filled |
| §6 Reproducibility | Filled |

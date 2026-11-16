# CMN-JAX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-TODO_after_Zenodo-lightgrey)](docs/PUBLISHING.md)
[![Status](https://img.shields.io/badge/status-preliminary-orange)](docs/story.md)

**One-liner:** Differentiable toy neuron models with ATP and ion constraints (JAX).

**Status: Preliminary independent research prototype** (undergraduate B3, solo-started 0→1).  
Public timestamp for a reproducible toy map — not a finished theory paper.

> **Not claiming:** biological reproduction of brain coding/memory/rehab, “new field” proof, or Adam/backprop replacement.  
> Findings are **toy-level previews** with ablation-style controls. See [`docs/story.md`](docs/story.md) and [`docs/results.md`](docs/results.md).

**Affiliation note:** Independent undergraduate research; **not** an official university lab project (see [`NOTICE`](NOTICE)).

Local working folder may still be named `jax-neuro-pnp`; the public GitHub repository is **[CMN-JAX](https://github.com/kaneman777/CMN-JAX)**.

---

## Quick links

| Doc | Role |
|-----|------|
| [`docs/story.md`](docs/story.md) | **Narrative of verified reach** (read this first) |
| [`docs/results.md`](docs/results.md) | Tables / phase summary (facts) |
| [`docs/survey.md`](docs/survey.md) | Literature gap & toy design |
| [`docs/model.md`](docs/model.md) | Equations & symbols |
| [`docs/experiments.md`](docs/experiments.md) | Protocols |
| [`docs/vision.md`](docs/vision.md) | Wishlist only — **not results** |
| [`docs/PUBLISHING.md`](docs/PUBLISHING.md) | GitHub Release + Zenodo DOI steps |
| [`STATUS.md`](STATUS.md) | Dated lab log |

---

## Research questions (toy scale)

| ID | Question (preview level) |
|----|--------------------------|
| **RQ1** | With Na⁺ load and pump ATP cost, do selective / cheaper response patterns appear? |
| **RQ2** | Can slow local \(K_e\) act as a short-term trace (cue→delay→probe)? |
| **RQ3** | After learning, does lowering ATP supply ceiling coarsen responses (weak-first)? |

Later map: two-cell **synapse (“train”)** vs **\(K_e\) diffusion (“bus”)**, and **task-dependent scoring** (C′ / C′′ / C′′′). Details in story/results — not hybrid-brain claims.

---

## Setup

```bash
# Python >= 3.10 recommended
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e ".[dev]"

# GPU: follow JAX install docs for CUDA builds
# https://jax.readthedocs.io/en/latest/installation.html
```

Core deps (`pyproject.toml`): `jax`, `optax`, `matplotlib`, `numpy` (plus optional `diffrax` / `equinox`).

---

## Quickstart (main scripts)

```bash
python scripts/00_forward_smoke.py           # Phase 1: forward smoke (spike→Na→ATP)
python scripts/01_grad_smoke.py              # Phase 2a: jax.grad smoke
python scripts/02_train_stim_gain.py         # Phase 2b: Optax stim_gain
python scripts/03_train_metabolic_penalty.py # Phase 2c: metabolic penalty A/B
python scripts/04_rq1_selective_metabolic.py # RQ1
python scripts/05_rq2_ke_memory.py           # RQ2 local K_e
python scripts/06_rq3_atp_stress.py          # RQ3 a_max stress
python scripts/07_rq4_katp_rq3.py            # 4-Bio simplified K_ATP
python scripts/08_rq4_local_metabolic_learning.py  # 4-Learn local vs Adam
python scripts/09_rq4_local_homeo.py         # 4-Learn A1 local+homeo
python scripts/10_b1_two_cell_asymm.py       # B1 synapse asymmetry
python scripts/11_b1_w_sweep.py              # B1 |w| hygiene sweep
python scripts/12_b1_compensation.py         # B1-② compensation (gate off)
python scripts/13_b1_compensation_katp.py    # B1-②′ + K_ATP
python scripts/14_b2_ke_diffusion.py         # B2 K_e diffusion
python scripts/15_cprime_syn_plus_ke.py      # C′ syn+Ke (cue→probe)
python scripts/16_cdoubleprime_co_drive.py   # C′′ co-drive
python scripts/17_ctripleprime_short_codrive.py  # C′′′ short co-drive
```

Figures are written under `figures/` (tracked PNGs for the public timestamp).

---

## Repository layout

```text
CMN-JAX/
├── README.md · LICENSE · NOTICE · CITATION.cff · STATUS.md
├── docs/          # story, results, survey, model, experiments, vision, PUBLISHING
├── src/jnpnp/     # HH / ions+ATP / two_cell (+ K_e diffusion)
├── scripts/       # 00–17 experiment drivers
├── figures/       # representative PNGs
└── tests/
```

---

## Citation

DOI: **TODO after Zenodo** (follow [`docs/PUBLISHING.md`](docs/PUBLISHING.md)).

Until a DOI exists, you may cite the software release:

```bibtex
@software{kaneko_cmnjax_2026,
  author       = {Kaneko, Takuma},
  title        = {{CMN-JAX}: Differentiable toy neurons with ATP and ion constraints},
  year         = {2026},
  version      = {0.1.0},
  url          = {https://github.com/kaneman777/CMN-JAX},
  note         = {Preliminary independent research prototype; DOI pending Zenodo}
}
```

Also see [`CITATION.cff`](CITATION.cff).

---

## License / Contact

- **License:** MIT — [`LICENSE`](LICENSE)
- **Author:** Takuma Kaneko ([@kaneman777](https://github.com/kaneman777))
- Issues / discussion: GitHub Issues on this repository

---

## Changelog (public)

- **2026-11-16:** Public prep P0 — MIT, story/results freeze pointer, publishing docs; intended release `v0.1.0`.
- **2026-11-14–16:** Solo prototype through RQ1–3, two-cell train/bus map, C′–C′′′ task scoring (see STATUS).

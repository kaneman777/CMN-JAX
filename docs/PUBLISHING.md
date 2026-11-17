# Publishing checklist (GitHub + Zenodo)

- **Created:** 2026-11-16
- **Repo:** https://github.com/kaneman777/CMN-JAX
- **DOI status:** **obtained** (2026-11-16)
  - Concept DOI: https://doi.org/10.5281/zenodo.24283139
  - Version DOI (v0.1.1): https://doi.org/10.5281/zenodo.24283140
  - Zenodo record: https://zenodo.org/records/24283140
  - Badge / citation live in `README.md` and `CITATION.cff`

This document remains the operational guide for a **public timestamp**. Claims stay
preliminary (see [`story.md`](story.md), [`results.md`](results.md)).
Steps below are kept for reproducibility / future version releases.

---

## 0. Preconditions

- [ ] Local tree commits cleanly (`LICENSE`, `README`, `CITATION.cff`, docs, code, tracked figures)
- [ ] GitHub account: `kaneman777`
- [ ] `gh` CLI authenticated (`gh auth login`) **or** use the GitHub website UI

---

## 1. Create / push the public GitHub repository

**Local status (2026-11-16):** `git init` done; initial commit on `main`
(`55486eb` — *Initial public release: preliminary CMN-JAX prototype v0.1.0*).
`origin` already points at `https://github.com/kaneman777/CMN-JAX.git` but the
remote repo did **not** exist yet when the Agent stopped (needs your GitHub login).

### A. Authenticate (one-time)

```powershell
# portable gh was unpacked here if winget was busy:
#   $env:LOCALAPPDATA\gh-cli\bin\gh.exe
gh auth login --hostname github.com --git-protocol https --web
```

Complete the device code in the browser as `kaneman777`.

### B. Create public repo and push

```powershell
cd path\to\jax-neuro-pnp   # local folder
gh repo create kaneman777/CMN-JAX --public --source=. --remote=origin --push
# If origin already exists:
#   gh repo create kaneman777/CMN-JAX --public
#   git push -u origin main
```

**Website alternative:** https://github.com/new → name `CMN-JAX`, **Public**,
no README/license (already in commit) → then `git push -u origin main`.

Confirm: https://github.com/kaneman777/CMN-JAX is **public**.

---

## 2. Cut GitHub Release `v0.1.0`

```bash
gh release create v0.1.0 --title "Preliminary prototype — story/results freeze" --notes-file - <<'EOF'
Preliminary independent undergraduate research prototype.

- Differentiable toy neuron with ATP + ion constraints (JAX)
- Verified toy-level map: RQ1–3 previews, two-cell synapse (train) / K_e diffusion (bus), task-dependent scoring (C′/C′′/C′′′)
- **Not claiming:** biological reproduction, field creation, Adam replacement

Primary docs:
- Story: docs/story.md
- Results: docs/results.md
- Survey: docs/survey.md
- Vision (wishlist only): docs/vision.md

Independent solo-started prototype; not a formal university lab project (see NOTICE).
EOF
```

Or: GitHub → Releases → Draft a new release → tag `v0.1.0`.

---

## 3. Zenodo DOI via GitHub (recommended)

Official docs:

- Zenodo–GitHub: https://docs.zenodo.org/docs/github/
- General upload: https://docs.zenodo.org/

### Shortest path

1. Sign in to [Zenodo](https://zenodo.org/) (or Sandbox first: https://sandbox.zenodo.org/).
2. GitHub icon → authorize Zenodo’s GitHub App for account `kaneman777`.
3. Enable the repository **CMN-JAX** in Zenodo’s GitHub listing.
4. On GitHub, (re)publish Release `v0.1.0` (or create a new patch release if needed). Zenodo archives the release and mints a DOI.
5. Copy the **Concept DOI** (version-agnostic) and/or **Version DOI** (`10.5281/zenodo.…`).

### After DOI exists — update this repo

- [ ] `README.md`: replace `TODO after Zenodo` badge / citation DOI
- [ ] `CITATION.cff`: uncomment `identifiers` and set the DOI
- [ ] `STATUS.md`: mark DOI obtained + date
- [ ] Commit + push (optional small release `v0.1.1` only if needed for Zenodo sync)

---

## 4. Next phase (optional, not required for timestamp)

1. Short **Tech Report** — **draft freeze:** [`tech_report.md`](tech_report.md) (2026-11-17). Optional later: LaTeX / arXiv polish without raising claim ceilings.
2. Optional preprint: arXiv / TechRxiv (check license + institutional requirements).
3. Keep vision.md out of the abstract’s claims.

---

## Troubleshooting

| Issue | Action |
|-------|--------|
| `gh` missing | Install [GitHub CLI](https://cli.github.com/) or use the website UI |
| Auth failed | `gh auth login` |
| Zenodo did not archive | Ensure repo is public, Zenodo toggle ON, then make a **new** release |
| Large files | Keep `.venv/`, PDFs, `outputs/` gitignored; figures PNGs are intentionally tracked |

# RL_MFPT: repo findings (2026-11-17)

Start here. This is the index of the cleanup-and-survey pass over `TW_working_branch`. Nothing has been committed yet.

| Doc | What's in it |
|---|---|
| [`CLEANUP_LOG.md`](CLEANUP_LOG.md) | Every file removed, why it was safe, and how to undo |
| [`ENVIRONMENT.md`](ENVIRONMENT.md) | The removed venv, the stale `environment.yml`, the real dependency census, and a proposed env |
| [`findings/01-langevin-pipeline.md`](findings/01-langevin-pipeline.md) | `1D_langevin_sim`, `1D_langevin_sim_backforth`, `2D_langevin_sim` |
| [`findings/02-openmm-systems.md`](findings/02-openmm-systems.md) | NaCl, PMg, alanine dipeptide (+ their `_backforth` forks) |
| [`findings/03-core-modules-and-paper.md`](findings/03-core-modules-and-paper.md) | Version drift of `MSM.py`/`dham.py`/`util.py`/`wham.py`, algorithm correctness, `paper_publication/` |
| [`findings/04-rl-experiments.md`](findings/04-rl-experiments.md) | All `RL_*` folders, `runs/`, `.pt` models and datasets |
| [`findings/05-model-potential-prototypes.md`](findings/05-model-potential-prototypes.md) | MATLAB 1D/2D prototypes, `1D/2D_model_potential_TW`, `CE_Opt` |

The `findings/*` docs were written by area-specific review agents and spot-checked by me (the items marked ✔ below).
They cite `file:line` and label every claim **[Verified]** or **[Inferred]**. Nothing was executed: no numpy/OpenMM in this
environment, and the scripts write into tracked directories. Some files they mention were removed afterwards; see
`CLEANUP_LOG.md` Round 2.

## Snapshot

- **Before → after:** 2,918 → 1,324 tracked files; working tree 1.2 GB → 937 MB; tracked content now ~491 MB.
- **`.git` is still 442 MB** (packed). Removals at HEAD don't shrink history; see "History" below.
- **Timeline:** MATLAB/Python model-potential prototypes (May–Oct 2023), then RL agents (Jun–Aug 2023), then OpenMM Langevin
  toy models plus NaCl/PMg/dialanine with DHAM→MSM→MFPT-optimised bias (Oct 2023 – Apr 2024), with a last touch on 2024-11-06.
- **Active line of work:** the "MFPT-opt" adaptive-bias loop: short biased MD → DHAM Markov matrix from all
  propagations → random + L-BFGS search for a 10–20-Gaussian bias that minimises the MSM MFPT to a local target → apply →
  repeat. It is benchmarked against unbiased MD and OpenMM metadynamics by "steps to reach target".
- `paper_publication/`: first commit 2024-01-22, last 2024-11-06.

## Headline findings

### 1. Correctness issues that may affect the published time-to-target numbers
The paper's MFPT-opt results, `paper_publication/data/time2reach/{1D,2D,NaCl}/total_steps_mfpt.csv` (and the 1D
unbiased CSV), are **byte-identical** to the CSVs written by the scripts below (checked with `cmp`). Whether a given row
was produced by buggy code depends on when it was *run*, which git can't tell. Commit dates are below where checked, so
read this table as "**may** affect the paper numbers".

| # | Issue | Where (examples) | Source |
|---|---|---|---|
| 1 | **Gaussian bias params are scrambled before DHAM.** The optimiser returns `concat(a, b, c)` (block layout), but callers do `reshape(1, n, 3)` (C order), and DHAM unpacks each row as `(a, b, c)`. DHAM therefore unbiases with the wrong potentials. | `1D_langevin_sim/util.py:262` → `langevin_sim_mfpt_opt.py:247,303` → `DHAM_it` (`:96,159`) → `dham.py:60,87`; same in 2D, NaCl, PMg | ✔ verified by me; found independently by two agents. **Introduced by 5e7a88b (2023-11-22), the "DHAM fix" commit itself.** Before it, DHAM sliced the flat array block-wise, which is correct (`61a51cb:1D_langevin_sim/dham.py:60-61`). After it, params go through `savetxt` → `loadtxt(...).reshape(-1,3)`. The 1D and NaCl MFPT CSVs gained or rewrote ~20–24 rows in c8c9cdb (2023-12-08), after the bug. |
| 2 | "Steps to reach" records the **last** frame inside the target, not the first crossing (no `break`). It also over-counts by one 5,000-step propagation. | `1D_langevin_sim/langevin_sim_mfpt_opt.py:107-111,382-385`; `mfpt_NaCl.py:112-114` | 01, 02. 11/25 NaCl values ≡ 4999 mod 5000 |
| 3 | PMg: `reach` is multiplied by `dcdfreq=100` although it is already in steps, and the bias is applied **10×** (`addBond` inside the Gaussian loop). | `1D_PMg/explore_sim_PMg.py:415`, `:211-215` | 02 |
| 4 | The unbiased baselines never record a first passage. The CSVs mostly hold the run-length cap (censored), plus some hand-typed values. | `1D_langevin_sim/langevin_sim_unbiased.py:114-117,142` | 01 |
| 5 | Model inconsistencies between optimiser/DHAM and OpenMM: the Gaussian width uses `(2c)²` vs OpenMM's `2σ²`; 2D DHAM drops the amplitude; `exp(-u/2*kT)` has a precedence bug; there is a row- vs column-stochastic mismatch after 2959e7a; `np.digitize` is off by one. | see 03 §"Algorithm correctness" C2–C8 | 03 |
| 6 | Alanine dipeptide: the φ/ψ atom indices are off by one (they pick O–H–HA–O), and the barostat runs at 5 K. | `2D_Dialanine/main.py:66-67`, `unbiased_sim.py:70,80` | 02 |

### 2. Code organisation
- **There is no shared library.** Each folder carries its own `util.py`/`dham.py`/`MSM.py`/`config.py`, imported by cwd-relative path.
  The 14 `util.py` copies are 14 distinct versions from 4 unrelated module families; `dham.py` has 7 versions across 10 copies.
- **Canonical picks (from 03):**
  - `MSM.py`: the `1D_PMg_backforth`/`2D_langevin_sim` copy. The root and `paper_publication` copies crash in `_bias_K`.
  - `dham.py`: `1D_langevin_sim_backforth` (1D) and `2D_langevin_sim` (2D). Only these carry the 5e7a88b per-window fix.
  - `util.py` and `wham.py`: `1D_langevin_sim_backforth`.
- The `*_backforth` folders are parallel WIP forks, not cleaner versions, and all their entry points have crash bugs.
- Every OpenMM script calls `getPlatformByName('CUDA')` **at import**, so nothing (not even analysis) runs on a Mac.
  Output directories must be created by hand first.

### 3. Repo hygiene
- **Regression from commit 45d011f ("Remove .png files"):** it deleted `2D_model_potential_TW/fes_digitize.png`, which
  `figure_gen_2D.py:20`, `2D_presentation_movie_plot.py:260` and `create_2d_fes.py:13` still open. A byte-identical copy survives at
  `2D_Dialanine/figs/fes_digitize.png`. **Not restored**, pending your call.
- `.gitignore` traps: `/1D_*/*` + `!*.py` silently ignores any new non-`.py` file in 7 folders (READMEs, toppar, `.sh`), and
  920 tracked files are already "ignored". The `./paper_publication/` lines match nothing (a leading `./` never matches).
  Only additive rules were added in this pass.
- `environment.yml` is a Windows-only, Python 3.9 deep-learning env and **lacks openmm and mdtraj**. See ENVIRONMENT.md.

## Decision table (needs your call)

Everything below is **still tracked**. Each item looks redundant, but it may be a result or provenance you want to keep.
"Refs" means code that reads it. These are the big-ticket items. Each `findings/0*.md` ends with a full redundancy table
(path | size | evidence | refs | class) covering the smaller CONFIRM items.

| Path | Size | What it is | Refs | Suggestion |
|---|---|---|---|---|
| Root `pos_M_DQN_model_{11thJuly_1,13July_1,13July_2,13July_2_N100,14July_1_N20}.pt` | 5 files, 80 MB | posM DQN checkpoints; filenames misstate N | none (`_13July_2_N20.pt` **is** read → keep) | Archive outside git, then delete |
| Root `data_1D_20N_10Gaussian.pt` | 72 MB | Older offline-RL dataset; actions aliased; differs from `data/` copy | none (`data/` copy is read by CQL → keep) | Delete |
| `RL_Qagent_1D/*.pt` + `models/` | 48 files, 30 MB | DQN checkpoints | none | Archive or delete |
| `RL_matlab/*.mat` | 3 files, 26 MB | MATLAB RL-Designer sessions (DDPG/TD3) | GUI only | Keep only if the MATLAB RL line matters |
| `figures.pptx` | 3.7 MB | Slide deck of 2D DHAM figures | — | Move out of the repo |
| `runs/ppo_trainer__*` | 9 runs, 0.2 MB | Own PPO TensorBoard logs (wandb `pos1D_ppo`) | — | Keep (tiny) or drop with the RL line |
| `1D_NaCl_TW/backup/` | 84 files, 12 MB | Older, different NaCl pipeline (amber FF, umbrella sampling). Its code is superseded, but its **US data, notebooks and `mm_metrics.csv` are unique** | only within `backup/` | Tag/archive, then delete, or keep |
| `*/bak/total_steps_*.csv` | 5 files | Older result CSVs | none | Delete unless needed for the record |
| `1D_PMg/step3_input.{crd,pdb}` | 21 MB | CHARMM-GUI originals, unreferenced; **now the only copy** | comment only | Keep for provenance |
| `1D_PMg_backforth/` inputs (`psf`, `restart.pdb`, `toppar/` ×55) | ~49 MB | Exact duplicates of `1D_PMg/`, but loaded by relative path | own script | Restructure (shared inputs, or a `--backforth` flag); can't just delete |
| `2D_Dialanine/saved_system/` | 33 files, 12 MB | Write-only run end-states; many identical XMLs | none | Delete the folder |
| `2D_Dialanine/trajectories/` | 14 files, 2.2 MB | Run outputs, overwritten each run; tracked despite `.gitignore` | none | Untrack |
| `2D_model_potential_TW/figs/` | 472 files, 34 MB | 422 run-diagnostic PNGs, a 14.5 MB `movie.gif` (rebuildable from frames), `prod_figs/` | `gif_gen.py` reads frames | Keep `prod_figs/` frames; drop the rest |
| `2D_model_potential_TW/data/` | 142 files, 1.3 MB | Run outputs | only `20230927-135127_*` (13 files) | Keep those 13; drop the rest |
| `2D_model_potential_TW/bak/` | 11 files, 0.5 MB | Dialanine-FES fit outputs (from `2D_langevin_sim/fit_dialanine_*.py`) | `fes_digitize_gauss_params_15{,_2}.txt` are the only tracked copies of fit params (05 §Redundancy) | Keep those 2 `.txt`; drop the PNGs |
| Root `fes_visualization.png` | 24 KB | Diagnostic plot written by `apply_fes` to `./fes_visualization.png` | none | Delete (your 45d011f removed the other copies; this one differs in content, so it was kept for your call) |
| `1D_model_potential_Edina/` | 14 files, 1 MB | Snapshot frozen 2023-05-19; Simian copied 12/14 files verbatim | none | Delete (note: its `test.mat`/`benchmark.mat` are also in Simian) |
| `1D_model_potential_Simian/{test2.mat,test.ma}` | 4.4 MB | Unrelated workspace dumps | none | Delete |
| `CE_Opt/` | 11 files, 1 MB | Unfinished cross-entropy experiment; vendored third-party code with no licence | none | Delete or archive |
| Remaining `*/langevin_approach/MetaD_auxfiles/` | 39 files, 0.4 MB | Final metaD bias grids, write-only | none | Delete |
| Root `MSM.py`, `kinetic_networks.py` | — | Unimported. `MSM.py` has the `_bias_K` crash; `kinetic_networks.py` has a NameError | none | Fold into a future shared package |
| Restore `2D_model_potential_TW/fes_digitize.png` | 0.2 MB | Input deleted by 45d011f. The only tracked copy left is `2D_Dialanine/figs/fes_digitize.png` (byte-identical); `2D_langevin_sim/fit_dialanine_*.py` also need one in their own folder | 3 scripts in `2D_model_potential_TW` + 2 in `2D_langevin_sim` | **Restore** (`git show 45d011f^:2D_model_potential_TW/fes_digitize.png > …`) |

### History (not done; needs coordination)
`.git` holds 442 MB packed. By uncompressed blob size across all history the largest contributors are: old `.log` files 208 MB, the two
dataset `.pt` files 146 MB, posM checkpoints 140 MB, `U*.npy` 120 MB, PNG/GIF 91 MB, and the venv 22 MB. Shrinking it requires `git filter-repo` plus a force-push. The remote also has `main`, `GS_Branch`,
`George_ND` and `OisinMoriarty` branches, so every collaborator would need to re-clone. Only worth it if clone size actually hurts.

# Environment notes (for a future rebuild)

_Written during the 2026-11 cleanup. Virtual environments and compiled bytecode were removed from the
repo. This file records what they were and what the code actually needs, so the environment can be
rebuilt rather than committed._

## 1. What was removed, and how to recreate it

| Removed | What it was | How to recreate |
|---|---|---|
| `RL_CLEANRL/Lib/`, `RL_CLEANRL/Scripts/`, `RL_CLEANRL/pyvenv.cfg` (≈1,470 files) | A **Windows `venv`**, created with `python -m venv C:\Users\13401\Documents\GitHub\RL_MFPT\RL_CLEANRL` (from `pyvenv.cfg`). It used **Python 3.11.4** (Microsoft Store build) and contained **only** `pip 23.1.2` and `setuptools 65.5.0`; no project packages were ever installed into it. | `python3.11 -m venv .venv` (now gitignored) |
| `RL_CLEANRL/cleanrl` (gitlink) | A **broken submodule pointer** to commit `f62ad1f942890db7b752f4ad79b65be53917d3c2`, with no `.gitmodules` entry, so nobody could ever check it out. It was most likely the CleanRL library [Inferred]. | `git clone https://github.com/vwxyzjn/cleanrl && git -C cleanrl checkout f62ad1f9` (verify the hash exists upstream) |
| `RL_CLEANRL/runs/`, `runs/{CartPole,Breakout,MicrortsMining}*` | TensorBoard logs from **tutorial** runs: CleanRL's `ppo.py`/`ppo_atari.py` and ppo-implementation-details' `ppo_multidiscrete.py` (wandb project `cleanRL`; none of these scripts is in this repo). The authors' own `runs/ppo_trainer__*` (wandb project `pos1D_ppo`) were **kept**. | Re-run the upstream tutorials if needed |
| 743 `*.pyc` / `__pycache__/` | Bytecode: 713 belonged to the venv above. The other 30 were project caches compiled by **two interpreters**, `cpython-39` (18 files, mostly the `RL_*` folders) and `cpython-311` (12 files, `2D_*`, `1D_model_potential_TW`, the RL utils). | Regenerated automatically on import |

## 2. Existing environment specs in the repo, and how usable they are

| File | Status |
|---|---|
| `environment.yml` (conda env `pDL`, last touched 2023-06-26, commit fb3e90d) | **Stale, and Windows-only.** Its build strings pin `win64_mkl`, `m2w64-*`, `vc`/`ucrt` and `prefix: C:\Users\13401\anaconda3\envs\pDL`, so `conda env create -f environment.yml` will fail on macOS/Linux. It pins Python 3.9.16 with a **CPU-only** PyTorch 2.0.1. It is a generic deep-learning env, not this project's: it **does not include `openmm` or `mdtraj`** (the two most-used scientific deps, see §3), nor `plotly`, `gym`, `tensorboard`, `seaborn`, `natsort` or `jax`. Its only pip extras are `python-graphviz` and `torchviz`. |
| `RL_PPO_implementation/pyproject.toml` + `poetry.lock` (**removed in the cleanup**; nothing referenced it, and the key deps are recorded here) | The Poetry spec of the third-party **"ppo_details"** project (ICLR 2022 blog post, `authors = ["Iclr Blog ..."]`). It needs `python >=3.7.1,<3.10`, `torch ^1.7.1`, `gym ^0.21.0`, `stable-baselines3 ^1.1.0` and `wandb 0.12.1`. It isn't this project's environment. A Dependabot branch (`origin/dependabot/pip/RL_PPO_implementation/...`) exists only because of this file. |

## 3. What the code actually imports

Third-party import *statements* across all tracked `.py` files, counted after the cleanup with `grep '^(import|from)'`:

| Package | Import stmts | Used in |
|---|---:|---|
| `openmm` (+ legacy `simtk` ×6) | 209 | All MD/Langevin folders: `1D_*`, `2D_langevin_sim`, `2D_Dialanine`, `CE_Opt`, `paper_publication` |
| `numpy` | 120 | everywhere |
| `matplotlib` (+ `mpl_toolkits`) | 118 | everywhere |
| `scipy` | 106 | everywhere |
| `torch` | 46 | `RL_*` folders |
| `tqdm` | 37 | most folders |
| `mdtraj` | 28 | `1D_*`, `2D_langevin_sim`, `2D_Dialanine`, `CE_Opt`, `paper_publication` |
| `PIL` (pillow) | 12 | `2D_model_potential_TW`, `2D_langevin_sim`, `RL_DQN_1D_N100`, `RL_Qagent_1D`, `1D_NaCl_TW*` |
| `numba` | 8 | `1D_langevin_sim_backforth`, `1D_PMg*`, `2D_langevin_sim`, `CE_Opt`, `paper_publication`, root |
| `plotly` | 6 | Langevin / PMg / `CE_Opt` |
| `gym` | 6 | `RL_pos_PPO_1D` only |
| `pandas` | 4 | `paper_publication`, `CE_Opt`, `1D_NaCl_TW/backup` |
| `seaborn` | 3 | `paper_publication`, `1D_NaCl_TW` |
| `torch.utils.tensorboard` | — | `RL_pos_PPO_1D` |
| `sklearn`, `jax` | 1 each | `1D_NaCl_TW_backforth` |
| `natsort` | 1 | `1D_langevin_sim_backforth/analysis.py` |
| `imageio`, `cv2` | 1 each | `2D_model_potential_TW` |
| `atari_py` | 1 | an RL tutorial script |

Notebooks (`*.ipynb`) additionally use `openmm`, `mdtraj`, `imageio` and `condacolab` (a Colab-only installer).

Local, non-package imports (`util`, `config`, `dham`, `MSM`, `util_2d`, `env`, `dqn_agent`, …) resolve to **the copy in the script's own directory**. See `docs/findings/03-core-modules-and-paper.md`.

**Platform:** OpenMM scripts call `Platform.getPlatformByName('CUDA')` 34 times, `'CPU'` 7 times and `'OpenCL'` twice. On a Mac (no CUDA), most MD scripts need that line switched to `'CPU'` or `'OpenCL'`.

## 4. Proposed cross-platform environment (an untested starting point)

A conda-forge spec without build strings, for the MD/MFPT core (Python 3.11 matches the newest bytecode found):

```yaml
name: rl_mfpt
channels: [conda-forge]
dependencies:
  - python=3.11
  - numpy
  - scipy
  - matplotlib
  - openmm>=8.0
  - mdtraj
  - numba
  - tqdm
  - plotly
  - pandas
  - seaborn
  - pillow
  - natsort
  - imageio
  - ipykernel
```

Optional RL extras (only for the `RL_*` folders, which date from mid-2023):
`pytorch` (CPU or CUDA build to taste), `tensorboard`, plus **`gym` 0.22–0.23** for `RL_pos_PPO_1D`.
It needs `reset(seed=)` (≥0.22) and the old 4-tuple `step` (<0.26). The `poetry.lock`'s gym 0.21 would break `env.py:61`.
Old `gym` is awkward on Python 3.11, so either port those scripts to `gymnasium` or use a separate Python 3.9 env.
`stable-baselines3`, `wandb` and `gymnasium` are **not** imported by any repo code. Details are in `docs/findings/04-rl-experiments.md`.

The cluster scripts (`2D_langevin_sim/do_python*.sh`, `1D_NaCl_TW/run_NaCl_explore_sim.sh`) activate a conda env called
**`biophys_env`**. Its spec isn't in the repo; if you still have it on the cluster, `conda env export --no-builds -n biophys_env`
there would be the most faithful record of the env the MD results were produced with.

Not done yet: pinning exact versions, or testing that the env solves on macOS arm64. Do both before relying on it.

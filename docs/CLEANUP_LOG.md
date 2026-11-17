# Cleanup log (2026-11-17)

Branch `TW_working_branch`, starting at commit `45d011f` (= `origin/TW_working_branch`).
**Nothing is committed and no history was rewritten.** All removals are staged deletions (`git rm`). The `.gitignore` edit
and `docs/` are **not** staged yet, so a commit needs `git add .gitignore docs` first.

- Review: `git status --short | grep -v '^D '` lists the non-deletion changes; `git diff --cached --stat` shows the full summary.
- Undo everything: `git restore --staged --worktree -- . && git clean -n docs/` (then `-f` to drop the new docs).
- Undo one path: `git restore --staged --worktree -- <path>`

## Round 1: mechanical junk (done)

Tracked files went from **2,918 to 1,371**.

| Removed | Count | Why it's safe |
|---|---:|---|
| `RL_CLEANRL/Lib/`, `RL_CLEANRL/Scripts/`, `RL_CLEANRL/pyvenv.cfg` | ~1,475 | A Windows venv (pip + setuptools only). Recorded in `ENVIRONMENT.md`. |
| `RL_CLEANRL/cleanrl` (gitlink) | 1 | A broken submodule (no `.gitmodules`). Its commit hash is recorded in `ENVIRONMENT.md`. |
| `**/*.pyc`, `__pycache__/` | 30 (+713 already counted in the venv row) | Bytecode, regenerated on import. |
| `.vs/` | 7 | Visual Studio workspace cache. |
| `**/*.log` | 13 | Run logs. `**/*.log` was already gitignored, and commit 2aa2aaa removed the others. |
| `**/nohup.out` | 3 | Batch stdout (`2D_model_potential_TW/nohup.out` was 14 MB). |
| `**/*.asv` | 9 | MATLAB editor autosaves. |
| `1D_NaCl_TW/backup/.ipynb_checkpoints/`, `.DS_Store` | 6 | Jupyter and macOS clutter. |
| Zero-byte files: `2D_Dialanine/s`, `{1D_langevin_sim,1D_langevin_sim_backforth}/bak/total_steps_metaD.csv`, `2D_model_potential_TW/data/20230925-162258_0_CV_total.npy` | 4 | Empty. The `.npy` can't even be loaded. |

`.gitignore`: **additive** rules only were appended (venvs, `*.py[cod]`, `.vs/`, `.DS_Store`, `.ipynb_checkpoints/`, `*.asv`, `nohup.out`).
Verified with `git ls-files -ci --exclude-standard` that no tracked file is *newly* matched.

## Round 2: redundant files (done)

Tracked files went from **1,371 to 1,324**. Every removed file has no reader (repo-wide `git grep` of its basename), and I re-checked the
evidence myself before removal (`cmp`/md5, npy header parsing, grep). Most are mechanical: byte-identical copies, all-zero outputs,
write-only caches, or third-party tutorial output. Rows marked **⚖ judgement** fit none of those categories and were removed on judgement.
Veto any of them with `git restore --staged --worktree -- <path>`.

| Removed | Size | Evidence |
|---|---:|---|
| `1D_langevin_sim_backforth/{U,U_denom,UB3d}.npy` | 188 MB | Write-only WHAM caches (`wham.py:83-84,209`); the only loads are commented out (`:92-93`); `U_denom.npy` ≡ `U.npy`. **`U.npy`/`UB3d.npy` cannot be regenerated from this checkout**: they need run `20231123-170259`'s `visited_states/` and `params/`, which aren't in the repo. Recover from git (`git show 45d011f:1D_langevin_sim_backforth/U.npy > U.npy`) if needed. |
| `1D_langevin_sim_backforth/bak/total_steps_mfpt.csv` | 120 B | ≡ `1D_langevin_sim/bak/` copy |
| 7 × `*/langevin_approach/MetaD_auxfiles/<tag>/bias_*_1.npy` (+ their dirs) | 0.6 MB | **All-zero** arrays: OpenMM's constructor-time save; those runs never reached a real save |
| `1D_NaCl_TW/backup/biasData/bias_1273876585_1.npy` | 136 B | ≡ one of the all-zero files above |
| ⚖ `2D_langevin_sim/equation_note.txt` | 65 B | A one-line title, no content |
| `DIALA/` (`dialaA.pdb`, `dialaB.pdb`) | 4 KB | ≡ `2D_Dialanine/` copies; `git grep DIALA` is empty |
| `1D_PMg_backforth/step3_input.{crd,pdb}` | 22 MB | ≡ `1D_PMg/` copies; never loaded (the script loads `.psf` + `traj_0.restart.pdb`) |
| ⚖ `1D_PMg_backforth/total_steps_mfpt.csv` | 160 B | An exact prefix (first 20 lines) of `1D_PMg/total_steps_mfpt.csv`, not a backforth result |
| `1D_PMg_backforth/metaD_NaCl_sim.py` | 4.7 KB | ≡ `1D_PMg/` copy, and it can't run in a PMg dir anyway |
| `1D_NaCl_TW_backforth/toppar/step3_input{.crd,.pdb,_2A.pdb}` | 0.5 MB | ≡ `1D_NaCl_TW/toppar/` copies; the backforth scripts load only `.psf` + `step3_input25A.pdb` |
| `1D_NaCl_TW/backup/fig/Figure_1.png`, `2D_Dialanine/figs/DHAM_prop1.png` | 71 KB | ≡ copies in sibling subfolders |
| `RL_Qagent_1D/transformer_agent.py` | — | ≡ `RL_DQN_1D_N100/mingpt.py`; imported by nothing |
| `RL_matlab/{explore_the_network,Markov_mfpt_calc}.m` | — | ≡ `1D_model_potential_{Edina,Simian}` copies; no caller in `RL_matlab` |
| `RL_CLEANRL/runs/` (2), `runs/{CartPole,Breakout,MicrortsMining}*` (6) | 1.5 MB | Tutorial runs (wandb project `cleanRL`; scripts `ppo`/`ppo_atari`/`ppo_multidiscrete` aren't in this repo). The authors' own `runs/ppo_trainer__*` (wandb `pos1D_ppo`) were **kept**. |
| ⚖ `RL_pos_PPO_1D/runs/` (4) | 3 KB | The authors' own aborted runs: hparams only, no scalars |
| `RL_PPO_implementation/` | 0.4 MB | Third-party `ppo_details` pyproject + lock; nothing references it. Deps are recorded in `ENVIRONMENT.md`. |
| `paper_publication/figure_gen_2D.py` | — | ≡ `2D_model_potential_TW/figure_gen_2D.py`; can't run from `paper_publication/` (its inputs live in `2D_model_potential_TW/data/`) |
| `paper_publication/pos_bias_m.mat` | — | ≡ 2 other copies (still present); only a commented-out reference |
| ⚖ `{2D_model_potential_TW,paper_publication}/.vscode/launch.json` | — | Stock VS Code templates |
| ⚖ `test.png` (repo root) | 19 KB | A debug plot; nothing reads it. (Root `fes_visualization.png` was removed, then **restored** and moved to the decision table, because it differs from the copy 45d011f removed.) |
| ⚖ `1D_model_potential_Simian/main_optbias (Edina Rosta's conflicted copy 2023-05-23).m` | — | A Dropbox conflict copy. Its diff vs `main_optbias.m` is `state_start=1` (vs 8), `Ngauss=20` (vs 10), and a missing `plot` line. |

`.gitignore`: also added `.vscode/` and `*conflicted copy*` (additive; no tracked file newly matched).

**Deliberately not removed** (see the decision table in `README.md`):
- The per-folder copies of `util.py`/`dham.py`/`MSM.py`/`config.py`, because scripts import them by cwd-relative path.
- Duplicated inputs that a script in its own folder loads by relative path (e.g. `1D_PMg_backforth/toppar/`).
- Anything that might be a research result: trained models, figures, result CSVs, notebooks, `backup/`.
- Partial de-duplication of `2D_Dialanine/saved_system/`: it would leave incomplete run sets, so the whole folder is listed for a decision instead.
- Items the findings docs label DELETE-SAFE that I held back:
  - `1D_model_potential_Simian/{test,benchmark}.mat`: ≡ the Edina copies, and Edina itself is up for deletion, so removing either side first risks losing both.
  - `1D_NaCl_TW/backup/{test2.py,run_NaCl_explore_sim.sh,mfpt_biase_NaCl_gen_2.py}`: judgement-only calls, and they belong with the `backup/` decision.

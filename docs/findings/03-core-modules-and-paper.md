# 03 · Core-module drift, root scripts and `paper_publication/`

Scope: `MSM.py`, `dham.py`, `util.py`, `util_2d.py`, `wham.py` and `config.py` copies; root `MSM.py`, `kinetic_networks.py`, `data_gen.py`; `paper_publication/`. Read-only static analysis: no repo script was run, and numpy/scipy are not installed in the shell, so no numerical check was possible.
Labels: **[Verified]** means read in the file, diff or command output (static). **[Inferred]** means reasoning or numerical consequence, untested. **Unlabelled statements that cite a `file:line`, an md5 or a commit are [Verified] (static reading).** Any runtime consequence (crash, wrong number, failure) is [Inferred] unless stated otherwise.
Version groups are by md5 (`md5 -q`). Dates are `git log -1 -- <path>`. Version-group IDs (M-A, D6, ...) are defined in the tables below.

## Summary

- **No copy is "the" library.** Every directory carries its own snapshot. `util.py` has 14 copies and 14 distinct versions, falling into 4 unrelated module families. `dham.py` has 7 versions across 10 copies, `MSM.py` 3 across 8 [Verified].
- **Most complete versions:** `MSM` group **M-B** (`1D_PMg_backforth`, `2D_langevin_sim`, `paper_publication/1D_K_opt_recalc`); `dham` **D6** for 1D (`1D_langevin_sim_backforth` = `1D_PMg` = `1D_PMg_backforth`) and **D2** for 2D (`2D_langevin_sim`); lean `util` **F2** (`1D_langevin_sim_backforth` family); `wham` **W2** (`1D_langevin_sim_backforth`).
- **Root `MSM.py` has a crash bug.** Root `MSM.py`, which is also `paper_publication/MSM.py`, carries a `_bias_K` call that crashes. It was fixed only in M-B. Nothing imports either root-level copy [Verified].
- **The 5e7a88b DHAM fix** (per-window Gaussian params) is in every DHAM copy except `2D_model_potential_TW` and `2D_Dialanine`. The cbc3484 power-method workaround was **reverted** by 5e7a88b and is active nowhere (`git grep -F 'compute_free_energy_power_method('` finds only the `def` lines and commented calls). c1c6c3d was only bound tuning, later overwritten [Verified].
- **Five correctness problems in the active code paths:**
  1. The optimiser and DHAM use a Gaussian of width `(2c)²`, while OpenMM applies `2σ²`.
  2. The 2D DHAM drops the bias amplitude.
  3. An operator-precedence bug in `exp(-u/2*kT)` in four copies.
  4. A row/column-stochastic mismatch after 2959e7a.
  5. `np.digitize` off-by-one, with transitions silently dropped.

  See *Algorithm correctness concerns*.
- **`paper_publication/`** holds the plotting scripts and data for the paper. The bar chart, analytic FES renders and 1D bias figures can be regenerated from tracked data. The `2Dtraj_*` figures cannot (gitignored `.dcd`/`.npy`). `figure_gen_2D.py` there cannot run at all. The `./paper_publication/` lines in `.gitignore` match nothing.

## Drift map

### `MSM.py` (8 copies, 3 versions)
| copy path | group | last commit | notes |
|---|---|---|---|
| `MSM.py` (root) | M-A `7a49…` | 95222c3 2024-01-22 | origin of the class (5d614fc). `_bias_K` crashes (C1). **No importer** |
| `paper_publication/MSM.py` | M-A | c17d449 2024-04-24 | byte-identical to root. **No importer** in `paper_publication/` |
| `1D_langevin_sim_backforth/MSM.py` | M-A | bed1cc7 2024-01-02 | used only through `_bias_M('direct_bias')`, which avoids C1 |
| `1D_PMg/MSM.py` | M-A | 95222c3 2024-01-22 | same usage |
| `1D_PMg_backforth/MSM.py` | **M-B `4934…`** | 95222c3 2024-01-22 | M-A + C1 fix (1 line). **Canonical** |
| `2D_langevin_sim/MSM.py` | M-B | c17d449 2024-04-24 | `2D_opt_K.py:139,162` calls `_bias_K`, so it needs M-B |
| `paper_publication/1D_K_opt_recalc/MSM.py` | M-B | c17d449 2024-04-24 | `main.py:92,123` calls `_bias_K`, so it needs M-B |
| `CE_Opt/MSM.py` | M-C `e069…` | c17d449 2024-04-24 | M-A + paper FES test case; **still has C1** (`:392` vs `:450`) |

### `dham.py` (10 copies, 7 versions)
| copy path | group | last commit | notes |
|---|---|---|---|
| `2D_Dialanine/dham.py` | D1b `9728…` | 7251d34 2023-09-08 | 2D, **pre-5e7a88b** (single bias `self.a[n]`, `:90`). Returns `MM.T*4.184` (C7) |
| `2D_model_potential_TW/dham.py` | D1a `a68f…` | 3fa7301 2023-09-25 | 2D, pre-5e7a88b (`:93`). Hard-coded 20×20 grid |
| `2D_langevin_sim/dham.py` | **D2 `56e7…`** | 5e7a88b 2023-11-22 | only 2D copy with the fix (`:98`). Drops amplitude (C3); transposed counts (C6) |
| `1D_langevin_sim/dham.py` | D3 `a21e…` | 2959e7a 2023-11-23 | fix (`:87`). DHAMsym always on. Returns `MM.T` (since 2959e7a). Ignores `num_bins` arg (C9) |
| `CE_Opt/dham.py` | D3 | c17d449 2024-04-24 | blob = `c17d449^:1D_langevin_sim/dham.py` |
| `1D_NaCl_TW/dham.py` | D4 `4e4d…` | c8c9cdb 2023-12-08 | D3 + edges `linspace(2.0, 9, config.num_bins+1)` (`:117`) |
| `1D_NaCl_TW_backforth/dham.py` | D5 `3e7e…` | c8c9cdb 2023-12-08 | D4 + `use_symmetry` flag + `config.qspace` |
| `1D_langevin_sim_backforth/dham.py` | **D6 `1f17…`** | bed1cc7 2024-01-02 | **canonical 1D** (see Key diffs) |
| `1D_PMg/dham.py`, `1D_PMg_backforth/dham.py` | D6 | 95222c3 2024-01-22 | identical |

### `util.py` (14 copies, 14 versions): four different modules sharing a name
| copy path | family / version | last commit | notes |
|---|---|---|---|
| `1D_langevin_sim/util.py` | F1 1D-OpenMM full `f7a0…` | c17d449 2024-04-24 | K/M helpers + `try_and_optim_M` + `apply_fes/bias`. Left wall moved to x=0.6 in c17d449 (`:338`) |
| `CE_Opt/util.py` | F1 `adbe…` | c17d449 2024-04-24 | blob = `c17d449^:1D_langevin_sim/util.py` (pre-wall-change snapshot) [Verified `git rev-parse`] |
| `1D_NaCl_TW/util.py` | F1-NaCl `3dfe…` | c8c9cdb 2023-12-08 | adds K-based `try_and_optim`; no OpenMM helpers |
| `1D_NaCl_TW_backforth/util.py` | F1-NaCl `6a95…` | c8c9cdb 2023-12-08 | = NaCl + `config.qspace`, bounds via `config.num_gaussian` |
| `1D_langevin_sim_backforth/util.py` | **F2 lean/MSM-class `114a…`** | bed1cc7 2024-01-02 | 6 functions; MFPT via `MSM` class. **Canonical 1D** |
| `1D_PMg/util.py` | F2 `1074…` | c17d449 2024-04-24 | F2 + one comment line (`:66`) |
| `1D_PMg_backforth/util.py` | F2 `fd9d…` | c17d449 2024-04-24 | F2 retuned: 2000 tries, a=1.0, a-bound 1.5 (`:54,59,130`) |
| `2D_langevin_sim/util.py` | F3 2D-OpenMM `4d14…` | 95222c3 2024-01-22 | `gaussian_2D(params,x,y)`, 2D `try_and_optim_M`, OpenMM helpers |
| `1D_model_potential_TW/util.py` | F4 K-matrix/RL `0dab…` | 835493b 2023-06-12 | earliest; origin of the `(2c)²` Gaussian (`:10-11`) |
| `RL_Qagent_1D/util.py` | F4 `4ce2…` | 71ce56f 2023-07-07 | imported by root `data_gen.py:2` |
| `RL_DQN_position/util.py` | F4 `b0c0…` | a1d5f3b 2023-06-20 | `mfpt_calc` uses `K` not `K.T` (C13) |
| `RL_DQN_1D_N100/util.py` | F4 `338a…` | 71ce56f 2023-07-07 | + torch `set_seed`, `sample` (minGPT) |
| `RL_posM_env/util.py` | F4 `014b…` | 8501a47 2023-07-20 | only F4 copy with a fixed Gaussian (`:14-18`) |
| `RL_pos_PPO_1D/util.py` | F4 `c14e…` | b4d8ca7 2023-08-09 | newer than posM_env but Gaussian bug re-present (`:15`) |

### `util_2d.py` and `wham.py`
| copy path | group | last commit | notes |
|---|---|---|---|
| `2D_model_potential_TW/util_2d.py` | U2a `6f88…` (506 l.) | a7914b5 2023-10-23 | + `create_K_png`, `try_and_optim_M_K` (cutoff mode), save helpers. **Canonical** for the K-matrix 2D model |
| `2D_Dialanine/util_2d.py` | U2b `0434…` (391 l.) | 849625c 2023-10-16 | + `clean_rad`, `bias_M_2D`; diff is 116+/231− lines |
| `1D_NaCl_TW_backforth/wham.py` | W1 `a9b1…` (359 l.) | c8c9cdb 2023-12-08 | original umbrella WHAM (harmonic `k_val`, `read(path)`, jax, KMeans, bootstrap). **No importer** |
| `1D_langevin_sim_backforth/wham.py` | **W2 `652e…`** (212 l.) | 95222c3 2024-01-22 | adapted to Gaussian-sum bias (`setup(..., global_gaussian_params)`, `:55-87`). Imported by `analysis.py:31` |

## Key diffs between versions

- **M-A → M-B** [Verified diff]. One line changes, `MSM.py:370`: `_is_adjacent(i,j, verbose=False)` becomes `_is_adjacent(i,j)`. The fix appeared in `1D_PMg_backforth` in 95222c3, in the same commit that left root at M-A. It was copied to the other two M-B dirs in c17d449.
- **M-A → M-C.** The 1D test FES in `_init_sample_K` is replaced by the paper multiwell FES (`CE_Opt/MSM.py:99-119`), and `Z` is renamed to `y` in the 2D/3D tests. C1 is kept.
- **MSM history** [Verified `git log -- MSM.py`]:
  - 5d614fc added the class.
  - 92f1916 added `_build_mfpt_matrix_M(method='JJhunter')` (numba). The same commit changed `_is_adjacent` to two arguments, which is the origin of C1.
  - 95222c3 added `time_step` to `build_MSM_from_M` and removed a 6-D test.
- **DHAM lineage** [Verified diffs]:
  - D1 (Sep 2023, 2D) is the base.
  - 5e7a88b, "using all previous gaussian params", changed `__init__` to take `gaussian_params[prop, g, 3|5]`. `u` is now rebuilt per trajectory `k` inside `build_MM` (`1D_langevin_sim/dham.py:83-91`) instead of once from one bias.
  - The same commit also re-enabled DHAMsym and switched peq back from `compute_free_energy_power_method` to `eig`. That reverted cbc3484: zero active power-method calls at 5e7a88b and at HEAD [Verified `git show` + `git grep`].
  - 2959e7a changed the 1D return from `MM` to `MM.T` (`:141`).
  - D4/D5 add NaCl edges.
- **D6 vs D3** [Verified diff]:
  - `config` and `MSM` imports.
  - `use_symmetry` flag, default off in both `count_transitions` and `run` (`:24,108`); the `DHAM_it` wrappers turn it on (`1D_langevin_sim_backforth/langevin_sim_mfpt_opt.py:145`, `1D_PMg/explore_sim_PMg.py:87`).
  - Optional dynamic bins (`:120-128`).
  - Attributes moved to `__init__`, which now honours `num_bins` (`:62`).
  - peq/FES come from `MSM._compute_peq_fes_M`.
  - Returns a **3-tuple** `(qspace, F, MM.T)` (`:146`), not D3's 2-tuple, so D3 and D6 are not drop-in compatible.
  - The `sump1 > 0` guard is commented out (`:94-95`, C10).
- **D2 vs D3.** 2D unravel, `gaussian_2d` without amplitude (`:17-18`), `Ntr[k,to,from]` (`:39`), `exp(+Δu/2kT)` (`:101`), returns `MM` untransposed (`:160`). No bootstrap.
- **F1 → F2.** The lean util drops `create_K_1D`, `mfpt_calc`, `bias_K_1D`, `compute_free_energy*`, `bias_M_1D` and `Markov_mfpt_calc` in favour of the `MSM` class. `try_and_optim_M` now biases through `MSM._bias_M('direct_bias')`, which also removes the C4 precedence bug.
- **F1 (1D_langevin_sim) vs CE_Opt** [Verified diff]: only the left-wall expression `:338` and the plot range `:440,446` differ.
- **Fix tracking:**
  - 5e7a88b: present in D2–D6, absent in D1.
  - c1c6c3d: amplitude upper bound 0.7 and `a=0.35`, overwritten by later bounds (now `(0.1,1.2)`, `1D_langevin_sim/util.py:326`).
  - cbc3484: reverted everywhere.
  - C1 fix: M-B only.
  - Gaussian precedence fix: `RL_posM_env` only.

## `config.py` parameters (8 copies; `1D_langevin_sim` ≡ `CE_Opt`)
| dir | num_sim | sim_steps | amp | T | prop_step | dt bias / unbias (ps) | bins | n_gauss | start → end | other |
|---|---|---|---|---|---|---|---|---|---|---|
| 1D_langevin_sim, CE_Opt | 20 | 5e7 | 6 | – | 5000 | 0.002 / 0.2 | `num_bins`=100 | 20 | 2.0 → 5.2 nm (unb. 3.9) | dcdfreq 500 |
| 1D_langevin_sim_backforth | 1 | 5e6 | 6 | 300 | 5000 | 0.002 / 0.002 | qspace 100 pts [0,2π]; DHAM 50 **dynamic** | 10 | 2.0 → 5.2 nm | max_cycle 20, num_propagation 400 |
| 1D_PMg | 15 | 5e6 | 6 | 300 | **50000** (was 5000, c17d449) | 0.002 / 0.002 | qspace 100 pts [1,8]; DHAM 100 static | 10 | 2.0 → 5.2 nm | restraint 1000 kJ/mol/nm², γ=1/ps |
| 1D_PMg_backforth | 15 | 5e6 | 6 | 300 | 50000 | 0.002 / 0.002 | qspace 100 pts **[1,10]**; DHAM 100 static | 10 | `2 → 5` (bare ints) | same |
| 1D_NaCl_TW | 20 | 5e7 | 6 | 300 | 5000 | 0.002 / 0.2 | `num_bins`=100 (dham class default 150) | 10 | 2.0 → 5.2 nm (copied, unused) | – |
| 1D_NaCl_TW_backforth | 20 | 5e7 | 6 | 300 | 5000 | 0.002 / 0.2 | 100; qspace `linspace(0,15,100)` | 10 | – | max_cycle 20 |
| 2D_langevin_sim | 5 | 1e8 | 6 | – | 5000 | 0.002 / 0.002 | `num_bins`=20 per axis | 20 | (5.0,4.0) → (1.0,1.5) | – |

All of them use `fes_mode='multiwell'` and `Platform 'CUDA'`, so importing them on a machine without CUDA should fail [Inferred]. `config.start_state`/`end_state` are not referenced by any `1D_NaCl_TW*` or `1D_PMg` script. Only `1D_PMg_backforth/explore_sim_PMg.py:296,431-434` reads them, as back-and-forth targets (`git grep -F config.start_state -e config.end_state`) [Verified].

## Root scripts

- **`MSM.py`** (22.6 kB, M-A). The original MSM class and its self-test (`__main__` runs the 3-D case, `:516-544`). No `.py`/`.ipynb` imports it: every `from MSM import` resolves to the importing script's own directory copy [Verified grep]. It carries C1.
- **`kinetic_networks.py`** (6.8 kB, 71ce56f 2023-07-07). Torch/numpy helpers: `compute_free_energy`, `Markov_mfpt_calc`, `compute_mfpt_msm`, `compute_mfpt_rate`, `create_rate_one_dim`. No importer [Verified grep]. Bugs:
  - `mfpt_peq_calc` raises `NameError` for any non-`str` input, because `markov` is assigned only in the `str` branch (`:42-46`).
  - `compute_free_energy` calls `torch.linalg.eig` on numpy input (`:13`).
- **`data_gen.py`** (1.9 kB, 71ce56f). Builds the offline-RL dataset and does produce `data_1D_20N_10Gaussian.pt` (`torch.save(..., "data_1D_20N_10Gaussian.pt")`, `:47`, written to cwd). It needs cwd = repo root (`from RL_Qagent_1D.util import *`, `:2`) and is unseeded. No importer.
  - The only reader is `RL_offline_CQL/main.py:8`, which loads `../data/data_1D_20N_10Gaussian.pt`, i.e. `data/`, **not** the root file. The two files differ (md5 `ffa4…` 75.2 MB vs `3a70…` 78.2 MB) [Verified].
  - The reward mixes formulas: the baseline uses the rate-matrix `mfpt_calc` (`:11`), while each step uses the discrete-time `markov_mfpt_calc(peq, K_biased)` on a **rate** matrix (`:26`; `RL_Qagent_1D/util.py:107` inverts `A + I - M`). Inconsistent units/definition [Inferred].

## `paper_publication/`

**Purpose** [Inferred from filenames/plots]: the figures and data for the paper's comparison of MFPT-opt vs WT-MetaD vs classical MD (time-to-reach), plus 1D/2D FES renders and 1D optimal-bias illustrations. 39 tracked files, 6.6 MB. The last content commit was c17d449; 45d011f later deleted 9 PNGs here.

| script | writes | reads | regenerable from tracked data? |
|---|---|---|---|
| `analysis.py` (flag `plot_boxchart`) | `figs/bar_plot_log.png` (`:105`); `figs/box_plot_log.png` via the commented `:104` | `data/time2reach/{1D,2D,NaCl}/*.csv` (`:31-49`, ×0.002 `:25`) | **yes** [Inferred] |
| `analysis.py` (`plot_2d_traj`) | `figs/2Dtraj_visual_scatter_metad.png` (`:232`), `2Dtraj_visual_3D_traj.png` (`:256`). `2Dtraj_visual_kde_mfpt.png` came from the 95222c3 version (`:198`), other `kde/line` PNGs from earlier toggles | `data/traj/2D/MSMopt/*.dcd`, `data/traj/2D/metad/20231129-1922082_metaD_pos_traj_.npy`, `system.pdb` (`:112-125`) | **no**: `.dcd` (`**/*.dcd`) and that `.npy` (`.gitignore:36`) are ignored and absent |
| `analysis.py` (`plot_1d_fes`, `plot_2d_fes`) | `1D_fes_visual_render{1D,3D}.png` (deleted in 45d011f), `figs/2D_fes_visual_render{2D,3D}.png` (`:398,424`) | analytic FES only | **yes** |
| `figure_gen_1D.py` | `1D_F.png`, `1D_opt_biasedF_mat_overlap.png`, `1D_opt_bias_mat_overlap.png` (to cwd, `:172,257,406`; deleted in 45d011f) | `pos_bias_1stMar_2.mat` (`:20`) | **yes**, but the `.mat` itself comes from MATLAB code outside this folder [Inferred] |
| `figure_gen_2D.py` (≡ `2D_model_potential_TW/figure_gen_2D.py`) | `figs/unbiased.png`, `figs/prod_figs/traj{1,2,3}.png` | `util_2d` (`:4`), `./fes_digitize.png` (`:20`), `./data/20230927-135127_*` (`:38-67`) | **not from here**: no `util_2d.py`, data or PNG in `paper_publication/`. It works only from `2D_model_potential_TW/` (data tracked there) once `fes_digitize.png` is restored. 45d011f deleted it; an identical copy is at `2D_Dialanine/figs/fes_digitize.png` (md5 `1e52…`) |
| `data/time2reach/recalc_metad_reachtime.py` | `data/time2reach/2D/total_steps_metaD.csv` (`:30`) | metaD `.dcd` + `data/traj/2D/metad/explore_traj.pdb` (`:16`) | no (`.dcd` ignored). Must be run with cwd = `paper_publication/` |
| `1D_K_opt_recalc/main.py` | `optim_bias_param_list.txt`, `bias_*.png`, `FES.png` (`:107,147,153`) | nothing (analytic K) | yes, but its output is **not read by any figure script** (grep) [Verified] |

- **`.gitignore`** [Verified `git check-ignore -v --no-index`]. Lines 34–35 (`./paper_publication/data/*`, `./paper_publication/`) match **nothing**, because gitignore does not accept a `./` prefix. New files under `paper_publication/` are therefore not ignored. Only line 36 (one `.npy`) and the global `**/*.dcd`, `**/*gz`, `**/*.log` rules apply there.
- **Data provenance** [Verified md5]: `time2reach` CSVs are copies of the run dirs. `2D/total_steps_mfpt.csv` ≡ `2D_langevin_sim/…`; `NaCl/*` ≡ `1D_NaCl_TW/…`; `1D/total_steps_{mfpt,unbiased}.csv` ≡ `1D_langevin_sim/…`.
- **Data quality:**
  - `1D/` and `2D/total_steps_unbiased.csv` each hold **one** hand-formatted value (`1.063e+08`, `18.526e+06`), so the "Classical MD" bar is n=1 for 1D and 2D.
  - `2D/total_steps_metaD.csv` has n=5. Its predecessor `2D/metaD_bak/` has 40 values, mostly `4999500`, an apparent cap.
  - Many NaCl MFPT-opt values equal `k·5000−1` (34999, 44999, …), i.e. `propagation_step` boundaries [Verified contents; interpretation Inferred].

## Algorithm correctness concerns (file:line). Noted, not fixed

1. **C1 `_bias_K` crash.**
   - `MSM.py:370` passes `verbose=False` to `_is_adjacent(self,i,j)` (`MSM.py:428`), so it raises `TypeError` on first use [signature mismatch Verified; the crash follows from the signature but was not run]. Same in `CE_Opt/MSM.py:392/450` and all M-A copies.
   - The default `_bias_M(method='logm_K')` also goes through it (`MSM.py:396-401`).
   - `CE_Opt/Cross Entropy Optimization.ipynb` (cell 8) calls `_bias_K` on M-C and sets `msm_temp.dim` instead of `num_dimensions`. It cannot run as written; the cell has no outputs.
2. **C2 1D Gaussian width mismatch.**
   - The optimiser and DHAM use `gaussian = a·exp(-(x-b)²/(2c)²)`, i.e. a 4c² denominator (`1D_langevin_sim/util.py:15-16`; used at `dham.py:11,88` and `util.py:227,292`; same in all F1/F2 copies, `1D_langevin_sim_backforth/util.py:11-12,65`).
   - The bias actually applied in OpenMM is `2σ²` (`1D_langevin_sim/util.py:557`, `1D_langevin_sim_backforth/util.py:363`), as are `get_total_bias` (`:614` / `:420`) and W2 `wham.py:5-6`.
   - [Inferred] Effects:
     - DHAM unbiases with a bias √2 wider than the one simulated.
     - The optimiser ranks a different bias from the one applied.
     - In `1D_langevin_sim_backforth/analysis.py:37-55`, WHAM and DHAM reweight the same trajectories with different bias shapes.
3. **C3 2D DHAM ignores amplitude.** `gaussian_2d(x,y,ax,…)` returns `exp(…)` without `ax` (`2D_langevin_sim/dham.py:17-18`, `2D_model_potential_TW/dham.py:16-17`, `2D_Dialanine/dham.py:17-18`). The util versions multiply by it (`util_2d.py:19-20`, `2D_langevin_sim/util.py:26-27`), and OpenMM applies `Ag·4.184` (`2D_langevin_sim/util.py:636,643`). [Inferred] Every hill is unbiased as 1 kcal/mol whatever `a` was (initial `a=0.01`, `2D_langevin_sim/util.py:38`).
4. **C4 operator precedence.**
   - `np.exp(-u_ij / 2*0.5981)` evaluates to `exp(-0.299·u)` instead of `exp(-u/(2kT)) = exp(-0.836·u)`. The bias effect on M is under-weighted by kT² ≈ 0.36 [Verified syntax; magnitude Inferred].
   - Locations: `1D_langevin_sim/util.py:237,302`, `CE_Opt/util.py:237,302`; `-u_ij / 2*kT` at `1D_NaCl_TW/util.py:208`, `1D_NaCl_TW_backforth/util.py:208`.
   - Correct in `MSM.py:410-411` and `2D_langevin_sim/util.py:217,258`.
   - The same pattern breaks the width of the F4 Gaussians, `-(x-b)**2 / 2*(c**2)` (`RL_Qagent_1D/util.py:11`, `RL_DQN_position/util.py:11`, `RL_DQN_1D_N100/util.py:15`, `RL_pos_PPO_1D/util.py:15`). It is harmless when c=1, e.g. `data_gen.py:20`.
5. **C5 transpose convention after 2959e7a.**
   - D3/D4 return `MM.T`, which is column-stochastic (`1D_langevin_sim/dham.py:141`, `1D_NaCl_TW/dham.py:141`).
   - F1 `try_and_optim_M` and `bias_M_1D` treat M as row-stochastic: row-convention sign, row normalisation (`util.py:233-245`), peq from `eig(M.T)` (`:248`), and `Markov_mfpt_calc = inv(I + 1πᵀ − M)` (`:163-179`, the row-stochastic Kemeny–Snell form).
   - [Inferred] For F1/NaCl the biased chain and MFPT therefore come from the transposed matrix.
   - The D6 + `MSM` path is self-consistent: column storage `:135`, column normalisation `MSM.py:417-422`, `inv(A+I−M.T)` `MSM.py:227`.
6. **C6 2D count convention.** `2D_langevin_sim/dham.py:39` counts `Ntr[k,to,from]`, unlike every other copy (`Ntr[k,from,to]`). So `trvec` (`:43`) counts arrivals; the exponent sign is flipped (`:101`); normalisation is per row (`:108-112`); the return is untransposed (`:160`), while peq uses `MM.T` (`:147`). [Inferred] The sign flip may offset the index swap, but the DHAM denominator then uses destination counts. Needs a check against an analytic FES.
7. **C7** `2D_Dialanine/dham.py:169` returns `(MM.T)*4.184`, which is not a stochastic matrix [Verified].
8. **C8 binning off-by-one.**
   - `np.digitize` is 1-based. With edges `linspace(0,2π,num_bins+1)` (`1D_langevin_sim/dham.py:117-119`), bin 0 is never filled. Any transition touching the last interval raises `IndexError`, which is silently skipped (`:33-36`).
   - `u[i]` is evaluated at the bin's right edge (`:88`).
   - D6 dynamic bins use `num_bins` *edges* (`1D_langevin_sim_backforth/dham.py:123`), so the data maximum always falls in the dropped index.
   - Elsewhere the authors do subtract 1 (`1D_PMg/explore_sim_PMg.py:367`). [Inferred from NumPy semantics]
9. **C9 `num_bins` ignored.**
   - D3/D4/D5 `__init__` never sets `self.num_bins`, so the class default is 150 (`1D_langevin_sim/dham.py:54,57-62`). Callers patch it with `d.num_bins = …` (`langevin_sim_mfpt_opt.py:163`, `1D_NaCl_TW/mfpt_NaCl.py:68`).
   - `1D_NaCl_TW/dham.py:117` builds edges from `config.num_bins` while counting with `self.num_bins`.
   - Stale callers `1D_NaCl_TW{,_backforth}/unbias_traj_fes_gen.py:44` call `DHAM(gaussian_params)` without the required `num_bins`, which raises `TypeError` [signature mismatch Verified; not run].
10. **C10** D6 removed the `if sump1 > 0` guard (`1D_langevin_sim_backforth/dham.py:94-95`). With DHAMsym on (as the callers set it, `langevin_sim_mfpt_opt.py:145`), `sumtr[i,j]>0` can coexist with `Σ_k trvec[k,i]=0`, giving inf and then NaN rows [Inferred].
11. **C11 complex/negative peq.**
    - `MSM._compute_peq_fes_*` never takes `.real` and does not guard `log` of non-positive entries (`MSM.py:55-59,74-78`). `util.compute_free_energy` takes `.real` but not a clip (`1D_langevin_sim/util.py:110-114`).
    - `argsort` on complex eigenvalues is lexicographic.
    - The Kemeny check never fails: the `raise` is commented out (`util.py:43-46`, `MSM.py:291`).
12. **C12 shape accident.** `MSM.py:197` uses `self.peq.T @ onevec` with a 1-D peq, which collapses to the scalar Σpeq, so it inverts `J − Kᵀ` rather than `1πᵀ − Kᵀ` (`util.py:58`). [Inferred, by derivation] The result is still correct, because any `u` with `uᵀ1≠0` gives `πᵀ(1uᵀ−Q)=uᵀ`. It is still fragile.
13. **C13** `RL_DQN_position/util.py:61` uses `inv(peq.T @ onevec − K)` instead of `− K.T` (compare `RL_Qagent_1D/util.py:62`). This is wrong for non-symmetric K [Inferred].
14. **C14 "v unstable" MFPT (cbc3484).**
    - The workaround replaced the eig peq with 1000 fixed power iterations without a convergence test (`util.py:118-130`), and it is reverted.
    - The root cause was never addressed:
      - `get_working_MM` removes zero rows/columns (`1D_langevin_sim/langevin_sim_mfpt_opt.py:115-121`), so the working M can be sub-stochastic or reducible. Then `I+1πᵀ−M` is ill-conditioned and `Markov_mfpt_calc` returns a `1e12` sentinel (`util.py:174-175`), which flattens the optimiser landscape [Inferred].
      - The working M is not renormalised before biasing in M-path code; the only normalisation happens after biasing (`util.py:240-245`).
15. **C15 `mfpt_helper` arguments.**
    - `args=(M,start,end,working_indices)` binds `working_indices` to `kT` (`1D_langevin_sim/util.py:286` vs `:320-323`). Harmless only because `kT` is hard-coded inside.
    - F2 `:104` rebuilds the MSM with the default `time_step=0.1` instead of `dcdfreq·dt` (`:48,56`). The argmin is unaffected, but the logged MFPTs mix units.
16. **C16** `MSM._bias_M('direct_bias')` copies only nearest-neighbour entries plus the diagonal (`MSM.py:404-414`). Multi-bin DHAM transitions are discarded before renormalising [Inferred impact].
17. **C17 paper model inconsistencies.**
    - `paper_publication/1D_K_opt_recalc/main.py:42-47` and `figure_gen_1D.py:125-131` build a fully connected K whose columns sum to −1, because the self term `exp(0)` is included. [Inferred] This is harmless for F (detailed balance, eigenvectors unchanged by −I), and `_bias_K` then rebuilds a nearest-neighbour generator (`MSM.py:367-376`).
    - The walls differ between scripts: 100 kcal/mol at −0.4 (`figure_gen_1D.py:121-122`), 100/4.184 kcal/mol (`main.py:36-37`), and 100 kJ/mol at +0.6 in the MD (`1D_langevin_sim/util.py:338`).
    - The CE_Opt and pre-c17d449 left wall `exp({k} * x - (-{offset}))` (`CE_Opt/util.py:338`) is a precedence slip and differs from its own plot formula (`:446`).
18. **C18 environment fragility.** `np.float_` appears in all 1D DHAM `bootstrap_error` (e.g. `1D_langevin_sim/dham.py:154,163`; removed in NumPy 2). `plt.cm.get_cmap` is used in `figure_gen_1D.py:140` (removed in Matplotlib 3.9). The pinned versions are numpy 1.24.3 and matplotlib 3.7.1 (`environment.yml:141,165`).

## Consolidation proposal (proposal only; nothing done)

1. **Package.** Create a top-level `rlmfpt/` package (`pyproject.toml`, `pip install -e .`) with these modules:
   - `msm.py`, from **M-B**, plus the paper FES test case from M-C.
   - `dham.py`, from **D6** for 1D, with a `Dham2D` class ported from **D2**.
   - `bias.py`, from **F2** `try_and_optim_M` with the OpenMM `apply_fes/apply_bias/update_bias/get_total_bias` helpers.
   - `wham.py`, from **W2**, keeping W1's umbrella reader as an optional path.
   - `kmodel.py`, from the F4 K-matrix helpers, taking `RL_posM_env/util.py` as the base because it has the fixed Gaussian.
   - `models2d.py`, from **U2a**.
   - Keep `config.py` per experiment, but pass values in explicitly rather than having library code `import config` (D4–D6 and F1/F2 read `config.qspace`/`num_bins` globally).
2. **Single conventions, fixed in one place:**
   - one Gaussian (`2σ²`, with amplitude) shared by the optimiser, DHAM, WHAM and OpenMM;
   - one stochastic-matrix orientation;
   - 0-based binning;
   - `.real` and a clip on peq.

   Each fix should get a regression test on an analytic 1D/2D FES: DHAM recovers F, the MFPT matches the closed form, and the Kemeny constant is constant.
3. **Migration.** Change imports directory by directory. Leave the existing copies until each experiment is re-run, since deleting them breaks `from util import *`.
4. **Risks:**
   - **Fixing C2–C5 changes the numbers.** Results already in `paper_publication/` came from the buggy paths and will not reproduce exactly.
   - **API differences.** D3 vs D6 return a 2-tuple vs a 3-tuple. `DHAM(gaussian_params)` vs `DHAM(global_gaussian_params, num_bins)`. util F1 vs F2 expose different names.
   - **Star imports.** `from util import *` and `from MSM import *` hide which names are used.
   - **Hard dependencies.** CUDA/OpenMM at import time (`config.py` calls `getPlatformByName('CUDA')`), and numba inside `MSM`.
   - **Hidden MATLAB step.** The MATLAB-produced `.mat` inputs stay outside the package.

## Redundancy table (`paper_publication/` and repo root only)
| path | size | evidence | refs found | class |
|---|---|---|---|---|
| `paper_publication/figure_gen_2D.py` | 10.2 kB | md5 = `2D_model_potential_TW/figure_gen_2D.py`; cannot run here (no `util_2d`, data, PNG) | none (as input) | **DELETE-SAFE** |
| `paper_publication/pos_bias_m.mat` | 20 kB | md5 = `1D_model_potential_Simian/` and `2D_model_potential_TW/pos_bias_m.mat` | only a commented `figure_gen_1D.py:18` | **DELETE-SAFE** |
| `paper_publication/data/traj/2D/MSMopt/20231120-162110_langevin_sim_explore_0.pdb` | 159 B | md5 = `data/traj/2D/system.pdb`. It is the per-run topology written next to each `.dcd` (`2D_langevin_sim/langevin_sim_mfpt_opt.py:51`) and pairs with the gitignored `.dcd`. `analysis.py:114` uses `system.pdb` instead | none by name | **CONFIRM** |
| `paper_publication/.vscode/launch.json` | 473 B | stock VS Code "current file" template | none | **DELETE-SAFE** |
| `test.png` (root) | 19.6 kB | scratch output name used by many scripts (`plt.savefig("./test.png")`); added with MSM in 5d614fc | output only | **DELETE-SAFE** |
| `fes_visualization.png` (root) | 24.7 kB | default `plot_path` of `apply_fes` (`util.py:330`); committed in c93d7f8 | output only | **DELETE-SAFE** |
| `MSM.py` (root), `paper_publication/MSM.py` | 22.6 kB each | M-A with C1; byte-identical | **no importer** | **CONFIRM** (core-module copies; seed for the package) |
| `paper_publication/fig1_30Gaussian.mat`, `pos_bias_1stMar_1.mat`, `pos_bias_29thFeb_2.mat`, `pos_bias_m_langevin_system.mat` | 10–319 kB | MATLAB results, Feb–Mar 2024 (header) | comments only (`figure_gen_1D.py:19,23`) or none | **CONFIRM** (research results) |
| `paper_publication/figs/*.png` (12) | 6.3 MB total | likely the paper figures [Inferred]; `2Dtraj_*` not regenerable | outputs of `analysis.py` | **CONFIRM** |
| `paper_publication/data/time2reach/2D/metaD_bak/total_steps_metaD.csv` | 361 B | superseded by recalc; skipped by `os.listdir` (dir) | none | **CONFIRM** |
| `data_1D_20N_10Gaussian.pt` (root) | 75.2 MB | `data_gen.py:47` output; differs from `data/` copy; unseeded | none (reader uses `../data/`) | **CONFIRM** |
| `kinetic_networks.py` (root) | 6.8 kB | unique source; `NameError` at `:42-46` | none | **CONFIRM** |
| `figures.pptx` (root) | 3.9 MB | 5bd394d "added figure generations" | none | **CONFIRM** |
| `pos_M_DQN_model_{11thJuly_1,13July_1,13July_2,13July_2_N100,14July_1_N20}.pt` (root) | 0.95–20.7 MB | all md5 distinct; `14July_1_N20` is written by `RL_posM_env/posM_dqn.py:211` | none as input | **CONFIRM** |
| `pos_M_DQN_model_13July_2_N20.pt` (root) | 954 kB | loaded by `RL_posM_env/main.py:50` (cwd-relative, so it only works from root) | 1 | **KEEP** |
| `data_gen.py` (root) | 1.9 kB | unique generator of the `.pt` dataset | none | **KEEP** |
| `paper_publication/{analysis,figure_gen_1D}.py`, `1D_K_opt_recalc/{main,MSM}.py`, `data/time2reach/recalc_metad_reachtime.py` | 1.7–22.6 kB | unique sources; `1D_K_opt_recalc/MSM.py` imported at `main.py:13` | ≥1 | **KEEP** |
| `paper_publication/pos_bias_1stMar_2.mat`, `data/time2reach/{1D,2D,NaCl}/*.csv`, `data/traj/2D/{system,metad/explore_traj}.pdb` | ≤25 kB | read by `figure_gen_1D.py:20`, `analysis.py:31-49,114`, `recalc…py:16` | ≥1 | **KEEP** |

## Open questions for the authors

1. Which Gaussian convention was intended, `(2c)²` (util, DHAM, optimiser) or `2σ²` (OpenMM, WHAM, `get_total_bias`)? Were the paper's 1D/NaCl MFPT-opt runs made with the C4 precedence bug and the C5 transpose present (D3/D4 + F1, dated after 2959e7a)?
2. Is the 2D DHAM's dropped amplitude (C3) and `Ntr[k,to,from]` (C6) deliberate? Was the 2D DHAM ever validated against the analytic multiwell FES?
3. Which `.mat` file is the final 1D optimal bias (`pos_bias_1stMar_2.mat` is hard-coded), and which MATLAB script produced it? Is `1D_K_opt_recalc` meant to replace it? Its output is unused.
4. Are the n=1 "Classical MD" values for 1D/2D (hand-entered) and n=5 metaD for 2D what the paper reports? Is the cap of 4999500 in `metaD_bak` a timeout?
5. Should `fes_digitize.png`, an input to `2D_model_potential_TW` (`util_2d.py:62`, `figure_gen_2D.py:20`) deleted in 45d011f, be restored from `2D_Dialanine/figs/`?
6. Should `.gitignore:34-35` read `/paper_publication/data/*`? As written they ignore nothing; with the fix the tracked `time2reach` CSVs would remain tracked but new data would be ignored.
7. Is root `data_1D_20N_10Gaussian.pt` or `data/data_1D_20N_10Gaussian.pt` the dataset behind any reported offline-RL result?

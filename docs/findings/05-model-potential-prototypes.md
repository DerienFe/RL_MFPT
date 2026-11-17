# 05: Model-potential prototypes (MATLAB/Python) and CE_Opt

Scope: `1D_model_potential_Edina/`, `1D_model_potential_Simian/`, `1D_model_potential_TW/`, `2D_model_potential/`, `2D_model_potential_TW/`, `CE_Opt/`.
Method: read-only reading of code and git history (`git log`, `git show`, `git log -S`), md5 comparisons, and parsing of `.mat` headers and variable names with a scratch stdlib script. Nothing in the repo was executed: no MATLAB, numpy or scipy is installed here.
Labels: **[Verified]** means backed by a file:line, a diff or command output. **[Inferred]** covers everything else, including any statement about runtime behaviour.

## Summary
- These are the project's earliest prototypes (May–Oct 2023). They work on discrete rate matrices `K` for a 1D 100-state potential and a 2D grid potential. Each one searches for Gaussian bias parameters that minimise the MFPT, using random search followed by `fminsearch`, `fmincon` or Nelder-Mead. **[Verified]** (`Eigenvector_1D_model.m:61-93`, `main_optbias.m:140`, `1D_model_potential_TW/main.py:124,206`)
- Lineage **[Verified]**:
  - Edina and Simian both appear in 7529485 (2023-05-19). Simian's `gaussmf.m` is older (900fbb3, 2023-05-15).
  - Edina has not changed since 2023-05-19. In acbf717 (2023-07-17) Simian was brought into line with Edina and extended: it gained `Markov_mfpt_calc.m`, `test.mat` and `benchmark.mat`; `mfpt_calc.m` was changed to Edina's signature; `Eigenvector_1D_model.m` was rewritten (+72/−191 lines across 4 files). So the shared files flowed Edina → Simian.
  - Simian's RL work continued in `RL_matlab/` (2023-08-09).
  - The Python ports are `1D_model_potential_TW` and `2D_model_potential_TW`.
  - The `langevin_approach/` sub-project inside `2D_model_potential_TW` moved to `2D_langevin_sim/` in Oct 2023.
  - `CE_Opt` (2024-04-24) is an unfinished copy of `1D_langevin_sim` with cross-entropy scaffolding.
- Neither `2D_model_potential_TW/main.py` nor `figure_gen_2D.py` can run in a fresh checkout. Their input `fes_digitize.png` was deleted by 45d011f ("Remove .png files", 2024-11-06). It can be recovered with `git show 45d011f^:2D_model_potential_TW/fes_digitize.png`. **[Verified]** (`util_2d.py:62`, `figure_gen_2D.py:20`)
- The brief's duplicate list is partly wrong. Only `compute_free_energy.m` is byte-identical in all four MATLAB folders. In `2D_model_potential`, `Markov_mfpt_calc.m`, `explore_the_network.m` and `Unbiasing_eigvector_data.m` differ from the 1D copies. **[Verified]** (md5 table below)
- 35 MB of the 37 MB in `2D_model_potential_TW` are run outputs under `figs/`; the 14.5 MB `movie.gif` alone accounts for 41%. Only the 13 `data/20230927-135127_*` files are read by current code. **[Verified]**

## Per-folder table
| Folder | Lang | What it models | Entry point / how to run | Status / successor |
|---|---|---|---|---|
| `1D_model_potential_Edina` (14 files, 1.0 MB) | MATLAB | 100-state 1D tilted sine potential (`create_K_1D.m:4-11`). Random 5-Gaussian bias search minimising Markov MFPT 45→89, then stochastic validation (`Eigenvector_1D_model.m:17-130`). Also an umbrella-sampling + DHAM pipeline. | In MATLAB: `cd` into the folder, then run `Eigenvector_1D_model`. The pipeline `Eigenvectors_1D_model_several_exp` → `several_umbrella_samp_exper` → `Unbiasing_eigvector_data` is broken (see Bugs). | Frozen snapshot: all commits are dated 2023-05-19. Simian adopted 12 of its 14 files byte-for-byte in acbf717, and its `DHAM_unbias.m` is identical to `2D_model_potential`'s. Its only unique content is `state_start=45` (`Eigenvector_1D_model.m:17`). **[Verified]** |
| `1D_model_potential_Simian` (45 files, 6.2 MB) | MATLAB | Same model. Adds per-start-state optimal bias tables (`pos_bias*.mat`), adaptive and predictive simulations with `DHAM_sym.m`, `fminsearch`/`fmincon` optimisation and an RL stub. | `main_optbias.m`; `main_adaptive_1D.m` and `main_adaptive_1D_Simian.m` (need `pos_bias*.mat`); `main_predictive_1D.m`; `main_RL_adaptive_1D.m` (deliberately halts at `:51`). | Last commit acbf717 (2023-07-17). The RL part continues in `RL_matlab/`, where `main_RL_adaptive_1D.m` and `propagate_N_steps.m` have diverged. **[Verified]** |
| `1D_model_potential_TW` (2 files, 18 KB) | Python | Python port of the Simian/Edina 1D workflow. Each MATLAB helper has a Python counterpart: `create_K_1D`, `compute_free_energy`, `mfpt_calc`, `markov_mfpt_calc`, `explore_the_network`, `min_mfpt`, plus a stubbed `DHAM_unbias` (`util.py:10-261`). | `cd 1D_model_potential_TW && python main.py` (it imports `util` from the working directory). | Yes, it is a port. The 7529485 version copies `Eigenvector_1D_model.m` (`state_start = 9`, 5 Gaussians). 835493b adds Nelder-Mead, following `main_optbias.m`. `main.py` was last changed on 2023-09-08. The MSM/Langevin code is the likely successor. **[Verified]** for the port; **[Inferred]** for the successor |
| `2D_model_potential` (33 files, 5.5 MB) | MATLAB | 13×13 grid on [-3,3]² with a 4-well log-sum potential (`create_K_2D.m:9-12`). Random 20-Gaussian search, then `fminsearch`, `GlobalSearch` and `MultiStart` on the MFPT from (-2,-2) to (2,2) (`Eigenvectors_2D_model_several_exp.m:60-191`). | `Eigenvectors_2D_model_several_exp.m`. Also `check_distribution_of_biases.m`, `free_energy_surface_2D.m` and `Unbiasing_eigvector_data.m`. | One commit only (835493b, 2023-06-12). Its Python counterpart was started in the same commit. **[Verified]** |
| `2D_model_potential_TW` (639 files, 37.3 MB) | Python | Started as a port of `2D_model_potential` (835493b `main.py` used `create_K_2D`, `mfpt_calc_2D` and a 4-index MFPT). From Sept 2023 it became a 20×20 MSM on an FES digitised from `fes_digitize.png`, with an adaptive loop: propagate → DHAM → re-optimise the bias (`main.py:164-339`). | `main.py` (`do_explore_run.sh` runs it 10 times), `2D_presentation_movie_plot.py`, `un_biased_run.py`, `figure_gen_1D.py`, `figure_gen_2D.py`, `gif_gen.py`. All need the working directory to be this folder, because I/O uses `./figs`, `./data` and `./fes_digitize.png`. | Code last changed 2024-04-24. `langevin_approach/` was deleted here in de96c82 (2023-10-20) and recreated as `2D_langevin_sim/` in a7914b5 (2023-10-23), where `fes_digitize_*` also moved into `bak/`. So `2D_langevin_sim` is the methodological successor (continuous OpenMM Langevin). It is not a drop-in replacement for this discrete-K prototype. `2D_Dialanine/` (7251d34) is a sibling fork. **[Verified]** |
| `CE_Opt` (11 files, 1.1 MB) | Python | Intended to replace the `scipy.minimize` call in `try_and_optim_M` (`util.py:181,318`) with a cross-entropy method (`readme.md:1-5`). | No working entry point. `CE_mfpt_opt.py` fails on import. `langevin_sim_mfpt_opt.py` is a byte-copy of `1D_langevin_sim`'s and never uses CE. | Unfinished, one commit (c17d449, 2024-04-24). Nothing imports it. **[Verified]** |

## MATLAB duplication map
The table shows the md5 prefix of each file in each folder. Folders marked "same" share the first hash in their row. "-" means the file is absent. **[Verified]**

| file | Edina | Simian | 2D_model_potential | RL_matlab |
|---|---|---|---|---|
| compute_free_energy.m | 327b88 | same | same | same |
| Markov_mfpt_calc.m | e8d0fa | same | **b635ef** (reshapes to n×n×n×n, `:6,19`) | same |
| explore_the_network.m | c4df53 | same | **436ff9** (umbrella-bias version, 11 inputs and 4 outputs, `:1-12`) | same |
| mfpt_calc.m | ad19ac | same | - | same |
| several_umbrella_samp_exper.m | 916135 | same | same | - |
| Unbiasing_eigvector_data.m | 334409 | same | **0a5bc8** (one blank line added, 46a47) | - |
| DHAM_unbias.m | b3207a | **685b49** (see below) | same as Edina | - |
| create_K_1D.m | 19d606 | same | - | **481013** (2 blank lines removed) |
| Eigenvector_1D_model.m | 4fbe67 | **1c24c8** | - | - |
| Eigenvectors_1D_model_several_exp.m, gaussmf.m, visualize_1D_model_potential.m, test.mat, benchmark.mat | identical pairs | same | - | - |
| main_RL_adaptive_1D.m, propagate_N_steps.m | - | 256a53 / 701136 | - | evolved (afc91e / 70d859) |

Files that differ between Edina and Simian **[Verified]**:
- `DHAM_unbias.m`. Edina's version uses a harmonic umbrella bias, `(Record_states, x_eq, force_constant, kT, n_states, bias_order_win, cutoff)`, and is identical to `2D_model_potential`'s. Simian's version uses a Gaussian bias, `(trajectory, bias_traj, kT, N, datlength, cutoff, amp)`, returns `[prob_dist, M0]`, raises the count threshold from `nc > 2` to `nc > 3`, and plots `mU2` (diff hunks at 1-13, 29-30, 33, 58-65).
- `Eigenvector_1D_model.m`. Line 17 sets `state_start=45` in Edina and `=9` in Simian; Simian also adds a comment at `:89-90`.
- Simian alone has 18 more `.m` files (counting the conflict copy), `pos_bias*.mat`, `test.ma`, `test2.mat` and `Figures_Simian/`.

Risk: same-named functions have different signatures (`explore_the_network`, `Markov_mfpt_calc`, `DHAM_unbias`). If more than one of these folders is on the MATLAB path, the wrong one will silently shadow the other. **[Inferred]**

`.mat` and `.fig` files: inputs or outputs? (variables read from the MAT headers) **[Verified]**

| File(s) | Contents | Role and evidence |
|---|---|---|
| `Simian/pos_bias.mat` | `pos_bias` cell 1×89 | **Input:** `main_adaptive_1D.m:51`, `main_RL_adaptive_1D.m:63`. Its generator is commented out at `main_adaptive_1D.m:43-50`. |
| `pos_bias_m.mat` ×3 (Simian, 2D_TW, paper_publication; identical) | `pos_bias` cell 1×89 | **Input:** `main_adaptive_1D_Simian.m:54` and `2D_model_potential_TW/figure_gen_1D.py:16-17`. `paper_publication/figure_gen_1D.py:18` has its load commented out; line 20 loads `pos_bias_1stMar_2.mat` instead. |
| `2D_model_potential/data_eigenvector_{1..5}_.mat` (v7.3) | `Record_states`, `x_rec`, `pi_biased`, `x_eq`, `Num_exper`, … | Written by `several_umbrella_samp_exper.m:65`. **Input** to `Unbiasing_eigvector_data.m:10`, which overwrites its own input at `:79`. |
| `test.mat` (Edina = Simian) | Workspace of `Eigenvector_1D_model.m` (`C_g` 1×5, `Record_states` 10×1e7, `mfpt_from_Markov`, …) | Output saved by hand. Nothing loads it. |
| `benchmark.mat` (Edina = Simian) | `F`, `K`, `KbT`, `kemeny0`, `reltime0`, `mfpt`, `peq` | Nothing loads it. No tracked code mentions `kemeny0` or `reltime0`. `KbT` appears in `create_K_2D.m:4` and in commented-out calls to a `min_reltime(v,K,N,KbT)` that is not in the repo (`main_optbias.m:139`, `main_adaptive_1D.m:46`, `main_RL_adaptive_1D.m:58`). The missing relaxation-time script is the likely source. **[Inferred]** |
| `Simian/test.ma` | Workspace dump with a truncated extension (`Record_states` 1×1e7) | Nothing loads it. |
| `Simian/test2.mat` (4.1 MB) | 547×547 `Adj`/`K`/`M`, `best_split_PCCA`, `best_split_HS`, `chi` | Unrelated network-clustering workspace. Nothing loads it. |
| `2D_model_potential/opt_params_{fmin,gs,ms}.mat` | 20×4 Gaussian parameters | **Outputs** of `Eigenvectors_2D_model_several_exp.m:102,144,172`. The comment at `:77` suggests loading them, but there is no `load()`. |
| 16 `.fig` files (`Figures_Simian/untitled*.fig`, `bar_plot.fig`, `2D_model_potential/figures/*.fig`) | MATLAB figures | Saved by hand from the GUI. No code writes or reads these names; the only `savefig` calls write `data_eigenvector_*_{prob_dist,free_energy}_.fig` (`Unbiasing_eigvector_data.m:45,77`), and none of those is tracked. |

## 2D_model_potential_TW outputs
All rows are **[Verified]** by `git ls-files`, `git grep` and `git log -S`, unless a row says otherwise.

| Output (tracked) | Count / size | Generator | Read by |
|---|---|---|---|
| `figs/<tag>_<n>_{traj,total_bias,fes_biased}.png` | 119+88+88 | `main.py:87,289,301`; `2D_presentation_movie_plot.py:92,303,315` | nothing |
| `figs/<tag>_<n>_DHAM.png` | 119 | `dham.py:166` | nothing |
| `figs/{DHAM,traj}_<tag>_<n>.png`, `biased_fes.png`, `unbiased.png` | 3+3+1+1 | Old names: the writing code was added in 4bdfd8f and removed in 3fa7301 (`git log -S'DHAM_{'`). `biased_fes.png` was written by code added in 706ae7b and later removed. `unbiased.png` comes from `figure_gen_2D.py:34`. | nothing |
| top-level `figs/*.png` total | 422, 7.83 MB, 35 run tags from 2023-09-22 to 2023-09-28 | as above | nothing |
| `figs/1D_gif/optim-*_overlap.png` | 30, 1.86 MB | `figure_gen_1D.py:176` (from `pos_bias_m.mat`) | `figure_gen_1D.py:183-196`, inside `if False:` |
| `figs/1D_gif/animation{,_2}.gif` | 2, 2.90 MB | `animation_2.gif` comes from `figure_gen_1D.py:189` (disabled). No commit ever wrote `animation.gif` (`git log -S` finds nothing), so it was probably made by hand. **[Inferred]** | nothing |
| `figs/prod_figs/traj0-6.png` | 7, 3.07 MB | `figure_gen_2D.py:147-153`, from run `20230927-135127` | nothing |
| `figs/prod_figs/movie_20230925-131244/{traj1-9,unbiased}.png` | 10, 5.0 MB | Added in e1e9276. In that commit, `figure_gen_2D.py` loaded `data/20230925-131244_*` and saved with `dpi=600`. The frames were then renamed into this folder. **[Inferred]** | `gif_gen.py:4-16` |
| `.../movie_20230925-131244/movie.gif` | 1, 14.5 MB | `gif_gen.py:12`, deterministic from the tracked frames | nothing |
| `data/<tag>_<n>_gaussian_params.npy`, `data/<tag>_<n>_CV_total.npy` | 108, ~1.13 MB | `main.py:236`, `util_2d.py:503,506`, `2D_presentation_movie_plot.py:250` | **only `20230927-135127_*` (13 files):** `figure_gen_2D.py:38-67` |
| `data/CV_total_<tag>_<n>.npy`, `data/gaussian_params_<tag>_<n>.npy` | 10+22, 269 KB | Older naming (2023-09-19 to 09-22) | nothing |
| `data/unbiased_mfpt_{raw,mean_std}_(14, 14)_(4, 6).txt` | 2 | `un_biased_run.py:55-56`. Current code uses `state_end=(4,7)` (`:20`). | nothing |

- **Can the outputs be regenerated?** Not exactly. The RNG is never seeded (`util_2d.py:26,335,450`; `main.py:59`), and every script needs the deleted `fes_digitize.png`. `movie.gif` is the exception: it can be rebuilt from its tracked frames. **[Verified]** for the code; **[Inferred]** for runtime behaviour
- **Ignore rules.** `/2D_model_potential_TW/data*` and `/figs*` were added in 849625c (2023-10-16), found with `git log -S`. The last files were added to these folders on 2023-10-02 (a56cdfa), and nothing has been added since 2023-10-16. Later commits only modified tracked `1D_gif` PNGs (c8c9cdb, c17d449). **[Verified]**
- **Untracking needs care.** Running `git rm --cached` on `data/` would leave fresh clones without the 13 files that `figure_gen_2D.py` needs. A `!` exception cannot re-include them while the rule ignores the `data` directory itself; the rule would have to become `/2D_model_potential_TW/data/*`. **[Inferred]** (documented git behaviour)
- **`bak/`.** It holds 6 PNGs and 5 parameter files from fitting the digitised dialanine FES. They were produced by `2D_langevin_sim/fit_dialanine_{gauss,fourcos}.py` (`:87,93`; `:131,137,168,174`) and moved here in a7914b5. **[Verified]**
  - `bak/fes_digitize_gauss_params_15.txt` has the same md5 (54ca6b…) as the deleted `2D_langevin_sim/params/gaussian_fes_param.txt`, which `2D_langevin_sim/test_util.py:10` still loads.
  - `_15_2.txt` is loaded by `fit_dialanine_gauss.py:65` when `test` is set.
  - In both cases `bak/` holds the only tracked copy.
- **`figure_gen_2D.py` duplicate.** It is byte-identical to `paper_publication/figure_gen_2D.py`. In this checkout, only the `2D_model_potential_TW` copy has its inputs: `paper_publication` has no `util_2d.py` (`git ls-files`) and no `data/20230927-135127_*`. `paper_publication/fes_digitize.png` was also deleted in 45d011f. Some of these may simply be gitignored rather than missing. **[Verified]**
- **`.vscode/launch.json`** is the stock VS Code "Python: Current File" template with nothing project-specific in it (`:7-13`). **[Verified]**
- **`copy_of_first_simulation.ipynb`** is an OpenMM/Colab single-particle prototype, a precursor of `langevin_approach`. Cell source contains `t_parameter = ` with no value (`:135`), which is a syntax error, and a stored output shows a Windows temp path, `C:\Users\13401\...` (`:102`). **[Verified]**

## CE_Opt
- **Own code vs. copied code.** **[Verified]**
  - Byte-identical to `1D_langevin_sim/`: `dham.py`, `config.py`, `langevin_sim_mfpt_opt.py` (duplicate_groups.txt).
  - `util.py` differs from `1D_langevin_sim/util.py` in 3 hunks (6 lines). `:338` and `:446` change the left boundary-barrier offset; `:440` changes the `linspace` grid end.
  - `MSM.py` is equally close to `paper_publication/MSM.py` and `1D_langevin_sim_backforth/MSM.py` (40 differing lines each).
- **`cross_entropy_method/`** is a vendored third-party library: "Written by Ido Greenberg, 2022" (`CEM.py:47`). There is no licence file. Nothing outside the package imports it; the only import in `CE_mfpt_opt.py:4` is commented out. **[Verified]**
- **`CE_mfpt_opt.py`** is the authors' own unfinished code. A function `CEM` is defined at `:7` and then used as a base class at `:70`, which raises `TypeError: function() argument 'code' must be code, not str`. I confirmed this with a two-line stdlib reproduction in the scratchpad. `update_sample_distribution` has no return (`:97-99`). **[Verified]**
- **`Cross Entropy Optimization.ipynb`** (982 KB, 22 cells, 12 embedded PNG outputs). It contains a generic CE optimiser plus `sphere`/`griewank` test functions. An `mfpt()` objective is defined (`:241`), but the actual run optimises `griewank` (`:454`). It imports `MSM` from the working directory (`:12`). **[Verified]**
- **Users.** None: `git grep` finds no references to `CE_Opt`, `CE_mfpt_opt` or `cross_entropy_method` outside the folder. **[Verified]**

## Bugs/smells (file:line)
MATLAB:
- `1D_*/Eigenvectors_1D_model_several_exp.m:14` calls `create_K_1D(N,kT,Barrier_height_1,Barrier_height_2)`, but `create_K_1D.m:1` takes 2 arguments. This has been the case since the first commit (7529485). **[Verified]**; it errors at runtime **[Inferred]**
- `1D_*/several_umbrella_samp_exper.m:40` calls `explore_the_network` with 11 inputs and 4 outputs. The 1D `explore_the_network.m:1` takes 10 inputs and returns 2. Only `2D_model_potential/explore_the_network.m:1-12` matches, and no driver in that folder calls the umbrella code. **[Verified]**
- `1D_*/gaussmf.m:1` `function[out] = gaussmf(X, [SIGMA, C])` is not valid MATLAB syntax. It also shadows the Fuzzy Logic Toolbox `gaussmf`, and nothing calls it. **[Verified]** for the code; that it fails to parse is **[Inferred]**
- `1D_*/explore_the_network.m:13-16`: `step` is only set if `state_end` is reached, so the output can be undefined. **[Verified]**
- `1D_*/visualize_1D_model_potential.m:33-40`: the y-labels are swapped (F is labelled "probability"). **[Verified]**
- `*/Unbiasing_eigvector_data.m:41`: the standard error divides by `sqrt(size(prob_record,2))`, which is n_states rather than Num_exper. **[Inferred]**
- `create_K_1D.m:9-10` and `visualize_1D_model_potential.m:17-18`: the first assignment to `y` is dead code. **[Verified]**
- `Simian/main_adaptive_1D_Simian.m:163,170,176-177`: `step_list = zeros(Nsim)` creates an N×N matrix, then the undefined `step` is used. `:183` has a bare `stoppoint` (a deliberate halt). **[Verified]**
- `Simian/main_RL_adaptive_1D.m:51`: the stray token `lkretjlekr` halts the script. **[Verified]**
- `Simian/min_logM_mfpt.m:9-13`: the bias construction is commented out, so the objective does not depend on `v`. **[Verified]**
- `Simian/predict_bias.m:19-20`: LB and UB have length Ngauss, but `v` has 2·Ngauss elements. `main_predictive_1D.m:55` also passes `v=bias`, an N-length bias profile, as if it were Gaussian parameters. **[Verified]** for the code; the consequence is **[Inferred]**
- `Simian/main_optbias (Edina Rosta's conflicted copy 2023-05-23).m` is a Dropbox conflict copy. Its name is not a valid script identifier. **[Verified]**
- `2D_model_potential/gaussian2d.m:4`: a parenthesis is misplaced. The code computes `exp(-((x-x0)²/(2σx²) + (y-y0)²)/(2σy²))`. The y-term is correct, but the x-term is divided by 2σx² and then again by 2σy². **[Verified]**
- `Eigenvectors_2D_model_several_exp.m:43`: `peq_biased` is computed from the unbiased `K`, not `K_biased`. At `:153,180`, `total_bias_gs/ms = total_bias + …` is overwritten on every iteration instead of accumulated. **[Verified]**
- `random_find_min_mfpt.m:10` (`ts = 0.1` printed and ignoring the caller), `perturb_K_2d.m:6`, `min_mfpt_helper.m:12` and `coord_to_index.m:14` (spacing 0.5, valid only for N=13) are hard-coded. `random_find_min_mfpt.m:21-23` seeds the minimum with the rate-matrix MFPT, but later compares it with Markov MFPT·ts. **[Verified]** for the code; that the comparison mixes estimators is **[Inferred]**
- `create_K_2D.m:36-44` opens a figure as a side effect of building K. `check_distribution_of_biases.m:45` builds `K_biased` without using `K` (possibly intentional). **[Verified]**/**[Inferred]**

Python:
- `1D_model_potential_TW/util.py:84`: `peq.T @ onevec` with a 1-D `peq` gives a scalar, whereas MATLAB builds an N×N outer product (`mfpt_calc.m:9-10`). `2D_model_potential_TW/util_2d.py:196,284` uses the corrected form, with the comment "was peq.T @ onevec.T". **[Inferred]** (numpy semantics; not executed)
- `1D_model_potential_TW/util.py:122` returns `K_biased.T` (rows sum to 0). `main.py:104` and `util.py:202` then pass it to `compute_free_energy` without transposing, so the stationary vector would be uniform. `util.py:172` omits MATLAB's `A=A'` (`Markov_mfpt_calc.m:9`). **[Inferred]**
- `1D_model_potential_TW/util.py:11`: the Gaussian uses `(2*c)**2` where MATLAB uses `2*std^2`. `main.py:184-185` hard-codes `[8, 88]` although `state_start=7` (`:19`). `main.py:132` and `:206` raise a NameError if the 10 random tries (`:124`) never improve. `util.py:161` has an unbound `step`. **[Verified]**
- `2D_model_potential_TW/util_2d.py:62` ignores its `img_path` argument. `:74` subsamples with `int(shape/N)`, which may not give N×N. `:246,250` and `dham.py:74,99-100` hard-code 20×20. **[Verified]**
- `util_2d.py:418-499`: `try_and_optim_M_K` ignores its `M`/`working_indices` arguments and rebuilds the true `K` from the PNG (`:437`). The adaptive loop in `main.py:228` therefore optimises on ground truth, not on the DHAM estimate. **[Verified]**
- `dham.py:16-17`: `gaussian_2d` drops the amplitude `ax`, which `util_2d.py:20` applies (a=0.1 at `util_2d.py:27`, a=2 at `:452`). The DHAM bias therefore differs from the simulated one. **[Verified]**
- `dham.py:136`: `np.digitize` returns 1-based bins over `linspace(0,401,401)`, which shifts states by one; `count_transitions` swallows the resulting `IndexError` (`:36-39`). **[Inferred]**
- `util_2d.py:361,395` normalises each row by the column sums (`axis=0` combined with `[:,None]`). **[Inferred]**
- `main.py:23` sets `state_end=(4,6)`, but `un_biased_run.py:20`, `figure_gen_2D.py:16` and `2D_presentation_movie_plot.py:24` use `(4,7)`. **[Verified]**
- `gif_gen.py:9` compares a bare filename with a path, so `unbiased.png` is added twice. `figure_gen_1D.py:179` comments out `import imageio`, which `:189` uses (inside `if False`). `fig_digitizer.py:1` imports `cv2` without using it, and `:53` runs at import. **[Verified]**
- No absolute paths appear in the code (git grep). All I/O is relative to the working directory, except `figure_gen_1D.py:15`, which resolves paths from `__file__`. The `.mat` headers say `PCWIN64`. **[Verified]**

## Third-party deps
- **Python** **[Verified]** (git grep of imports):
  - Core: `numpy`, `scipy` (linalg, optimize, io, stats, constants), `matplotlib` (including `mpl_toolkits`).
  - Used in 2D_TW: `Pillow` (`util_2d.py:12`), `tqdm`, `imageio` (`gif_gen.py:1`).
  - Imported but unused: `opencv-python` (`fig_digitizer.py:1`).
  - CE_Opt: `openmm` (`config.py:4-5`, `util.py:12`), `mdtraj` (`langevin_sim_mfpt_opt.py:16`), `pandas` (`cross_entropy_method/CEM.py:52`).
  - Lazy imports: `numba` (`CE_Opt/MSM.py:263`), `plotly` (`CE_Opt/util.py:479`).
  - Notebook only: `condacolab`.
- **MATLAB toolboxes** **[Verified]**:
  - Statistics and ML (`randsample`, e.g. `explore_the_network.m:9`).
  - Optimization (`fmincon`, `optimoptions`, `createOptimProblem`).
  - Global Optimization (`GlobalSearch`, `MultiStart`, `Eigenvectors_2D_model_several_exp.m:139,168`).
  - Reinforcement Learning (`createMDP`, `main_RL_adaptive_1D.m:46`).
  - Parallel Computing (`parfor`, `Eigenvectors_1D_model_several_exp.m:44`). Without the toolbox the loop runs serially. **[Inferred]**

## Redundancy table
I grepped the whole repo for each basename or pattern with `git grep`, excluding `docs/`. "refs" counts code references found. Counts, sizes and refs are **[Verified]** (`git ls-files`, `tracked_sizes.tsv`, `git grep`). The classes are judgements.
| path/glob | count | size | evidence | refs | class |
|---|---|---|---|---|---|
| `1D_model_potential_Simian/main_optbias (Edina Rosta's conflicted copy 2023-05-23).m` | 1 | 4.3 KB | Dropbox conflict copy. It differs from `main_optbias.m` only in `state_start` 1 vs 8, `Ngauss` 20 vs 10 and one plot line. | 0 | DELETE-SAFE |
| `1D_model_potential_Simian/test.ma` | 1 | 463 KB | Unique workspace dump (2023-05-21) with a truncated extension. Like Edina's `test.mat`, it holds unseeded run data. | 0 | CONFIRM |
| `1D_model_potential_Simian/{test,benchmark}.mat` | 2 | 1.0 MB | Byte-identical to the Edina copies and never loaded | 0 | DELETE-SAFE, **but only if the Edina copies are kept**. If the Edina folder is retired, keep these instead. |
| `2D_model_potential_TW/.vscode/launch.json` | 1 | <1 KB | Stock editor template | 0 | DELETE-SAFE |
| `1D_model_potential_Edina/` (whole folder) | 14 | 1.0 MB | Superseded snapshot: Simian adopted 12 of its files in acbf717, and `DHAM_unbias.m` is identical to `2D_model_potential`'s. Unique content is one line (`Eigenvector_1D_model.m:17`). | 0 | CONFIRM |
| `1D_model_potential_Edina/{test,benchmark}.mat` | 2 | 1.0 MB | Workspace dumps that are never loaded | 0 | CONFIRM |
| `1D_model_potential_Simian/test2.mat` | 1 | 4.1 MB | Unrelated PCCA/HS clustering workspace | 0 | CONFIRM |
| `1D_model_potential_Simian/Figures_Simian/*` (8 `.fig` + 1 screenshot `.png`) | 9 | 479 KB | Saved by hand from the GUI | 0 | CONFIRM |
| `2D_model_potential/figures/*.fig` | 8 | 457 KB | Saved by hand from `Eigenvectors_2D_model_several_exp.m` plots | 0 | CONFIRM |
| `2D_model_potential/opt_params_*.mat` | 3 | 2 KB | Optimisation results; outputs only | 0 (comment `:77`) | CONFIRM |
| `2D_model_potential_TW/figs/*.png` (top level, timestamped run diagnostics) | 422 | 7.83 MB | Outputs of `main.py`/`dham.py`. Cannot be regenerated exactly (no seed, PNG input deleted). | 0 | CONFIRM |
| `2D_model_potential_TW/figs/prod_figs/movie_20230925-131244/movie.gif` | 1 | 14.5 MB | Deterministic output of `gif_gen.py` from tracked frames | 0 | CONFIRM |
| `2D_model_potential_TW/figs/prod_figs/movie_20230925-131244/*.png` | 10 | 5.0 MB | Presentation frames | `gif_gen.py:4` | CONFIRM |
| `2D_model_potential_TW/figs/prod_figs/traj*.png` | 7 | 3.07 MB | Output of `figure_gen_2D.py:147-153` | 0 | CONFIRM |
| `2D_model_potential_TW/figs/1D_gif/*` | 32 | 4.75 MB | Output of `figure_gen_1D.py:176`; the gif block is disabled | `figure_gen_1D.py:183` | CONFIRM |
| `2D_model_potential_TW/data/20230925-131244_*` | 10 | 45 KB | Inputs of the movie frames (version e1e9276) | 0 (current code) | CONFIRM |
| `2D_model_potential_TW/data/` other timestamped runs and old-naming files, plus 2 `unbiased_mfpt` txt | 119 | 1.13 MB | Run outputs that no code reads | 0 | CONFIRM |
| `2D_model_potential_TW/data/20230927-135127_*` | 13 | 228 KB | Read by `figure_gen_2D.py:38-67` | 13 | KEEP |
| `2D_model_potential_TW/bak/*.png` + `fes_digitize_{gauss_params,fourier_params,wrapnorm_params}.txt` | 9 | 569 KB | Fit outputs of `2D_langevin_sim/fit_dialanine_*.py` | 0 (by these names) | CONFIRM |
| `2D_model_potential_TW/bak/fes_digitize_gauss_params_15{,_2}.txt` | 2 | 4 KB | Only tracked copies of inputs to `2D_langevin_sim/test_util.py:10` (as `params/gaussian_fes_param.txt`) and `fit_dialanine_gauss.py:65` | 2 | KEEP |
| `2D_model_potential_TW/create_2d_fes.py`, `fig_digitizer.py` | 2 | 6 KB | `create_K_png` duplicates `util_2d.py:53-99`; neither file is imported anywhere | 0 | CONFIRM |
| `2D_model_potential_TW/copy_of_first_simulation.ipynb` | 1 | 95 KB | OpenMM Colab prototype with an unfinished cell | 0 | CONFIRM |
| `1D_model_potential_Simian/main_predictive_1D_v0.m`, `gaussmf.m` ×2, `jjhunter.m` | 4 | 4 KB | Superseded v0 / invalid and uncalled / referenced only in a comment (`min_M_mfpt.m:33`) | 0 | CONFIRM |
| `CE_Opt/` (whole folder; 3 byte-copies of `1D_langevin_sim`, vendored `cross_entropy_method/`, 982 KB notebook) | 11 | 1.1 MB | Unfinished experiment that nothing imports | 0 | CONFIRM |
| `Simian/pos_bias.mat`, `{Simian,2D_model_potential_TW}/pos_bias_m.mat`, `2D_model_potential/data_eigenvector_*_.mat` | 8 | 5.0 MB | Loaded by code (see the `.mat` table). The `paper_publication` copy of `pos_bias_m.mat` is referenced only in a comment; that folder belongs to another agent. | yes | KEEP |
| `2D_model_potential_TW/figure_gen_2D.py` (identical to `paper_publication`'s) | 1 | 10 KB | The only copy in this checkout that has its inputs (`util_2d.py` and the data) alongside | — | KEEP |
| All other `.m`/`.py` source, including the byte-identical MATLAB helpers | — | — | Each folder runs on its own | — | KEEP |

## Open questions for the authors
1. Should `fes_digitize.png` be restored from `45d011f^`? `util_2d.py:62`, `figure_gen_2D.py:20`, `2D_presentation_movie_plot.py:260` and `2D_langevin_sim/fit_dialanine_*.py` all need it, and the `paper_publication` copy was deleted in the same commit.
2. Is `1D_model_potential_Edina` still needed? Apart from `state_start=45`, everything in it exists byte-for-byte in Simian or `2D_model_potential`.
3. Where did `benchmark.mat` (`KbT`, `kemeny0`, `reltime0`) and `test2.mat` (547-state PCCA/HS) come from? Are they reference results?
4. Which `2D_model_potential_TW` figures or movies went into talks or the paper? This decides the fate of 35 MB under `figs/`, including the 14.5 MB `movie.gif`.
5. Was `try_and_optim_M_K` meant to use the true `K` (`util_2d.py:437`)? Do any reported 2D results depend on it, or on the amplitude mismatch in `dham.py:17`?
6. Were results from `1D_model_potential_TW` ever reported? The `mfpt_calc`/stationary-distribution issues at `util.py:84,122` would affect them.
7. Is CE_Opt abandoned? If not, what is the licence of the vendored `cross_entropy_method` (Ido Greenberg, 2022), and could it be a pip dependency instead?
8. Is the `data*`/`figs*` ignore rule meant to untrack those folders? If so, the 13 `20230927-135127_*` files need the narrower rule `data/*` plus a `!` exception.
9. Should the 1D umbrella-sampling pipeline (signature mismatches at `Eigenvectors_1D_model_several_exp.m:14` and `several_umbrella_samp_exper.m:40`) be fixed or retired?

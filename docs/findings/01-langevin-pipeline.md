# 01 – Langevin toy-model pipeline (`1D_langevin_sim/`, `1D_langevin_sim_backforth/`, `2D_langevin_sim/`)

Labels: **[Verified]** = backed by the cited `file:line`, a diff, or a command result. **[Inferred]** = a reasoned conclusion that was not executed. No repo script was run.

## Summary
1. Each directory is a self-contained OpenMM toy model: one pseudo-atom on an analytic multi-well FES (1D: x in [0, 2π] nm; 2D: x, y). Three protocols compete on time-to-target: unbiased MD, OpenMM well-tempered metaD, and "MFPT-opt" (5000-step propagation → DHAM Markov matrix → random + L-BFGS search over a Gaussian bias that minimises MSM MFPT → apply the bias → repeat). **[Verified]**
2. Every protocol appends its time-to-target to `total_steps_{mfpt,metaD,unbiased}.csv`. `analysis.py` draws box plots from these files. The 1D mfpt/unbiased and 2D mfpt CSVs are md5-identical to `paper_publication/data/time2reach/`. **[Verified]**
3. `1D_langevin_sim_backforth` is a fork of the 1D directory (Nov 2023 to Jan 2024). It adds an MSM-class optimiser, dynamic DHAM bins and a start↔end cycle loop. As committed, its three entry points have crash bugs. **[Verified]** for code, **[Inferred]** for runtime.
4. `U.npy`, `U_denom.npy` and `UB3d.npy` (3 × 62.7 MB, which is 188 MB of this area's 189.5 MB) are write-only WHAM intermediates from backforth `analysis.py`. No code reads them, so they are **DELETE-SAFE**. **[Verified]**
5. Top bug: all three MFPT-opt loops reshape the Gaussian bias parameters in C order before DHAM, which scrambles (a, b, c). The step counts also overcount by one propagation, and neither unbiased script measures a first-passage time. **[Verified]**

## Directory map
| File | Role | Entry point? |
|---|---|---|
| `*/config.py` | Module-level parameters. Calls `Platform.getPlatformByName('CUDA')` on import (1D:25, bf:16, 2D:25). | no |
| `*/langevin_sim_mfpt_opt.py` | MFPT-optimised adaptive-bias loop | `__main__` (1D:179, bf:176, 2D:168) |
| `*/langevin_sim_unbiased.py` | Plain Langevin run | `__main__` (1D:29, bf:28, 2D:28) |
| `*/langevin_sim_metad.py` | OpenMM `Metadynamics` run | `__main__` (:27 in all three); also launched by `2D/do_python_gpu.sh:22` |
| `*/analysis.py` | Post-processing toggled by flags (1D:23-29, 2D:23-29). Backforth version = WHAM vs DHAM vs true FES. | top-level script, no guard |
| `*/dham.py`, `*/util.py`, `bf/MSM.py`, `bf/wham.py`, `2D/MSM.py` | Libraries (black box) | `MSM.py:445` self-test only |
| `1D/test_4_george.ipynb` | Scratch test of 3 `get_working_MM` variants | no |
| `2D/2D_opt_K.py` | One-shot bias optimisation on an analytic 20×20 rate matrix K. The only user of `2D/MSM.py` (:80). | top-level script |
| `2D/fit_dialanine_{gauss,fourcos}.py` | Fit Gaussians or a Fourier series to a dialanine FES image (`./fes_digitize.png`) | top-level script |
| `2D/plain_multi_well_2D_fes.py` | Standalone plot of the 2D multi-well FES | `__main__` :39 |
| `2D/test.py`, `2D/do_python.sh` | SGE smoke test (`do_python.sh:20` runs `test.py`) | .sh |
| `2D/test_util.py` | Broken scratch test of `util.gaussian_2D` | top-level script |
| `2D/equation_note.txt` | Single-line title only, no content | – |
| `*/total_steps_*.csv`, `*/bak/*.csv` | Result logs (time-to-target in MD steps) | – |
| `1D,2D/langevin_approach/MetaD_auxfiles/<tag>/bias_<id>_<n>.npy` | OpenMM metaD bias grids (outputs) | – |
| `bf/U.npy`, `bf/U_denom.npy`, `bf/UB3d.npy` | WHAM caches, shape (56, 5000, 56) float32 (npy header) | – |

## Workflow & how to run
**MFPT-opt loop (1D)**: build the system (1 particle, 12 amu) with `apply_fes(mode='multiwell')` and stiff y/z restraints (`1D/langevin_sim_mfpt_opt.py:200-213`). Use a Langevin integrator at 300 K, 1/ps, 2 fs (:223-225).
1. Propagation 0: a tiny random bias near `start_state` (:245, :167-176). The script saves `params/<tag>_gaussian_fes_param_0.txt` (:249), runs 5000 steps writing every frame to DCD (:43-49), reloads with mdtraj (:56-60) and stores x in `pos_traj` (:75).
2. `DHAM_it` runs over all propagations so far, using all previous biases, and returns the FES and Markov matrix (:95-101). `get_working_MM` drops unvisited states (:115-121).
3. Local target = the visited bin closest to `end_state` (:283). Local start = the last visited bin (:290-293).
4. `util.try_and_optim_M`: 1000 random 20-Gaussian biases are scored by MSM MFPT, then the best is refined with L-BFGS-B (`1D/util.py:215-327`).
5. `update_bias` sets the context parameters (:357) and the loop propagates again. It stops when the particle is within 0.1 nm of the target (:107-111) or after `sim_steps/propagation_step` propagations (:240). It then appends to `total_steps_mfpt.csv` (:388).
- **2D**: the same loop on a digitised 20×20 grid ravelled to 1D (`2D/...mfpt_opt.py:62-72`). The target uses `find_closest_index` (:124-146) and the local start is the most-visited bin (:273). Params go through a txt round-trip each propagation (:98).
- **backforth**: wraps the loop in `while cycle_count < max_cycle` (:238) and swaps target start↔end on each reach (:378-381). It uses `MSM`-based `try_and_optim_M(qspace=…)` (`bf/util.py:18`) and dynamic DHAM bins (`config.use_dynamic_bins`, `bf/dham.py:108`).

**How to run** **[Verified]** from code, **[Inferred]** that nothing else is needed:
- Run `cd <dir> && python langevin_sim_{mfpt_opt,unbiased,metad}.py`. All paths and imports (`import config`, `from util import *`) are relative to the cwd.
- A **CUDA GPU is mandatory**: `import config` fails without it (config.py:25), which breaks even `analysis.py`. Some scripts also hard-code CUDA again (1D unbiased:84, metad:124).
- Pre-create the output directories, because these scripts do not create them: `trajectory/explore`, `params`, `figs/explore`, `visited_states` (mfpt_opt), and `trajectory/metaD`, `visited_states`, `figs/metaD` (metad:170,177,228). 2D unbiased also needs `trajectory/unbias`, `figs/unbias` (:98,131), and backforth unbiased needs `trajectories/unbias` (plural, :95). Only 1D unbiased (:99-101,132-133,151-152) and the metaD aux directory (metad:89-92) are created automatically.
- Cluster: `2D/do_python_gpu.sh` is an SGE script for UCL Myriad (comment at :1), 2 GPUs, 48 h, conda env `biophys_env` (:17-20).
- `.gitignore:16-26` ignores everything in these directories except `*.py`. The non-`.py` files already tracked were committed before that rule (rule added in c93d7f8 on 2024-11-06; the U*.npy files were added in 95222c3 on 2024-01-22), so appending to a tracked CSV still dirties git. **[Verified]**

**Key config values** **[Verified]** (1D / backforth / 2D):
| param | 1D | backforth | 2D |
|---|---|---|---|
| num_sim / sim_steps / sim_steps_unbiased | 20 / 5e7 / 1e7 | 1 / 5e6 / – | 5 / 1e8 / 5e7 |
| propagation_step / dcdfreq / dcdfreq_mfpt | 5000 / 500 / 1 | 5000 / 100 / 1 | 5000 / 500 / 1 |
| stepsize / stepsize_unbias (ps) | 0.002 / 0.2 | 0.002 / 0.002 | 0.002 / 0.002 |
| bins / num_gaussian / amp | num_bins 100 / 20 / 6 | DHAM_num_bins 50, qspace 100 / 10 / 6 | num_bins 20 / 20 / 6 |
| start → end (nm) | 2.0 → 5.2 (unbiased 3.9) | 2.0 ↔ 5.2 | (5.0, 4.0) → (1.0, 1.5) |
| extra | `NUMBER_OF_PROCESSES` (dead) | `max_cycle=20`, `num_propagation=400`, `use_dynamic_bins`, `T`, `load_global_gaussian_params_from_txt` | – |

## backforth vs plain 1D diff
`diff -rq 1D_langevin_sim 1D_langevin_sim_backforth` **[Verified]**: `bak/` is identical. Only in backforth: `MSM.py`, `wham.py`, `U.npy`, `U_denom.npy`, `UB3d.npy`. Only in plain: `langevin_approach/`, `test_4_george.ipynb`, `total_steps_{metaD,unbiased}.csv`. Changed lines (−plain/+bf): config −21/+30, dham −47/+52, util −244/+50, mfpt_opt −223/+256, metad −17/+11, unbiased −29/+21, analysis −282/+71.
- **config**: num_sim 20→1, sim_steps 5e7→5e6, dcdfreq 500→100, num_gaussian 20→10. Replaces `num_bins` with `qspace_*` + `DHAM_num_bins=50` + `use_dynamic_bins=True`. Adds `max_cycle`, `num_propagation`, `T`. Drops `sim_steps_unbiased` and `end_state_unbiased`.
- **mfpt_opt**: outer cycle loop with target swap (bf:238-424). Two particles X1 (12 amu) and X2 (1 amu), FES applied to both (bf:184-213). Asserts on the param shape (bf:81-84). `np.digitize(...)-1` fixes (bf:286,290). Full-trajectory DHAM FES plot after each cycle (bf:405-424). `total_steps = i_prop*propagation_step` (bf:428) instead of adding `reach`.
- **util/dham**: the optimiser and DHAM call the `MSM` class (`bf/util.py:5,46-70`; `bf/dham.py:13,133-146`). DHAM returns `(qspace, F, M)` instead of `(F, M)` (`bf/dham.py:146` vs `1D/dham.py:142`).
- **metad**: CV `x` → `(x-x0)^2`, biasFactor 5→2, height `amp/2`→0.1 kJ/mol. Writes the DCD with `DCDFile` instead of appending reporters.
- **unbiased**: has a `while reach is None` loop (plain has a fixed-length `for`), but with an ndarray `.append` bug (see Bugs) and `trajectories/` instead of `trajectory/`.
- **analysis**: rewritten into a WHAM vs DHAM comparison on run `20231123-170259` (bf:10-57) plus an inline `draw_fes` (bf:61-81).
- Plain 1D was edited later (util, analysis: c17d449 on 2024-04-24; mfpt_opt: 2023-12-14). Backforth's last commit is 95222c3 on 2024-01-22, so neither directory supersedes the other. **[Verified]** git log.

## Data & outputs
| Script | Writes (relative to the script directory) | **[Verified]** lines |
|---|---|---|
| mfpt_opt (1D/2D) | `trajectory/explore/<tag>_langevin_sim_explore_<i>.{dcd,pdb}`, `params/<tag>_gaussian_fes_param_<i>.txt`, `figs/explore/*.png`, `total_steps_mfpt.csv` (append), `visited_states/<tag>_pos_traj.txt`, `./fes_visualization.png` (apply_fes) | 1D:43,52,249,349,354,388,393 · 2D:42,216,232,285,315,329,358,371,376 |
| mfpt_opt (bf) | same, with `_cyc_<c>_prop_<i>` names and `params/<tag>_gaussian_param_prop_<i>.txt` | bf:52,61,248,332,336,423,431,434 |
| unbiased | `trajectory/unbias/<tag>_unbias_traj.dcd`, `figs/unbias/*.png`, `total_steps_unbiased.csv`, `visited_states/<tag>_langevin_2D_unbias_traj.txt`. 2D also writes `./fes_<tag>.mat` | 1D:102,134,145,153 · 2D:60,98,131,142,148 |
| metad | OpenMM writes `langevin_approach/MetaD_auxfiles/<tag>/bias_<id>_<n>.npy`. Also `trajectory/metaD/<tag>_metaD_traj.dcd`, `visited_states/<tag>_metaD_*.npy`, `total_steps_metaD.csv`, `figs/metaD/*.png`. The 2D version writes only the DCD and potential-energy npy; its CSV code is commented out (2D:212-222). | 1D:89,170,177,181-182,203,228 · 2D:191,198 |
| bf analysis | Reads `visited_states/20231123-170259_pos_traj.txt` and `params/20231123-170259*`. Writes `U.npy`, `UB3d.npy` (`wham.py:83-84`), `U_denom.npy` (`wham.py:209`), `fes.png` | bf analysis:10-16,88 |
| 2D_opt_K | `figs/test_2D_fes*.png`, `figs/free_energy_K_2D.png` | :61,100,114,177 |

- **Three big npy files**: `calculate_UB3d` fills U[i,j,k] = bias of window k at sample (i,j) and UB = exp(−U/kT) (`bf/wham.py:63-86`). `calc_denom` re-saves the same `self.U` as `U_denom.npy` (:193,209), which explains why the md5 matches `U.npy`. The only loads are commented out (:92-93). `converge` tests `self.UB`, which is never assigned, so the arrays are always recomputed (:90-91). The shape (56, 5000, 56) matches `analysis.py:20,24` (num_sim=56, 5000 frames). They are regenerable in principle, but the inputs (`visited_states/`, `params/` for run 20231123-170259) are not in this checkout. **[Verified]**
- **MetaD aux files are outputs**. OpenMM writes `bias_<id>_<saveIndex>.npy` into `biasDir` and reads only *other* ids' files from that directory (multi-walker sync; `openmm/app/metadynamics.py:226-262` in OpenMM 8.6.1 on this machine, not necessarily the authors' version). Each run uses a fresh `<time_tag>` directory (1D metad:89), so no run ever reads an old one. No repo code references `MetaD_auxfiles` except the writers. Twenty-five 1D files have shape (158,) = ceil(5·2π/0.2), which matches the current `BiasVariable(…,0,2π,0.2)` (1D metad:109). The others ((200,), (63,), (625,), …) come from older CV settings, and the 2D (200, 200) files match 2D metad:124-125. Exactly the seven `bias_*_1.npy` files are all zeros. OpenMM writes index 1 from its constructor (`metadynamics.py:124`), so those runs never reached a periodic save. **[Verified]** (npy headers plus a non-zero-byte count).
- **`bak/` folders are outputs**: older `total_steps_*.csv`, and nothing references `bak/` (`git grep "bak/"` finds nothing in code). **[Verified]**
- **Readers of the CSVs**: only `analysis.py` with `plot=True` (default False; 1D:182-184, 2D:168-170). The paper uses its own copies (`paper_publication/analysis.py:31`). **[Verified]**
- **CSV quality**: 2D `total_steps_metaD.csv` row 2 is `46415004999500` (two values fused, CRLF). Rows like `4999500`/`49999500` are the run-length cap (= (N/500 − 1)·500), i.e. censored runs, not first-passage times. The 1D `total_steps_unbiased.csv` holds one hand-written value, `1.063e+08`, and 2D unbiased row 1 is `18.526e+06`; the code writes ints. The 1D metaD CSV has 6 extra rows beyond the paper copy (diff rows 25-30). **[Verified]** (od, diff); hand-entered is **[Inferred]**.

## Bugs / smells
1. **Gaussian-param layout scrambled before DHAM.** Biases are concatenated block-wise (a…, b…, c…) (`1D/...mfpt_opt.py:176`, `1D/util.py:262`, `2D/util.py:45,205`), and OpenMM unpacks them block-wise (`1D/util.py:544-546`). The DHAM input, however, is a C-order `reshape(1,n,3)` (1D mfpt_opt:247,303; bf:247,335) or `reshape(-1,5)` (2D:98), and DHAM unpacks `params[k,g,:]` as (a, b, c) (`1D/dham.py:87`, `bf/dham.py:89`, `2D/dham.py:98`). So row g = (a_3g, a_3g+1, a_3g+2)…, and DHAM "unbiases" with the wrong potentials. Backforth `analysis.py:28` uses `order='F'`, which is correct. **[Verified]**
2. **`reach` is the last frame inside the target, not the first** (no `break`): 1D:107-111, bf:105-109, 2D:108-112. **[Verified]** 3 of the 15 rows in `2D/total_steps_mfpt.csv` end in `4999`, which fits. **[Inferred]**
3. **MFPT step overcount**: `total_steps = i_prop*5000 + reach` is computed after `i_prop += 1` (1D:382,385; 2D:365,368), which adds one whole propagation. If the propagation cap is hit, `reach` is None and the line raises a TypeError, so no row is written. **[Verified]**
4. **Unbiased runs never record a first passage.** 1D detection is commented out (1D unbiased:114-117) so the script always writes 9,999,500 (:142). 2D sets `reach` but never breaks out of the `for` loop (2D:100-112), so it always writes 49,999,500: 17 of 18 rows in 2D `total_steps_unbiased.csv`. **[Verified]**
5. **A new `DCDReporter` is appended on every loop iteration** (1D metad:170, 2D metad:191): 1e5 to 2e5 reporters on the same file. **[Verified]** It probably causes severe slowdown or file-descriptor exhaustion. **[Inferred]**
6. **Backforth mfpt_opt cannot start as committed**: 2 particles (bf:184-197) but `start_state` has 1 position (bf config:17, used at :252). OpenMM raises "Called setPositions() on a Context with the wrong number of positions" (string found in local libOpenMM 8.6.1). **[Inferred]**
7. bf mfpt_opt `propagate()` names its files with the *global* `i_prop`, not its `prop_index` argument (bf:52,61,63,64). **[Verified]**
8. bf mfpt_opt :393 reads `pos_traj[i_prop]` *after* the increment: an all-zero row, so the local start is bin 0 after every target swap. It raises an IndexError when `i_prop == num_propagation`. **[Verified]**
9. bf `total_steps` sums all cycles and ignores `reach` (bf:428). All CSV values are multiples of 5000. **[Verified]**
10. bf unbiased: `pos_traj = np.zeros(...)` (:91), then `.append` (:101), which raises AttributeError. It also calls `imshow` on a 1D FES (:117). **[Verified]**
11. bf metad uses `config.num_bins` (bf metad:206), which the backforth config does not define, so it raises AttributeError after the whole simulation. **[Verified]** (config attribute scan)
12. The "recent trajectory" slice `pos_traj[i_prop:]` selects future, all-zero rows (1D:341, bf:324, 2D:352), so the plots are wrong. **[Verified]**
13. `np.digitize` off-by-one or out-of-range indexing: 1D mfpt :291,343-347 (no −1; fixed in bf:290,326). `1D/analysis.py:298` can index `fes[100]`. In 2D, digitised values in 1..20 go into `ravel_multi_index` (2D:66-72), which would raise a ValueError at x ≥ 2π. **[Inferred]**
14. 2D ravel order mismatch: the trajectory is ravelled with `order='F'` (2D:72, "temporary fix"), but the target and closest-state code uses `'C'` (:264, :137-145). **[Verified]**
15. Operator precedence in the MFPT biasing: `np.exp(-u_ij / 2*0.5981)` multiplies by kT/2 instead of dividing by 2kT (`1D/util.py:237,302`). `minimize(args=…)` binds `working_indices` to `kT` (util:286 vs 320-323); this is harmless because `kT` is unused. **[Verified]** (util is black box; flagged because it sits on the workflow path)
16. The analysis targets do not match the configs: 1D `analysis.py:283` and 2D `analysis.py:259` test for reaching [1.0, 1.0], but the 1D end state is 5.2 and the 2D end state is (1.0, 1.5) (2D config:32). 1D analysis also uses a different grid `linspace(0,2π+1,100)` (:291). **[Verified]**
17. Dead NaCl copy-paste in analysis: `trajectories/metaD` plus `./toppar/step3_input25A.pdb` (1D:159-164; 2D:145-150), `Unbiased_Profile.npy` (1D:107; 2D:93). **[Verified]**
18. Hard-coded run timestamps and counts in analysis: 1D:33,56-57,83,131,137,248-249; 2D:32-33,59,80,85,227-228; bf:10,15,20 (`num_sim=56`). **[Verified]**
19. The integrator step is hard-coded to 0.002 ps and ignores `config.stepsize` (1D mfpt:225, bf:229, 2D:204, 2D metad:145). 1D unbiased integrates at 0.002 ps (:78, though the comment says 0.2) but tags the DCD with `stepsize_unbias`=0.2 ps (:103), so DCD times are 100× too large. **[Verified]**
20. Unit comments are wrong: "convert to kcal/mol" next to `* 4.184`, which converts kcal→kJ (1D mfpt:307; 2D:311,317). **[Verified]**
21. Dead code: `find_closest_index` (2D-only logic) is unused in 1D/bf (1D:123-145, bf:121-143). The multiprocessing block sits inside a string (1D:395-401, 2D:378-384), and it is the only use of `NUMBER_OF_PROCESSES`. `cur_pos = pos_traj[-1]` (1D unbiased:112). `npoints` is unused (metad:32). `print("all done")` runs at module level outside `__main__` (1D mfpt:402, metad:232, 2D mfpt:385, bf mfpt:435). **[Verified]**
22. Unclosed DCD handles in the unbiased scripts (1D:102, 2D:98, bf:95). `plot_inteval = len//10000` can be 0, which gives a slice-step error (1D metad:220). **[Verified]**
23. The metaD CV grids do not cover the whole path. 2D uses CV `(x-x0)^2` on a grid [2, 10] (2D:107,124-125), so there is no bias within √2 nm of the target. Backforth uses `(x-5.2)^2` on [0, 2π] (bf metad:101,108), so the start state (CV = 10.24) lies outside the grid. **[Inferred]** design smell.
23b. In backforth, `cycle_count` and `current_target_state` are initialised before the `for i_sim` loop (bf:177-178 vs :188) and never reset, so a second sim would skip the cycle loop. This is latent because `num_sim = 1`. **[Verified]**
24. `2D_opt_K.py`: no `__main__` guard. `from config import *` requires CUDA (:7). Hard-coded `(20,20)` instead of `N` (:102,166). The final plot calls `colorbar()` with no image drawn (:166-177), which recent matplotlib rejects. **[Inferred]**
25. `test_util.py:13` calls `gaussian_2D(i,j,*params)`, but the signature is `(params,x,y)` (`2D/util.py:26`). It also reads the absent file `./params/gaussian_fes_param.txt` (:10). **[Verified]**
26. `fit_dialanine_fourcos.py:179`: `from scipy.linalg import leastsq` does not exist (it lives in `scipy.optimize`). It is only reached when `target_leastsq=True` (:90). Both fit scripts read `./fes_digitize.png` (gauss:38, fourcos:86), which was never tracked here; a copy exists at `2D_Dialanine/figs/fes_digitize.png`. **[Verified]**
27. Shell scripts: in `do_python.sh` the shebang is on line 4 (:4), there is a typo `#% -N` (:7) and `h_rt` is 10 s (:9). Both scripts start python with `nohup … &` inside a batch job (do_python.sh:20, do_python_gpu.sh:22), so the job exits at once. **[Inferred]** `do_python_gpu.sh:12` hard-codes a cluster home path as `-wd` but runs a relative `python langevin_sim_metad.py`. A personal email is hard-coded at do_python.sh:13 and do_python_gpu.sh:15. **[Verified]**
28. No Windows paths and no machine-local absolute paths in `.py` files (grep for `/home/`, `/Users/`, `X:\\`): only the cluster path in item 27. The only Windows trace is CRLF line endings in the CSVs (`od -c`). **[Verified]**

## Third-party deps (this area)
numpy, scipy (linalg, optimize, constants, io), matplotlib (incl. `mpl_toolkits.mplot3d` in MSM.py:11), **openmm** (app, unit, `app.metadynamics`; CUDA platform), **mdtraj**, tqdm, **natsort** (bf analysis:7), **Pillow** (`2D/util.py:12`, fit_*), **numba** (lazy, MSM.py:241), **plotly** (lazy, `1D/util.py:479`, `bf/util.py:285`; `fig.write_image` at bf util:312 also needs kaleido **[Inferred]**). jax is commented out (`bf/wham.py:3`). Stdlib: csv, os, time, glob, math, sys. **[Verified]** (import grep)

## Redundancy table
| Path | Size (B) | Evidence | Refs found (basename grep, whole repo) | Class |
|---|---|---|---|---|
| `1D_langevin_sim_backforth/U_denom.npy` | 62,720,128 | md5 twin of `U.npy`; the same `self.U` re-saved (wham.py:193,209) | only writer `bf/wham.py:209` | **DELETE-SAFE** |
| `1D_langevin_sim_backforth/U.npy` | 62,720,128 | write-only cache (wham.py:83); loads commented out (:92-93); always recomputed (:90) | only writer `bf/wham.py` | **DELETE-SAFE** (inputs to regenerate are absent) |
| `1D_langevin_sim_backforth/UB3d.npy` | 62,720,128 | = exp(−U/kT) (wham.py:82,84); same as above | only writer `bf/wham.py` | **DELETE-SAFE** |
| `1D_langevin_sim_backforth/bak/total_steps_mfpt.csv` | 120 | md5 twin of `1D_langevin_sim/bak/total_steps_mfpt.csv` | every basename hit is a cwd-relative writer/reader; no code contains `bak/` | **DELETE-SAFE** |
| 1D `MetaD_auxfiles/{20231113-154229,20231113-155840,20231113-155912,20231115-171851,20231115-172003}/bias_*_1.npy`; 2D `MetaD_auxfiles/{20231031-124636,20231031-124843}/bias_*_1.npy` | 6,040 + 640,256 | All-zero arrays (0 non-zero bytes after the header): the constructor-time save, so the run never reached a periodic save. 3 of the 1D files are md5-identical. | only writers (metad.py:89) | **DELETE-SAFE** (with their directories) |
| Remaining 38 1D + 1 2D (`20231031-124913`) `MetaD_auxfiles/*/bias_*.npy` | 54,680 + 320,128 | final metaD bias grids; write-only | only writers | **CONFIRM** |
| `1D_langevin_sim/bak/total_steps_mfpt.csv` | 120 | older MFPT results, unread | basename hits are cwd-relative only; no `bak/` in code | **CONFIRM** |
| `2D_langevin_sim/bak/total_steps_{metaD,mfpt,unbiased}.csv` | 294/143/152 | older results; unbiased values come from a version that stopped at the target | same as above | **CONFIRM** |
| `1D_langevin_sim_backforth/total_steps_mfpt.csv` | 39 | written by bf mfpt:431; not read by bf analysis | writers only | **CONFIRM** |
| `1D_langevin_sim/test_4_george.ipynb` | 5,161 | scratch notebook; the production `get_working_MM` is at mfpt_opt:115-121 | none | **CONFIRM** |
| `2D_langevin_sim/test.py` + `do_python.sh` | 81 + 433 | SGE smoke test (prints one line; 10 s walltime) | `test.py` ← `do_python.sh:20` only | **CONFIRM** |
| `2D_langevin_sim/test_util.py` | 500 | broken (bug 25) | none | **CONFIRM** |
| `2D_langevin_sim/equation_note.txt` | 65 | single comment line, no equations | none | **DELETE-SAFE** |
| `2D_langevin_sim/fit_dialanine_{gauss,fourcos}.py` | 3,549/8,559 | dialanine image fitting, unrelated to `fes_mode='multiwell'`; input image not here; outputs in `2D_model_potential_TW/bak/` | none | **CONFIRM** |
| `2D_langevin_sim/plain_multi_well_2D_fes.py` | 1,402 | standalone plot; same well parameters as `2D_opt_K.py:13-25` | none | **CONFIRM** |
| `1D,2D/total_steps_{mfpt,metaD,unbiased}.csv` | 9–436 | read by `analysis.py` (1D:182-184, 2D:168-170); some are md5 twins of `paper_publication/data/time2reach/` | analysis.py, writers | KEEP |
| `bf/MSM.py`, `bf/dham.py`, `1D/dham.py`, `1D/config.py`, `1D/langevin_sim_mfpt_opt.py`, `2D/MSM.py` | – | md5 twins in root / `CE_Opt` / `1D_PMg*` / `paper_publication`, but imported by relative path in their own directory | local imports | KEEP (for the drift agent) |
| `2D/2D_opt_K.py`, the other scripts | – | unique source | – | KEEP |

Note: deleting files only removes them from HEAD. The 188 MB stays in git history (added in 95222c3) unless history is rewritten. **[Verified]** git log.

## Open questions for the authors
1. Were the published MFPT-opt numbers (`paper_publication/data/time2reach/{1D,2D}/total_steps_mfpt.csv`, md5-identical to these) produced with the C-order parameter scramble (bug 1) and the +1-propagation overcount (bug 3)?
2. How was the 1D unbiased value `1.063e+08` ("Unbiased (Estimated)", 1D analysis:229) obtained, and should the censored `4999500`/`49999500` rows count as first-passage times?
3. In backforth, what is the second 1-amu particle for (bf:184-197)? Did the 2024-01-22 version ever run?
4. Do you still have `visited_states/20231123-170259_pos_traj.txt` and `params/20231123-170259_*` anywhere? They are needed to regenerate the U*.npy files and to rerun backforth `analysis.py`.
5. Are the non-zero MetaD bias grids needed as provenance for the paper's metaD numbers, given that `recalc_metad_reachtime.py` recomputes them from DCDs?
6. Which of the 6 extra rows in `1D/total_steps_metaD.csv` (not in the paper copy) are valid runs, and with which settings?
7. Is `fit_dialanine_*.py` superseded by `2D_Dialanine/` or `2D_model_potential_TW/`?

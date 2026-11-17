# 02 · OpenMM molecular-system simulations

Scope: `1D_NaCl_TW/` (+`backup/`, `bak/`), `1D_NaCl_TW_backforth/`, `1D_PMg/`, `1D_PMg_backforth/`, `2D_Dialanine/`, `DIALA/`.
**[Verified]** means read in the file, diffed, or shown by a command. **[Inferred]** means reasoning from code or general OpenMM knowledge; nothing was executed.
Nothing in the repo was run. `dham.py`, `util.py`, `MSM.py` and `config.py` are treated as black boxes; their version drift is covered elsewhere.

## Summary
- There are three molecular systems. Each runs the same loop: short biased MD, then DHAM builds an MSM from all data so far, then `try_and_optim_M` fits a 10-Gaussian bias that minimises the MFPT to a target, and the loop repeats until the CV reaches the target. The run's cost is reported as the total number of MD steps. **[Verified]** `mfpt_NaCl.py:1-9,234-351`.
  - NaCl: 1-D CV is the Na–Cl distance.
  - PMg: 1-D CV is the Mg²⁺–phosphate distance inside a protein.
  - Alanine dipeptide: 2-D CV is meant to be (φ, ψ), but the atom indices are wrong (see Bugs).
- For NaCl, unbiased MD and OpenMM metadynamics serve as the baselines. `analysis.py` turns the three step-count CSVs into the box plot. **[Verified]**
- Each `*_backforth` directory adds an outer loop that alternates the target between the start and end states (start→end→start…, up to `max_cycle=20`). In both directories the final step-count line dereferences `reach=None`, so they almost certainly crash at the end. **[Inferred]**
- Several high-impact bugs:
  - The bias parameters are passed to DHAM in the wrong layout.
  - The "steps to reach" counter records the last hit, not the first (B3; the NaCl CSV shows the fingerprint). PMg also multiplies the counter by 100 (B5; every PMg value is a multiple of 100).
  - PMg applies its bias 10 times.
  - The dialanine dihedrals use the wrong atoms.
  - The dialanine barostat runs at 5 K.
- `1D_NaCl_TW/backup/` is an older, workshop-derived pipeline: umbrella sampling, one-shot bias optimisation, then MD. Its scripts cannot run in place. Its notebooks, umbrella-sampling data and `mm_metrics.csv` results are unique.
- About 27 MB is DELETE-SAFE: 22 MB of `1D_PMg_backforth` copies of unreferenced inputs plus 4.5 MB of duplicate saved-system XMLs. A further ~51 MB of exact duplicates (the PMg_backforth toppar, psf and restart files) is **KEEP**, because each script loads them by relative path.
- **Corrections to the brief:**
  - PMg `toppar/` has **55** files, not ~45, and `toppar.str` lists all 55. **[Verified]**
  - `2D_Dialanine/trajectories/*` (7 `.dcd` + 7 `.txt`) is **present and tracked**, despite `.gitignore:11`. **[Verified]** `git ls-files`.
  - `.gitignore:13,22,...` ignores everything in `/1D_NaCl_TW/*`, `/1D_PMg/*` etc. except `*.py`. All inputs there (psf, pdb, toppar, npy, csv) are tracked but ignored, so a re-add would silently drop them. **[Verified]** `git check-ignore -v --no-index`.

## 1. Per-system

### 1.1 NaCl dissociation (`1D_NaCl_TW/`)
- **System [Verified]:** `toppar/step3_input.psf` (1700 atoms) holds 1 SOD, 1 CLA and 566 TIP3 (CHARMM-GUI).
  - The start structure `toppar/step3_input25A.pdb` has Na–Cl = 2.57 Å and `CRYST1 0 0 0`, i.e. no box.
  - `toppar/toppar_water_ions.str` is **hand-modified**: SOD/CLA LJ ε = −5.0 kcal/mol (`:280,:298`), and NBFIX SOD–CLA is −6.0 kcal/mol at 2.5 Å (`:304,:361`). The comment at `mfpt_NaCl.py:214` says the same.
  - `createSystem` passes no `nonbondedMethod` (`mfpt_NaCl.py:217-219`), so OpenMM's default NoCutoff applies and the 1 nm cutoff is ignored. The system is a water droplet in vacuum. **[Inferred]**
- **CV:** atoms 0–1 distance, in Å, recorded every step (`mfpt_NaCl.py:88-93`).
  - The bias is a `CustomBondForce` sum of 10 Gaussians on that bond (`:156-177`).
  - Start ≈2.57 Å. The local target is 7 Å (`:269`); the "reached" test is ≥8 Å (`:113`). Unbiased MD and metaD both use 7 Å (`unbias_NaCl_sim.py:83`, `metaD_NaCl_sim.py:131`).
- **Entry points (run with the cwd = the directory; every path is cwd-relative):**

| Script | Role | Writes (dirs must already exist unless noted) |
|---|---|---|
| `mfpt_NaCl.py` | MSM-optimised adaptive biasing; 20 sims × ≤150 props × 5000 steps | `trajectories/explore/`, `gaussian_params/`, `distance/`, `bias_plots/`, `visited_state/`, appends `total_steps_mfpt.csv` |
| `unbias_NaCl_sim.py` | Plain MD until 7 Å, 20 sims, T=298.15 K | `trajectories/unbiased/`, `visited_state/`, appends `total_steps_unbiased.csv` |
| `metaD_NaCl_sim.py` | `openmm.app.Metadynamics` on r (0.2–1.0 nm, σ 0.02, h 1.5 kcal/mol, bias factor 5), fixed 5e7 steps | `trajectories/metaD/`, `visited_state/`; auto-creates `MetaD_auxfiles/<tag>/` and `fes_metaD/` (`:74-76,:114-115`) |
| `analysis.py` | Flag-driven (`:13-16`); default `plot=True` gives seaborn box plot of the 3 CSVs, ×0.002 ps/step | `box_plot_log.png` |
| `unbias_traj_fes_gen.py` | Boltzmann-inversion FES from one hard-coded unbiased run (`:15`) | **broken** (see Bugs) |
| `run_NaCl_explore_sim.sh` | Meant to launch 20 background jobs | **broken**: calls non-existent `explore_bias_NaCl_gen.py` (`:9`) |

- **How to run [Inferred]:** `cd 1D_NaCl_TW && mkdir -p trajectories/{explore,unbiased,metaD} gaussian_params distance bias_plots visited_state && python mfpt_NaCl.py`. A CUDA-capable OpenMM is mandatory (see Bugs B9).
  - To get a working launcher, change `run_NaCl_explore_sim.sh:9` to `mfpt_NaCl.py`, add `#!/bin/bash`, and replace `conda activate` with `eval "$(conda shell.bash hook)"; conda activate biophys_env`.
  - The parallel launch is unsafe as written (B8).
- **Live `config.py` params [Verified]** (from `grep config\.` counts): `num_sim=20`, `sim_steps=5e7` (metaD length), `T=300`, `propagation_step=5000`, `stepsize=2 fs`, `num_bins=100`, `dcdfreq=500` (DCD stride and metaD chunk), `dcdfreq_mfpt=1`, `num_gaussian=10`.
  - Defined but never read (Langevin-era leftovers): `NUMBER_OF_PROCESSES`, `sim_steps_unbiased`, `pbc`, `amp`, `stepsize_unbias`, `start_state`, `end_state`, `end_state_unbiased`, `fes_mode`, `platform`.
  - `qspace` is hard-coded as `linspace(2,9,101)` Å (`mfpt_NaCl.py:227`, `dham.py:117`, `util.py:271`).

### 1.2 PMg: Mg²⁺–phosphate unbinding in a protein (`1D_PMg/`)
- **System [Verified]:** `step3_input.psf` (CHARMM-GUI FF-Converter, 10/11/23) has 100,027 atoms: PROA (2621 atoms) and PROB (5178), GDP, one MG, one H2PO4, 30,664 TIP3, and K⁺/Cl⁻.
  - The PSF atoms at 0-based 7799 and 7840 are `MG` and `H2PO4 P` (psf lines for 7800/7841), which matches `explore_sim_PMg.py:134-135,215`.
  - The start is `traj_0.restart.pdb`, with Mg–P ≈ 3.3 Å and `CRYST1 0 0 0`.
  - Run with `CutoffNonPeriodic` (`:278`), i.e. a solvated protein with no PBC. **[Inferred]** smell.
  - The "Balint system" label rests only on the comment `# Balint added this` (`:321,:397`), next to the unused MSM-matrix-power code. Whether "PMg" is Balint's project is **[Inferred]**; see Open questions.
- **CV:** the Mg–P distance in Å, recorded every step.
  - Target ≥8 Å (`:173`, `:329`). Initial bias at 3.7 Å (`:296`).
  - DHAM uses `config.qspace = linspace(1,8,100)`.
  - `minimizing=False` (`:33`), so the run starts from the restart coordinates and velocities are set to T.
- **Entry point:** `explore_sim_PMg.py`.
  - Writes into `trajectory/explore_traj/`, `params/`, `distance/` and `figs/explore/`, plus `./test_dham_<i>.png` because DHAM is called with `plot=True` (`:101`).
  - Appends to `total_steps_mfpt.csv`.
  - Run with `cd 1D_PMg && mkdir -p trajectory/explore_traj params distance figs/explore && python explore_sim_PMg.py`.
- `metaD_NaCl_sim.py` here is a stale NaCl script. It loads `toppar/step3_input.psf` and `toppar/step3_input25A.pdb` (`:40-41`), neither of which exists in `1D_PMg/toppar/`, so it cannot run here. **[Verified]** by `ls`. It also loads `toppar/toppar_water_ions.str` (`:54`). That file does exist, but the PMg copy is the stock CHARMM file without the NaCl NBFIX change. **[Verified]** by `diff`.
- **Live config [Verified]:**
  - Simulation: `num_sim=15`, `propagation_step=50000`, `num_propagation=400`, `T=300`, `stepsize=2 fs`, `fricCoef=1/ps`.
  - Sampling and binning: `dcdfreq=100`, `dcdfreq_mfpt=1`, `qspace_low/high=1/8`, `qspace_num_bins=100`, `DHAM_num_bins=100`, `use_dynamic_bins=False`.
  - Bias: `num_gaussian=10`, `load_global_gaussian_params_from_txt=False` (so the `params/*.txt` files are write-only).
  - Platform: `platform=CUDA`, evaluated at import (`config.py:17`).

### 1.3 Alanine dipeptide (`2D_Dialanine/`, `DIALA/`)
All bullets in this section are **[Verified]** by reading the cited lines, unless marked otherwise.
- `main.py` runs adaptive exploration in vacuum.
  - Force field: `ForceField('amber14-all.xml')` on `dialaB.pdb` (`:198-203`); 22 atoms, no solvent.
  - Bias: a 2-D Gaussian sum written as a `CMAPTorsionForce` map on a 20×20 (φ, ψ) grid (`:249-255`).
  - Target: (−110°, 75°) (`:34`). Loop: up to `max_propagation=10` × 100k steps (`:28-29`).
  - Output: `./trajectories/dialanine_exploring_traj_<i>.{dcd,txt}`, with no time tag, so each run overwrites the last (`:53,:71`).
  - It calls `plt.show()` repeatedly (`:90,:235,:304,:339`, and in `dham.py:154`), so it is interactive-only. **[Verified]**
- `unbiased_sim.py` sets up and runs the solvated system:
  - Setup: TIP3P 30 Å box, PME, 50 ps NVT warm-up, then NPT with 5 copies × 1 ns (500k steps).
  - Writes `trajectories/unbiased/*`, plus `saved_system/unbiased/system_<tag>_<i>_{system,integrator,state}.xml` (`:106-111`).
  - **`metaD_sim.py` is the same file**: `diff` shows only trailing blank lines (`:121-129`), so it does no metadynamics. **[Verified]**
- `create_psf.py` is a 12-line stub. It loads `dialaA.pdb`, builds a system and saves nothing (`:12` is a dangling comment). It uses the deprecated `simtk` namespace (`:1`).
- **`DIALA/`** holds byte-identical copies of `dialaA.pdb` and `dialaB.pdb` (md5 groups f3eb…, 85c5…). They arrived in the same commit as `2D_Dialanine` (7251d34, 2023-09-08), and nothing in the repo mentions `DIALA` (`git grep DIALA` is empty). **[Verified]**

## 2. `*_backforth` vs plain (from `diff`)
**NaCl**
- Byte-identical: `metaD_NaCl_sim.py`, `unbias_NaCl_sim.py`, `unbias_traj_fes_gen.py`, `Unbiased_Profile.npy` and all 6 `toppar/*`. **[Verified]**
- `config.py` drops `start_state`/`end_state*`, adds `qspace=np.linspace(0,15,num_bins)` (`:26`) and `max_cycle=20` (`:38`).
- `dham.py` makes symmetrisation optional (`use_symmetry`, default True in `run`) and reads `config.qspace` instead of `linspace(2,9,101)`.
- `util.py` uses `config.qspace`, `best_mfpt=np.inf`, an `a`-bound of 0.1–1.2 (was 0.1–2) and `tol=1e-4`.
- `analysis.py` swaps seaborn for matplotlib violin and box plots. Only the backforth dir has the extra `wham.py`, which no file imports (only `1D_langevin_sim_backforth/analysis.py:31` imports a *different* `wham.py`). **[Verified]**
- `mfpt_NaCl.py` changes:
  - `max_propagation=400` (`:37`).
  - Output dirs renamed: `trajectory/explore` (singular, `:99`), `params/` (`:115`), `figs/explore/` (`:368`).
  - The reach test becomes `|d−target|<0.1 Å` (`:150-154`).
  - An outer `while cycle_count < config.max_cycle` (`:265`) flips the target between 7.0 Å and the initial distance (`:252-257,:395-403`).
  - After each cycle it computes a full-trajectory DHAM FES plot (`:414-448`).
  - It no longer plots the unbiased reference.
- `mkdir -p trajectory/explore params distance figs/explore visited_state`.

**PMg** (each bullet is **[Verified]** by the `diff`/md5 output unless marked)
- Byte-identical: `metaD_NaCl_sim.py`, `dham.py`, `step3_input.{psf,crd,pdb}`, `traj_0.restart.pdb`, `toppar.str` and all 55 `toppar/*`. **[Verified]** md5 groups.
- `config.py`: `start_state=2`, `end_state=5` (Å, plain numbers instead of `Vec3` Quantities, `:18-19`); `qspace_high=10` (`:36`).
- `util.py`: 2000 random tries (was 1000), initial `a=1.0` (was 0.6), `a` upper bound 1.5 (was 1.0).
- `MSM.py`: a single `_is_adjacent(...verbose=False)` argument differs.
- `explore_sim_PMg.py`: an outer cycle loop between 5 Å and 2 Å, reach when `|d−target|<0.5 Å` (`:179-180`); DCD and figure names gain `cyc_<n>`; a per-cycle DHAM FES plot is added.
- `1D_PMg_backforth/total_steps_mfpt.csv` equals the first 20 lines of `1D_PMg/total_steps_mfpt.csv` (`head -20 | cmp`). It is a copied prefix, not a backforth result. **[Verified]** / **[Inferred]**

## 3. `1D_NaCl_TW/backup/` (86 files, 12.6 MB)
It is **not** a superseded copy of the top level. It is an earlier, different pipeline, partly copied from Oisín Moriarty's checkout of an `enhanced-sampling-workshop-2022` repo (`subprocesses*.py:5-6`, `traj_analysis.py:6`, `integrator_demo_TW.py:36`). **[Verified]**
- Git commits to the directory span 2023-10-16 to 2023-10-23 (`git log --format=%ad`). File names such as `_2ndAug` suggest the work itself dates from Aug 2023. **[Inferred]**
- **Different physics:**
  - The backup scripts build the system with `ForceField('amber14-all.xml','amber14/tip3p.xml')` and leave the modified CHARMM `.str` commented out as "old way" (`mfpt_bias_NaCl_gen.py:37-46`, `oisin_metadynamics.py:38-47`).
  - The top level uses the modified-LJ CHARMM file.
  - Results from the two are therefore not comparable. **[Verified]** / **[Inferred]**
- **Different algorithm:** `mfpt_bias_NaCl_gen*.py` optimises a bias once on a K matrix built from the precomputed `Unbiased_Profile.npy` (`util.try_and_optim`) and serialises the biased System to `output_files/NaCl_solvated_system_mfpt_bias_<n>`. `mfpt_bias_NaCl_MD_run*.py` reloads that XML (`:55`/`:19`), runs MD and logs `mm_metrics.csv`. `bar_chart.py` plots that file. **[Verified]**
- **Unique and worth keeping (CONFIRM):**
  - The umbrella-sampling notebooks `NaCl_Umbrella_Sampling{,_TW}.ipynb` (10 and 18 cells with outputs). They read and write `output_files/{NaCl_solvated_system,_1,bias.csv,dist_win_*.csv}` and `images/{US.jpg,dissociation.png,lin_spaced_windows.png,solvated_system.png}`, and import a `src.wham`/`src.dham` package that is absent from the repo.
  - Umbrella-sampling data: `NaCl_US_long.npy` (shape 10×10000, referenced nowhere), `all_dists.npy` (7500; the name only appears as a notebook variable) and `dist_win_*.csv`.
  - 26 serialised Systems (8.5 MB): 24 biased `_mfpt_bias_*` plus the unbiased `NaCl_solvated_system` and `_1`.
  - Results: `mm_metrics.csv`, older `total_steps_*.csv`, figures.
- **Not runnable in place [Verified]:** no `toppar/` or `util.py` in `backup/`, while the scripts use `from util import *` and `toppar/...`.
  - `subprocesses*.py` and `traj_analysis.py` hard-code `/Users/oisinmoriarty/...`.
  - `test2.py` is broken: `app.PME` with `app` undefined (`:10`), a non-existent `CustomCVForce.distance` (`:23`), and missing `input.pdb`/`input.psf`.
  - `metadynamics.py` is a vendored `simtk` Metadynamics class with the old `deltaT` API; nothing imports it.
- **Verdict:** the code is superseded by `mfpt_NaCl.py` and `metaD_NaCl_sim.py`; the data, notebooks and results are not. Archive it as a unit (CONFIRM), apart from the narrow DELETE-SAFE items in §7.

## 4. Inputs actually used (all cwd-relative; no absolute paths in live code) **[Verified]** by grep
| Dir | Loaded by code | Present but loaded by nothing live |
|---|---|---|
| NaCl (both) | `toppar/step3_input.psf`, `toppar/step3_input25A.pdb`, `toppar/toppar_water_ions.str` (`CharmmParameterSet`, `mfpt_NaCl.py:34-35,214`); `Unbiased_Profile.npy` (`mfpt_NaCl.py:207`, `util.py:139,316,380`) | `toppar/step3_input.crd` (no reference anywhere, incl. glob/`.crd` grep); `toppar/step3_input.pdb` (backup notebooks only); `toppar/step3_input_2A.pdb` (backup `*_2.py:14-15` only) |
| PMg (both) | `step3_input.psf` (`:30`), `traj_0.restart.pdb` (`:31`), `toppar.str` (`:275`), which `read_params` turns into `CharmmParameterSet(*all 55 files)` (`:238-246`) | `step3_input.crd` (14.1 MB, never referenced), `step3_input.pdb` (7.9 MB, comment at `:31` only) |
| Dialanine | `dialaB.pdb` (`main.py:23,198`, `unbiased_sim.py:37`); `dialaA.pdb` (`create_psf.py:4` only); OpenMM built-ins `amber14-all.xml`, `tip3p.xml` | `saved_system/**` (written at `unbiased_sim.py:106-111`, read by nothing) |

PMg `toppar/`: `diff <(sed s#toppar/## toppar.str|sort) <(ls toppar|sort)` is empty, so all 55 files are loaded. Most (lipid, carb, nano, label, polymer …) are probably unnecessary for protein+GDP+Mg+Pi+water+KCl [Inferred]. Pruning them means editing `toppar.str`, so all 55 stay **KEEP**.

## 5. Bugs and smells
| # | Where | Finding |
|---|---|---|
| B1 | `mfpt_NaCl.py:47-50,100`; `util.py:306,387`; `dham.py:87`; PMg `explore_sim_PMg.py:299,376` + PMg `dham.py:89` | The bias is stored as a flat `[a0..a9,b0..b9,c0..c9]` and fed to DHAM via `reshape(-1,3)` / `reshape(1,n,3)`. DHAM then unpacks each row as (a,b,c), so it sees `(a0,a1,a2)`, … and the unbiasing uses the wrong Gaussians. **[Verified]** by reading. |
| B2 | `mfpt_NaCl.py:242` vs `:295` | Prop-0 parameters are saved in kJ/mol & nm (`:47-49`); later ones in kcal/mol & Å (`:299-301` comments). DHAM evaluates every set on an Å grid in kcal/mol. **[Verified]** |
| B3 | `mfpt_NaCl.py:112-114`, PMg `:172-175`, NaCl-bf `:150-154`, PMg-bf `:179-182` | The reach loops have no `break`, so they record the **last** frame ≥ target, not the first crossing. Fingerprint: 11/25 values in `1D_NaCl_TW/total_steps_mfpt.csv` and 10/38 in `bak/` are ≡4999 mod 5000 (chance ≈1/5000). For PMg, 2/61 are ≡49900 mod 50000 (≈0.1 expected), which is weaker evidence. **[Verified]** counts. |
| B4 | `mfpt_NaCl.py:272,346,351`; PMg `:334,410,415` | `i`/`i_prop` is incremented after the reaching propagation, so `total_steps` over-counts by one `propagation_step`. If `max_propagation` is hit, `reach is None` gives a `TypeError` at `:351`. **[Inferred]** |
| B5 | PMg `explore_sim_PMg.py:415` | `reach` is already in steps (`dcdfreq_mfpt=1`, `:175`) but is multiplied by `config.dcdfreq=100`. All 61/61 PMg values are multiples of 100. **[Verified]** awk. |
| B6 | PMg `explore_sim_PMg.py:211-215` (bf `:221`) | `custom_bias.addBond(7799,7840)` sits **inside** `for i in range(num_gaussian)`, so 10 identical bonds apply the bias 10×. NaCl has it outside the loop (`mfpt_NaCl.py:175`). **[Verified]** |
| B7 | NaCl-bf `mfpt_NaCl.py:403,453`; PMg-bf `:436,481` | `reach=None` is reset after every cycle, then `reach*...` runs after the loop, which should raise `TypeError`. That would explain why NaCl-bf has no CSV and PMg-bf only has a copied prefix. NaCl-bf also has `cycle_count=0` at `:226` outside `for i_sim` (`:239`), so later sims skip cycling, and `i` at `:453` is a stale loop variable from `:417`. **[Inferred]** |
| B8 | `run_NaCl_explore_sim.sh:4,7-9` (+ `backup/` copy) | It targets a non-existent script; `conda activate` fails in a non-interactive shell; there is no shebang (`{1..20}` needs bash). 20 concurrent processes would share the second-resolution `time_tag` (`mfpt_NaCl.py:229`), so they clobber each other's `gaussian_params/<tag>_*.txt` (re-read at `:100`) and interleave `total_steps_mfpt.csv`. **[Inferred]** |
| B9 | `config.py:26` (NaCl, both), PMg `config.py:17`; `mfpt_NaCl.py:221`, `unbias_NaCl_sim.py:43,66`, `metaD_NaCl_sim.py:86`, `main.py:22,204`, `unbiased_sim.py:24` | `getPlatformByName('CUDA')` runs at import, so anything importing `config`, `util` or `dham` (even the analysis-only `unbias_traj_fes_gen.py`) fails without CUDA. The fetched platform is then never passed to the context (`mfpt_NaCl.py:247`, `unbias_NaCl_sim.py:75`, `main.py:254`). **[Verified]** code; **[Inferred]** failure. |
| B10 | `2D_Dialanine/main.py:66-67,251,312` | φ/ψ use atoms `[5,7,9,15]`/`[7,9,15,17]`. The 0-based atoms in `dialaB.pdb` are 4 C, 5 O, 6 N, 7 H, 8 CA, 9 HA, 14 C, 15 O, 16 N, 17 H. These are therefore O–H–HA–O and H–HA–O–H, both for the CV and for the CMAP bias. The correct sets are `[4,6,8,14]` and `[6,8,14,16]`. **[Verified]** |
| B11 | `2D_Dialanine/unbiased_sim.py:70,80` | `T=5` (the warm-up start) is reused for `MonteCarloBarostat(..., T*kelvin)`. The saved XML shows `temperature="5"` for the barostat and 300 K for the integrator, against the 298.15 K at `:25`. The comments say "50ns"/"1000 ns", but 25k/500k steps × 2 fs = 50 ps / 1 ns (`:30,:65-66,:83`). **[Verified]** |
| B12 | `1D_NaCl_TW/unbias_traj_fes_gen.py:44-45,15` | `DHAM(gaussian_params)` and `setup(CV,T,prop_index)` miss the required `num_bins`/`time_tag` (`dham.py:57,64`), so this raises `TypeError`. It also loads one hard-coded `visited_state/20231019-164036_*.txt`. **[Verified]** signatures. |
| B13 | `1D_PMg*/metaD_NaCl_sim.py:40-41,54,65` | It loads `toppar/step3_input.psf` and `step3_input25A.pdb`, which are absent in the PMg dirs. **[Verified]** It uses `BiasVariable(r, 2.0, 10, 0.2)`, i.e. Å values on a CV in nm. **[Inferred]** |
| B14 | `analysis.py:92-117` vs scripts | The box plot compares MSM-opt (≥8 Å, 300 K, step-resolved) against unbiased (7 Å, **298.15 K**, 100-step stride) and metaD (7 Å, 300 K, 500-step stride; a run that never crosses records its last frame, `metaD_NaCl_sim.py:134-135`). Thresholds and temperatures differ (`unbias_NaCl_sim.py:46`). **[Verified]** |
| B15 | PMg `explore_sim_PMg.py:278` + pdb `CRYST1 0` | A 100k-atom solvated box run with `CutoffNonPeriodic` and no box. Similarly, NaCl with the default NoCutoff (`mfpt_NaCl.py:217`). **[Inferred]** physics smell. |
| B16 | misc. | Magic atom indices 7799/7840 (PMg `:134-135`). Hard-coded `range(10)` with `+10/+20` offsets instead of `num_gaussian` (`mfpt_NaCl.py:298-301`, PMg `:380-383`, PMg `:84`). `sys.path.append("..")` placed *after* `from util import *` (`mfpt_NaCl.py:19-20`, PMg `:23,29`). Dead code: PMg `animate`, `last_dcd_frames`, `update_metrics`, `add_backbone_posres`, plus `matrix_power(M,10000)` per propagation (`:45-75,248,323,399`). `DCDFile(dt=2)` with a unitless int is read as 2 ps (`unbias_NaCl_sim.py:80`, `main.py:54`). PMg x-label says "nm" for Å (`:359`). The PMg distance file is named `NaCl_exploring_traj` (`:422`). `np.digitize` + `ravel_multi_index` overflow at φ/ψ=π (`main.py:78-82`). **[Verified]** |
| B17 | `backup/subprocesses*.py:5-6,11-12`, `backup/traj_analysis.py:6` | The only absolute paths, all `/Users/oisinmoriarty/...` (macOS). No HPC or Windows paths in this area. **[Verified]** grep. |

## 6. Third-party dependencies (for an environment file)
Every package below is **[Verified]** by the import-statement grep over this area, with the cited `file:line`.
- **Required:** `openmm` with a **CUDA** platform (`openmm.app` incl. `Metadynamics`/`BiasVariable`, `CharmmPsfFile`, `CharmmParameterSet`, `CMAPTorsionForce`, `XmlSerializer`), `mdtraj`, `numpy`, `scipy`, `matplotlib`, `tqdm`.
- `pillow`: imported but unused at `mfpt_NaCl.py:28` and `unbias_NaCl_sim.py:20`; it still has to be installed for those imports to succeed.
- **Optional / lazy:**
  - `seaborn` (`1D_NaCl_TW/analysis.py:91`)
  - `numba` (`1D_PMg*/MSM.py:241`)
  - `plotly` (`1D_PMg*/util.py:285`)
  - `jax`, `scikit-learn` (only `1D_NaCl_TW_backforth/wham.py:4,10`, unused)
  - `pandas` (`backup/bar_chart.py`)
  - `jupyter` (backup notebooks)
  - `tkinter` (`matplotlib.use('TkAgg')`, `backup/integrator_demo*.py`)
- **Legacy namespace:** `simtk.*` is deprecated (`2D_Dialanine/create_psf.py:1`, `backup/{metadynamics,test2}.py`). The conda env name used by the authors is `biophys_env` (`run_NaCl_explore_sim.sh:4`).

## 7. Redundancy table
Repo-wide `git grep` was run for each basename, including `.ipynb`, `.m`, `.sh` and glob or `.crd` patterns. "Refs" lists readers only; writers are noted.

| Path | Size | Evidence | Refs found | Class |
|---|---|---|---|---|
| `DIALA/dialaA.pdb`, `DIALA/dialaB.pdb` | 2×1.8 KB | md5-identical to `2D_Dialanine/` copies | none (`git grep DIALA` empty) | **DELETE-SAFE** |
| `2D_Dialanine/saved_system/unbiased/*_system.xml` except one per md5 group (5 of the 160135/160328 group + 4 of the 121832 group) | 9×0.5 MB ≈4.5 MB | md5 groups x6 and x5 | written only (`unbiased_sim.py:106`, `metaD_sim.py:106`) | **DELETE-SAFE** |
| `.../*_integrator.xml` except one per group (5+4) | 9×~170 B | md5 groups x6 and x5 | written only | **DELETE-SAFE** |
| `1D_PMg_backforth/step3_input.crd`, `step3_input.pdb` | 14.1+7.9 MB | md5 = `1D_PMg/` copies | crd: none; pdb: comment only (`:31`) | **DELETE-SAFE** if `1D_PMg/` copies are kept |
| `1D_NaCl_TW_backforth/toppar/step3_input{.crd,.pdb,_2A.pdb}` | 0.51 MB | md5 = `1D_NaCl_TW/toppar/` | only `1D_NaCl_TW/backup` code and notebooks | **DELETE-SAFE** if plain copies are kept |
| `1D_PMg_backforth/total_steps_mfpt.csv` | 160 B | = `head -20` of `1D_PMg/` CSV (a prefix, not md5-identical) | none (append-only output) | **DELETE-SAFE** if `1D_PMg/total_steps_mfpt.csv` is kept |
| `1D_PMg_backforth/metaD_NaCl_sim.py` | 4.7 KB | md5 = `1D_PMg/` copy; cannot run (B13) | none | **DELETE-SAFE** |
| `1D_NaCl_TW/backup/fig/Figure_1.png` | 53 KB | md5 = `backup/fig/explore_traj_fig/Figure_1.png` | none | **DELETE-SAFE** |
| `2D_Dialanine/figs/DHAM_prop1.png` | 18 KB | md5 = `figs/Example_Amax5/DHAM_prop1.png` | none (code writes only `figs/DHAM.png`) | **DELETE-SAFE** |
| `1D_NaCl_TW/backup/biasData/bias_1273876585_1.npy` | 136 B | shape (1,); md5 = a `1D_langevin_sim/.../MetaD_auxfiles` file | none ("biasData" unreferenced) | **DELETE-SAFE** |
| `1D_NaCl_TW/backup/test2.py` | 1.6 KB | broken scratch (undefined `app`, missing inputs) | none | **DELETE-SAFE** (judgement, not md5) |
| `1D_NaCl_TW/backup/run_NaCl_explore_sim.sh` | 205 B | 1-char diff from top-level; same dead target | none | **DELETE-SAFE** (judgement, not md5) |
| `1D_NaCl_TW/backup/mfpt_biase_NaCl_gen_2.py` | 4.7 KB | typo-named copy; `diff` vs `mfpt_bias_NaCl_gen_2.py` is 1 line (output name) | none | **DELETE-SAFE** (judgement, not md5) |
| `1D_PMg/step3_input.crd`, `step3_input.pdb` | 22 MB | CHARMM-GUI originals; unreferenced | none / comment | **CONFIRM** (provenance; needed to restart from scratch) |
| `1D_NaCl_TW/toppar/step3_input{.crd,.pdb,_2A.pdb}` | 0.51 MB | unreferenced by live code | backup scripts and notebooks only | **CONFIRM** (follows `backup/`) |
| `1D_NaCl_TW/backup/**` (rest: notebooks, `output_files/`, `NaCl_US_long.npy`, `all_dists.npy`, `mm_metrics.csv`, figures, scripts, `Unbiased_Profile.npy`) | ≈12.5 MB | older pipeline + unique US data/results (§3) | only by other `backup/` files (relative paths) | **CONFIRM** (archive as a unit) |
| `1D_NaCl_TW/bak/total_steps_mfpt.csv` | 269 B | older 38-run MFPT results, unique | none | **CONFIRM** |
| `1D_NaCl_TW_backforth/wham.py` | 12.6 KB | unique variant; nothing imports it | none | **CONFIRM** |
| `1D_NaCl_TW_backforth/{metaD_NaCl_sim,unbias_NaCl_sim,unbias_traj_fes_gen}.py` | 12 KB | md5 = plain copies; config values used are identical | none (entry points) | **CONFIRM** |
| `1D_PMg/metaD_NaCl_sim.py` | 4.7 KB | stale NaCl script, cannot run here (B13) | none | **CONFIRM** |
| `1D_PMg/total_steps_mfpt.csv` | 527 B | unique results (inflated by B3–B5) | none | **CONFIRM** |
| `2D_Dialanine/metaD_sim.py`, `create_psf.py` | 5.3 KB, 0.3 KB | copy of `unbiased_sim.py` / stub | none | **CONFIRM** |
| `2D_Dialanine/saved_system/**` (11 unique `*_state.xml` + 1 system/integrator per group) | ≈8 MB | run end-states; unread | written only | **CONFIRM** |
| `2D_Dialanine/trajectories/*`, `figs/**` (rest) | 2.3 MB, 1.4 MB | outputs of `main.py`/`dham.py`; overwritten per run | written/reread within one run only | **CONFIRM** |
| `1D_NaCl_TW/total_steps_{mfpt,metaD,unbiased}.csv` | <1 KB | md5 = `paper_publication/data/time2reach/NaCl/` | `analysis.py:92-94` | **KEEP** |
| `1D_NaCl_TW_backforth/{Unbiased_Profile.npy,toppar/step3_input.psf,step3_input25A.pdb,toppar_water_ions.str}` | 0.37 MB | md5 = plain | own scripts, relative paths | **KEEP** |
| `1D_PMg_backforth/{step3_input.psf,traj_0.restart.pdb,toppar.str,toppar/*}` | 51 MB | md5 = `1D_PMg/` | own `explore_sim_PMg.py:30-31,281` | **KEEP** (dedupe only by restructuring) |

## 8. Open questions for the authors
1. Is "PMg" the "Balint system"? Which protein is PROA/PROB, and which reaction is studied (Pi release from a GDP·Mg site)? Who is Balint (`# Balint added this`)?
2. Why is PMg run with `CutoffNonPeriodic` and no box, and NaCl with the NoCutoff default (B15)? Is that intentional?
3. Should the published NaCl and PMg step counts (`paper_publication/data/time2reach/NaCl/*` equals `1D_NaCl_TW/*.csv`) be regenerated after B1–B5 are fixed?
4. Should `1D_*_backforth` become a flag (`--backforth`) on one script, sharing a single `toppar/` and input set (≈73 MB of exact PMg duplicates)? Do the backforth scripts ever finish (B7)?
5. Can PMg `toppar.str` be pruned to the files the PSF actually needs (prot, na/nad_ppi, water_ions, …)?
6. Should `backup/` move to an archive branch or release, with the umbrella-sampling data (`NaCl_US_long.npy`, `dist_win_*.csv`) kept as the reference FES source? Where did `Unbiased_Profile.npy` (2×200) come from?
7. Was `2D_Dialanine/metaD_sim.py` meant to implement metadynamics? Should the dialanine trajectories and figures stay tracked despite `.gitignore:11`?
8. Should inputs under `/1D_*/*` be un-ignored, e.g. `!/1D_PMg/toppar/**`, so they cannot be silently dropped?

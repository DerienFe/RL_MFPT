# 04 — RL experiments and their artefacts

Scope: `RL_Qagent_1D/`, `RL_DQN_position/`, `RL_DQN_1D_N100/`, `RL_offline_CQL/`, `RL_posM_env/`, `RL_pos_PPO_1D/`,
`RL_PPO_implementation/`, `RL_matlab/`, `RL_CLEANRL/runs/`, `runs/`, root `pos_M_DQN_model_*.pt`, `data_gen.py`,
`data_1D_20N_10Gaussian.pt`, `data/data_1D_20N_10Gaussian.pt`, `figures.pptx`, `test.png`, `fes_visualization.png`.
Method: read all code, `git log`/`git grep`, md5, and **torch-free inspection of `.pt` files**: each is a zip whose
`data.pkl` I parsed with a restricted unpickler (stub classes only, nothing executed) to get tensor shapes/values.
Torch, scipy and tensorboard are not installed here, so nothing was trained or `torch.load`ed.
Labels: **[Verified]** = seen in file/diff/command output; **[Inferred]** = reasoning, attribution or runtime behaviour not executed.

## Summary

- The RL work spans **2023-06-19 → 2023-08-09** in commits (source files only; 45d011f/4128108 touched only PNG/pyc) [Verified].
  It went through: K-matrix-state DQN (`RL_Qagent_1D`) → position-state DQN (`RL_DQN_position`, broken) → N=100 fork with a
  decision-transformer stub (`RL_DQN_1D_N100`) → offline dataset + CQL stub (`data_gen.py`, `RL_offline_CQL`) →
  (position, Markov-matrix) DQN (`RL_posM_env`) → gym env + PPO (`RL_pos_PPO_1D`, **latest Python**), with a parallel MATLAB
  RL-Toolbox attempt (`RL_matlab`, DDPG/TD3) [Verified dates; Inferred lineage].
- None of the folders reaches a validated result. Every entry point has at least one crash-level or logic bug (see Bugs) [Verified code; Inferred runtime].
- `runs/` mixes **third-party tutorial runs** (CleanRL `ppo`/`ppo_atari`, ICLR-blog `ppo_multidiscrete`) with the
  authors' own `ppo_trainer` runs from `RL_pos_PPO_1D/ppo_trainer.py`. The hparam text inside the event files shows which is which [Verified].
- The two `data_1D_20N_10Gaussian.pt` files are **different versions**. The root copy is what committed `data_gen.py` writes, and its
  action field is aliased: every record of an episode holds the full 10-position list, so the per-step action is recoverable only as element `i % 10`.
  The `data/` copy has per-step action tensors from a generator variant that is not in git. Only the `data/` copy is read by code [Verified; generator variant Inferred].
- Root `pos_M_DQN_model_*.pt` total **84.7 MB** (not ~100 MB). Their filenames misstate N; the tensor shapes are what count [Verified].
- `RL_PPO_implementation/` is an unused copy of the ICLR-blog `ppo_details` dependency spec [Verified unused; Inferred origin].

## Per-folder table

| Folder | Algorithm | Env: state / action / reward | Entry point & how to run | Status |
|---|---|---|---|---|
| `RL_Qagent_1D` (first commit af86dcb 2023-06-19) | DQN: 3-layer MLP, replay, target net (`dqn_agent.py:9-97`); A2C variant (`a2c_agent.py`, `main_a2c.py`); unused minGPT copy (`transformer_agent.py`) [Verified] | `All_known_1D` (`env.py:12`). State = full N×N rate matrix K. Action = grid index of one Gaussian, 10 per episode (`main.py:12`). Reward = fractional MFPT reduction vs unbiased (`env.py:75`) [Verified] | `cd RL_Qagent_1D && python main.py` (bare `from util import *`). Writes `./model_29thJune*_N100.pt`, `./0..10.png`, `animation.gif` to cwd (`main.py:75,115,158`) [Verified] | DQN: working prototype, N=20 (`main.py:8`). Commit msg "1D_all_known - DQN finished" (7505da4, 2023-06-28) [Verified]; not validated [Inferred]. A2C (`main_a2c.py`, 2023-06-27): broken (B5b) |
| `RL_DQN_position` (a1d5f3b 2023-06-20, single commit) | DQN (agent = Qagent copy, 2-line diff) | `position_1D` (`env.py:18`). State = scalar position. Action = Gaussian centre. Reward = −\|pos−end\| after sampling `expm(K·dt)` (`env.py:50-67`) [Verified] | `main.py`. Imports need the folder on sys.path, but it saves to `./RL_Qagent_1D/model_1.pt` (`main.py:52`) [Verified] | Abandoned. Crashes (`main.py:46`, see B7-B10) [Verified code; Inferred runtime] |
| `RL_DQN_1D_N100` (71ce56f 2023-07-07) | DQN with 6-layer MLP (`dqn_agent.py`, 15-line diff vs Qagent). Decision-transformer stub: `mingpt.py`, `mingpt_train.py`, `util.sample` [Verified] | Same `All_known_1D`, N=100, 8→88 (`main.py:8-11`). Reward = 1/MFPT (`env.py:79`) [Verified] | `cd RL_DQN_1D_N100 && python main.py` [Verified] | Fork of Qagent for N=100. The DT part is a non-functional stub: `mingpt_train.py` has only imports [Verified] |
| `RL_offline_CQL` (71ce56f) | Conservative Q-learning (offline) | Offline transitions from `data/data_1D_20N_10Gaussian.pt`: K 20×20, Gaussian positions, reward = MFPT reduction [Verified] | `cd RL_offline_CQL && python main.py` (`../data/...`, `main.py:8`) [Verified] | Abandoned stub. Crashes at `main.py:20/23/24` [Verified code] |
| `RL_posM_env` (71ce56f→dfee4dd 2023-07-17) | DQN, two-branch net: MLP(position) + CNN(M), outputs averaged (`posM_dqn.py:14-51`) [Verified] | `posM_env_1d` (`env.py:18`). State = (position, Markov matrix M=expm(K·dt)). Action = length-N Gaussian-count vector (top-k Q, k=10). Reward = −\|end−pos\| after `simulate` 1000 steps; done at end state (`env.py:72-111`) [Verified] | `python RL_posM_env/main.py` **from repo root**: the checkpoint names are cwd-relative (`posM_dqn.py:211`, `main.py:50`), and the files sit at root [Verified paths; Inferred cwd] | Prototype. Commit "finished posM model" 2023-07-17, then "To migrate to ppo agent" (8501a47) [Verified] |
| `RL_pos_PPO_1D` (8501a47 2023-07-20 → b4d8ca7 2023-08-09) | PPO, MultiDiscrete actions. Structure and "TRY NOT TO MODIFY" comments of CleanRL/ICLR-blog `ppo_multidiscrete.py` [Verified text; Inferred source] | gym `pos_1D_env` (`env.py:21`). Obs = `Discrete(N)` position. Action = `MultiDiscrete([num_gaussians]*N)` counts (`env.py:47`). Reward = −\|end−pos\| (`env.py:107`). Registered `pos_1D-v0`: N=100, 3 gaussians, 200 steps (`ppo_trainer.py:17-31`) [Verified] | `python RL_pos_PPO_1D/ppo_trainer.py` **from repo root** (`runs/` at `:159`, `./RL_pos_PPO_1D/models/` at `:325`). `local_env_test.py` loads a model [Verified] | **Latest Python RL.** Prototype; the logged runs stopped after ≤73 PPO updates (B24) [Verified] |
| `RL_PPO_implementation` (8501a47) | none (only `pyproject.toml` + `poetry.lock`) | — | none | Third-party dependency spec, unused (see Redundancy table) |
| `RL_matlab` (b4d8ca7 2023-08-09; sessions dated 2023-06-26, 2023-07-31) | MATLAB RL Toolbox: DDPG in `ReinforcementLearningDesignerSession_1.mat`, TD3 in `_TW_1/_TW_2.mat` [Verified class names in MAT streams; Inferred use] | Obs = position (`rlFiniteSetSpec(1:100)`). Action = continuous 2×20 (centre, width) for 20 Gaussians. Reward = farthest pos − end, −10 / +100 (`myStepFunction.m:45-58`) [Verified] | `main_RL_adaptive_1D.m` builds `MFPT_env` and says "do the rest… in the RL designer app" (`:13-16`). `MFPT_env_Edina.m` builds an `rlFunctionEnv` [Verified] | Exploratory; buggy (M1-M7) |

**Which is latest:** `RL_pos_PPO_1D` (last source commit 2023-08-09; newest model `ppo_trainer__1__1690282116` = 2023-07-25 UTC)
and `RL_matlab` (same commit; session files created 2023-07-31) [Verified].

## Lineage between folders

- `RL_Qagent_1D` → `RL_DQN_1D_N100`: copied and parameterised. Diffs: `main.py` 12 lines, `env.py` 8, `dqn_agent.py` 15, `util.py` 59 (adds `set_seed`, `sample`) [Verified].
- `RL_Qagent_1D/dqn_agent.py` → `RL_DQN_position/dqn_agent.py`: identical except the trailing newline (2 changed lines) [Verified]. `util.py`: 31 lines differ, including a flipped bias sign (B11).
- `RL_Qagent_1D/util.py` → `data_gen.py` (`from RL_Qagent_1D.util import *`, `data_gen.py:2`) → `data/` → `RL_offline_CQL` [Verified].
- `RL_posM_env/util.py` ↔ `RL_pos_PPO_1D/util.py`: 21 lines differ. posM has the corrected `gaussian()` and `M.T` inside `simulate`; PPO keeps the old `gaussian()` and transposes in `env.py:96` [Verified].
- `RL_CLEANRL` + `RL_PPO_implementation` + tutorial `runs/` (2023-07-17→25) were the PPO study behind `ppo_trainer.py` [Inferred from dates and shared code shape].
- No other repo code imports any RL folder. The only cross-reference is `data_gen.py` → `RL_Qagent_1D.util` [Verified `git grep`].

## Model & dataset artefacts (who writes / reads)

| Artefact | Content (tensor shapes) | Written by | Read by |
|---|---|---|---|
| `pos_M_DQN_model_11thJuly_1.pt`, `_13July_1.pt` (20.7 MB each) | posM DQN, N=100 (`fc1M` 128×40000) [Verified] | older `posM_dqn.py` save names (355654a, 44f070f) [Verified history] | nothing at HEAD |
| `pos_M_DQN_model_13July_2.pt` (0.95 MB) | N=20 (`fc1M` 128×1600, `fc2M` 20×128) [Verified] | `posM_dqn.py` at 44f070f [Verified] | nothing |
| `pos_M_DQN_model_13July_2_N20.pt` (0.95 MB) | N=20 [Verified] | — | `RL_posM_env/main.py:50` (shape mismatch with HEAD's N=100 net, B17) [Verified] |
| `pos_M_DQN_model_13July_2_N100.pt` (20.7 MB) | N=100. Internal zip prefix `pos_M_DQN_model_13July_2_N20/`, i.e. renamed after saving [Verified] | — | nothing |
| `pos_M_DQN_model_14July_1_N20.pt` (20.7 MB) | **N=100** despite the name [Verified] | `posM_dqn.py:211` (HEAD) [Verified] | nothing |
| `data_1D_20N_10Gaussian.pt` (root, 75.2 MB) | 10,000 records `[K(20,20) f64, action:list, reward f64, K'(20,20), done]`. Action list aliased: 1000 distinct list objects; rec0 and rec1 both `[8,4,8,17,9,9,17,16,14,14]` [Verified] | `data_gen.py:47` (cwd-relative, run from root) [Verified] | **nothing** |
| `data/data_1D_20N_10Gaussian.pt` (78.2 MB) | 10,000 records. Action = f64 tensor of the positions so far: lengths cycle 1..10, e.g. `[4]`, `[4,6]`, `[4,6,14]` [Verified] | a `data_gen.py` variant not in git [Inferred]. Both files were added in the same commit 71ce56f [Verified] | `RL_offline_CQL/main.py:8` [Verified] |
| `RL_Qagent_1D/model_26thJune1_{100,200,400}.pt` (5.2 MB each) | N=100 3-layer DQN (`fc1` 128×10000), all three inspected [Verified] | Qagent `main.py` train() name pattern; re-added in 71ce56f [Verified] | nothing |
| `RL_Qagent_1D/model_29thJune1_N20.pt` (284 KB) | N=20 (`fc1` 128×400) [Verified] | an older `main.py` naming (`_N20`) [Inferred] | nothing (`main.py:73` loads `…_N100.pt` names only after writing them itself) |
| `RL_Qagent_1D/models/*.pt` (44 files, 15.0 MB) | `model_{1..10}`, `model_26thJune*`, `model_27thJune*`: N=20 DQN (`fc1` 128×400, `fc3` 20×128). `model_abc_*`, `model2_abc_*`: `fc3` 500×128 (older 500-action design) [Verified samples]. Not the same files as the folder-root namesakes (284 KB vs 5.2 MB) | Qagent `main.py` history: `./RL_Qagent_1D/model_{i}.pt`, `model2_abc_{i}` (ac26d0f, a18e4fd); moved to `models/` in 7505da4 [Verified] | nothing at HEAD. `RL_DQN_position/main.py:52,60` points at `./RL_Qagent_1D/model_1.pt` (folder root, not `models/`) and saves before it loads [Verified] |
| `RL_pos_PPO_1D/models/ppo_trainer__1__{1690192924,1690198587,1690282116}.pt` | actor 1000/500/300 × 64 = 10/5/3 Gaussians × N=100 [Verified] | `ppo_trainer.py:325`; timestamps match `runs/ppo_trainer__1__*` [Verified] | `local_env_test.py:52` loads `…1690192924.pt` only [Verified] |
| `RL_Qagent_1D/total_rewards.txt` (2500 lines), `figs/*` (16 files, 413 KB) | training curve; FES frames + `animation.gif` | `main.py:82` (now inside a dead string), `main.py:115,124,158` [Verified] | nothing |

## runs/ provenance

The event files were parsed with `strings` for the hparam table and tag names. All were written on host `ArchBermond` [Verified].

| Run dir(s) | Date (UTC) | Hparams found | Origin |
|---|---|---|---|
| `RL_CLEANRL/runs/CartPole-v1__ppo__1__1689612896`, `…/BreakoutNoFrameskip-v4__ppo_atari__1__1689614171` | 2023-07-17 | `env_id`, `wandb_project_name\|cleanRL\|` [Verified] | CleanRL examples `ppo.py`/`ppo_atari.py` [Inferred]. Third-party tutorial output |
| `runs/CartPole-v1__ppo__1__{1689853738,1689853759,1690190570,1690190804}`, `runs/BreakoutNoFrameskip-v4__ppo_atari__1__1690288022` | 07-20 → 07-25 | same `cleanRL` project. The two 632 B files have no scalars [Verified] | CleanRL tutorial output [Inferred] |
| `runs/MicrortsMining-v1__ppo_multidiscrete__1__1689843103` (684 B) | 2023-07-20 | `gym_id`, `wandb_project_name\|ppo-implementation-details\|`; no scalars [Verified] | ICLR-blog "37 PPO details" repo, whose deps are `RL_PPO_implementation` [Inferred]. Crashed at start |
| `runs/ppo_trainer__1__*` (9 runs, 177 KB) | 07-24 → 07-25 | `wandb_project_name\|pos1D_ppo\|` = `ppo_trainer.py:46`; later runs show `num_steps\|500\|` [Verified] | **Authors' own** (`RL_pos_PPO_1D/ppo_trainer.py`). 1690192924/1690198587/1690282116 logged 73/18/20 PPO updates; 1690213668 and 1690214796 logged 1–2; four (≤992 B) logged a few episodes only [Verified counts] |
| `RL_pos_PPO_1D/runs/ppo_trainer__1__*` (4 × 673 B) | 2023-07-20 | hparams only, no scalars [Verified] | Own runs started inside the folder that died immediately [Inferred] |

`SummaryWriter` occurs only at `RL_pos_PPO_1D/ppo_trainer.py:12,159`, and no code reads `runs/` [Verified `git grep`].
`ppo_trainer` is repo code; every CartPole/Breakout/Microrts run comes from scripts that are not in the repo.

## RL_matlab

- **Purpose.** `MFPT_env.m` is a `rl.env.MATLABEnvironment` class. `MFPT_env_Edina.m` with `myResetFunction.m`/`myStepFunction.m` is the function-handle env (`rlFunctionEnv`).
  `propagate_N_steps.m`, `bias_K_1D.m`, `gaussian_bias.m`, `create_K_1D.m`, `compute_free_energy.m` and `mfpt_calc.m` are helpers they call.
  `explore_the_network.m` and `Markov_mfpt_calc.m` have **no caller in RL_matlab** [Verified `grep`].
- **Sessions.** All three are MAT v5 files saved on PCWIN64 [Verified header]. `_1.mat` (20.4 MB, 2023-06-26) holds `rlDDPGAgent`, `MFPT_env` and `rlNumericSpec`.
  `_TW_1.mat` and `_TW_2.mat` (3.5 MB each, 2023-07-31 15:30/16:23) hold `rlTD3Agent`, `rlFunctionEnv`, `myStepFunction` and `myResetFunction` [Verified strings from zlib streams].
  No code loads them; only the RL Designer app does. They are needed only to reopen those Designer sessions and their agents [Inferred].
- **Byte-identical copies** [Verified `cmp`]:
  - `compute_free_energy.m` = `1D_model_potential_{Edina,Simian}/` = `2D_model_potential/`. Called by `MFPT_env.m:77` and `myResetFunction.m:20`.
  - `mfpt_calc.m` = `1D_model_potential_{Edina,Simian}/`. Called by `MFPT_env.m:78`.
  - `explore_the_network.m` and `Markov_mfpt_calc.m` = `1D_model_potential_{Edina,Simian}/`. Not called.
- **Near-duplicates:** `create_K_1D.m` differs only by 2 blank lines. `propagate_N_steps.m` has 9 changed lines vs Simian's (drops `tot_time`, see M6). `main_RL_adaptive_1D.m` differs entirely (135 lines) from Simian's [Verified].

## Bugs / smells (file:line)

Shared numerics
- B1 `gaussian()` precedence `/ 2*(c**2)` multiplies by c² instead of dividing, so σ_eff = 1/c: `RL_Qagent_1D/util.py:11`, `RL_DQN_1D_N100/util.py:15`, `RL_DQN_position/util.py:11`, `RL_pos_PPO_1D/util.py:15`. Fixed only in `RL_posM_env/util.py:18`. With c=0.5 (`RL_pos_PPO_1D/env.py:37`), PPO Gaussians are 4× wider than intended [Verified].
- B2 The bias grid `np.linspace(0, N, N)` (spacing N/(N−1)) does not line up with the integer action positions or the plotting grid `linspace(0, N-1, N)`: `RL_Qagent_1D/env.py:50,88` vs `:124`; `RL_posM_env/env.py:83`; `RL_pos_PPO_1D/env.py:92` [Verified].
- B3 The env uses kT=0.5981, but `bias_K_1D` is called without kT and defaults to 0.596: `RL_Qagent_1D/env.py:51`, `util.py:78`. In MATLAB, `bias_K_1D.m:3` hard-codes 0.5981 [Verified].

RL_Qagent_1D / RL_DQN_1D_N100
- B4 `main.py:8` sets N=20, but checkpoints are named `_N100` (`:73,:75`). `train()` reads the global loop variable `i` (`:57-61`) and always writes `model_26thJune*` names [Verified]. That is likely why `models/model_26thJune*.pt` contain N=20 weights (shapes [Verified], cause [Inferred]).
- B5b `RL_Qagent_1D/a2c_agent.py:25-28`: `forward` has no reshape (unlike `dqn_agent.py:20`). An N×N state (`main_a2c.py:8,13`) therefore hits `Linear(N*N, …)` with last dim N, a shape error. The model is saved with `episode+1` in names that also say `26thJune` (`main_a2c.py:48`) [Verified code; Inferred runtime].
- B5 `main.py:78-105` (plotting, `savetxt`) sits inside a string literal and never runs. `simulate()` (`:109-130`) acts ε-greedily and overwrites `0..10.png` ten times [Verified].
- B6 `RL_DQN_1D_N100/util.py:178` calls undefined `top_k_logits`. `mingpt_train.py:28-29` imports `sample` and `atari_py` and defines nothing. `mingpt.py:227` hard-codes Atari frames (`reshape(-1, 4, 84, 84)`). This is the Decision-Transformer Atari minGPT, not adapted [Verified code; Inferred source].

RL_DQN_position
- B7 `main.py:46` calls `env.get_mfpt`, which `position_1D` does not define (`env.py:18-76`) [Verified].
- B8 State is a scalar (`env.py:40`), but the DQN reshapes it to (−1, 100) (`dqn_agent.py:20`, `main.py:11`). That fails in `replay` [Verified code; Inferred runtime].
- B9 `env.py:59-60` samples the same row 1e5 times without advancing. `step()` calls `transition` twice (`:74-75`, plus `:65`), so reward and next state come from different samples [Verified].
- B10 `main.py:52,60` uses a repo-root path while the imports need the folder as cwd [Verified].
- B11 The bias sign is flipped vs Qagent (`util.py:90-91` vs `RL_Qagent_1D/util.py:91-92`). Qagent's sign matches its column-convention K [Inferred].

Offline CQL / data generation
- B12 `RL_offline_CQL/main.py:20`: `actions` is undefined (`:12` is commented out). `:23` calls `CQL(state_dim=…, action_dim=…)`, but the signature is `state_size, action_size` (`cql_agent.py:8`). `:24` calls `nn.Module.train(data, epochs=…)`. `CQLAgent` has no `remember` and is unused (`cql_agent.py:43`) [Verified].
- B13 The CQL penalty sign is reversed: `alpha*(Q_sel − logsumexp)` (`cql_agent.py:69`). float64 batches go into float32 layers (`:53-56`) [Inferred].
- B14 `data_gen.py:19,39` appends the same mutable `action` list to every record of an episode, so the labels are aliased (confirmed in the root `.pt`) [Verified].
- B15 `data_gen.py:11` takes the baseline from the rate-matrix `mfpt_calc`, but `:26` applies the Markov-matrix `markov_mfpt_calc` to a rate matrix. Sampled rewards are ≈0.977–0.999 in **both** dataset copies whatever the action [Verified values; Inferred cause].
- B16 `data_gen.py:2` imports `RL_Qagent_1D.util`, so it must run from the repo root. It sets no RNG seed and writes to cwd (`:47`) [Verified].

RL_posM_env
- B17 `main.py:50` loads the N=20 checkpoint (`fc1M` 128×1600) into the N=100 net (`main.py:8`; `posM_dqn.py:24` hard-codes 40000). That load is a size-mismatch error, reached only after 1000 training episodes (`main.py:37-38`) [Verified shapes; Inferred runtime].
- B18 `env.py:124` references undefined `K`. `test.py:13` (` = schur(A)`) fails `ast.parse` with IndentationError [Verified].
- B19 `posM_dqn.py:196` syncs the target net every step, so there is no target lag. `:211` hard-codes a dated "N20" name for an N=100 net [Verified]. `:38` `pos.unsqueeze(0)` is shape-fragile [Inferred].

RL_pos_PPO_1D
- B20 `ppo_trainer.py:52` (`--num_steps`=500) and `:61` (`--num-steps`=128) share a dest. The first default wins (500, confirmed with stdlib argparse and the tfevents hparams), and the comment at `:88` is wrong [Verified].
- B21 `--gae-lambda` (`:67`) is never used. Advantages are `returns − values` (`:240`) [Verified].
- B22 cwd coupling: `SummaryWriter("runs/…")` (`:159`) and `./RL_pos_PPO_1D/models/` (`:325`) need the repo root as cwd, while `entry_point='env:pos_1D_env'` (`:19`) needs the script dir on sys.path. The generic module name `env` can shadow other modules [Verified code; Inferred risk].
- B23 `local_env_test.py:48` uses `args`, which is defined only under ppo_trainer's `__main__` (`:156`). Line `:47` re-imports ppo_trainer, which re-registers `pos_1D-v0`. Line `:52` loads the 10-Gaussian model (actor 1000×64) while HEAD registers 3 Gaussians (300); `strict=False` does not forgive shape mismatches [Verified code/shapes; Inferred runtime].
- B24 `torch.save` runs only after the full loop (`:325`, added in b4d8ca7). The runs that have models logged only 73/18/20 of ≥2,500 planned updates (`total_timesteps` 1e7 in hparams), so the `.pt` files probably came from uncommitted code [Verified counts; Inferred origin].
- B25 Old gym API: 4-tuple `step` (`:217`, `env.py:128`), obs-only `reset` (`env.py:82`, `ppo_trainer.py:192`), list-of-dict infos (`:221-222`). Also `from distutils.util import strtobool` (`:5`). `make_env` ignores `seed` (`:94-100`). `terminated`/`max_simulation_steps` are set but never used (`env.py:53-55,124-125`) [Verified].

RL_matlab
- M1 `MFPT_env.m:82-99` is a cart-pole `myResetFunction` pasted from the MATLAB docs into the class methods (no `this`) [Verified].
- M2 `MFPT_env.m:116-117` reads `Action(:,1)`/`(:,2)` from a [2,20] action spec (`:65`), so it gets 2 values, not 20 [Verified].
- M3 `MFPT_env.m:121` calls `gaussian_bias(C, std, N)` against the signature `(x, C_g, std_g)` (`gaussian_bias.m:1`) [Verified].
- M4 `MFPT_env.m:168`: `getReward` uses `NextObservation`, which is out of scope there [Verified].
- M5 `myStepFunction.m:55-72`: a missing `end` puts the "third part" penalty inside `else`. `:67` `C<min && C>max` is never true [Verified].
- M6 `propagate_N_steps.m:10` uses undefined `tot_time`, so it errors when the target is reached [Verified].
- M7 `create_K_1D.m:19` already returns K′, yet `myResetFunction.m:20` and `MFPT_env.m:77` transpose it again [Verified].

No Windows or absolute paths appear in RL source (`git grep` for `C:`/`/Users/`); only the `.mat` headers say PCWIN64 [Verified].

## Third-party deps (from the import census of `RL_*/*.py` + `data_gen.py`)

| Package | Used at | Constraint |
|---|---|---|
| torch (+`torch.utils.tensorboard`) | everywhere; `ppo_trainer.py:12` | Checkpoints are zip format v3 (`version`=3) [Verified]. Root `environment.yml` pins pytorch 2.0.1 CPU [Verified] |
| tensorboard | `ppo_trainer.py:12` | **missing from root `environment.yml`** [Verified] |
| gym | `RL_pos_PPO_1D/{env,ppo_trainer,local_env_test}.py` | needs ≥0.22 for `super().reset(seed=)` (`env.py:61`) and <0.26 for the 4-tuple step / obs-only reset; list-of-dict vector infos suggest ≤0.23. Use the range 0.22–0.23 (e.g. 0.23.1) [Inferred]. **Missing from `environment.yml`** [Verified] |
| numpy, scipy (`expm`, `logm`), matplotlib | all | env.yml: numpy 1.24.3, scipy 1.10.1, matplotlib 3.7.1 [Verified] |
| Pillow | `RL_Qagent_1D/main.py:134`, `RL_DQN_1D_N100/main.py:134` | env.yml pillow 9.4.0 [Verified] |
| tqdm, atari_py | `RL_DQN_1D_N100/mingpt_train.py:18,29` (stub only) | atari_py is only needed if the DT stub is revived [Inferred] |
| Python | — | pyc removed in cleanup were cpython-39 **and** cpython-311 [Verified `git status`]. `distutils` requires ≤3.11 [Verified API fact] |

stable-baselines3, wandb and gymnasium are **not imported** by any repo code. They appear only in `RL_PPO_implementation/pyproject.toml:10,12` and as `--wandb-*` flags (`ppo_trainer.py:44-49`) [Verified].
`RL_PPO_implementation/poetry.lock` pins gym 0.21.0, torch 1.10.1, sb3 1.2.0, tensorboard 2.8.0 and python <3.10. Its gym 0.21 would break `env.py:61` [Verified pins; Inferred break].

## Redundancy table

Refs come from `git grep -F <basename>` over the whole repo excluding `docs/`; W = writer, R = reader.

| Path | Size | Evidence | Refs found | Class |
|---|---|---|---|---|
| `RL_Qagent_1D/transformer_agent.py` | 13.9 KB | md5 = `RL_DQN_1D_N100/mingpt.py`; third-party minGPT | none (no import of either) | **DELETE-SAFE** |
| `RL_matlab/explore_the_network.m`, `RL_matlab/Markov_mfpt_calc.m` | 0.6 KB, 0.3 KB | byte-identical to `1D_model_potential_{Edina,Simian}/` | no caller inside `RL_matlab` | **DELETE-SAFE** |
| `RL_CLEANRL/runs/*` (2 runs) | 850 KB | CleanRL tutorial (`cleanRL` wandb project) | none | **DELETE-SAFE** |
| `runs/CartPole-v1__ppo__*` (4), `runs/BreakoutNoFrameskip-v4__ppo_atari__*`, `runs/MicrortsMining-v1__ppo_multidiscrete__*` | 658 KB | tutorial hparams (`cleanRL` / `ppo-implementation-details`) | none | **DELETE-SAFE** |
| `RL_PPO_implementation/` (`pyproject.toml`, `poetry.lock`) | 396 KB | name `ppo_details`, author "Iclr Blog"; no code | none | **DELETE-SAFE** |
| `RL_pos_PPO_1D/runs/ppo_trainer__1__*` (4) | 4 × 673 B | hparams only, no scalars | none | **DELETE-SAFE** |
| `test.png` (root) | 19.6 KB | debug line plot of one Gaussian on [0, 2π]; the exact writer is not identified | W only: `savefig("./test.png")` in 7 scripts (e.g. `1D_langevin_sim/langevin_sim_mfpt_opt.py:72`); no R | **DELETE-SAFE** |
| `RL_matlab/compute_free_energy.m`, `mfpt_calc.m` | <1 KB | byte-identical copies, but called in-folder | R `MFPT_env.m:77-78`, `myResetFunction.m:20` | KEEP |
| `data/data_1D_20N_10Gaussian.pt` | 78.2 MB | only per-step-action dataset; its generator is not in git | R `RL_offline_CQL/main.py:8` | KEEP |
| `pos_M_DQN_model_13July_2_N20.pt` | 0.95 MB | — | R `RL_posM_env/main.py:50` | KEEP |
| `RL_pos_PPO_1D/models/ppo_trainer__1__1690192924.pt` | 330 KB | — | R `local_env_test.py:52` | KEEP |
| `RL_DQN_1D_N100/mingpt.py` | 13.9 KB | canonical copy of the duplicate pair (README's unfinished DT goal) | none | KEEP |
| `pos_M_DQN_model_{11thJuly_1,13July_1,13July_2,13July_2_N100,14July_1_N20}.pt` | 83.7 MB | trained posM checkpoints, all md5-unique | W `posM_dqn.py:211` (14July only); no R | CONFIRM |
| `RL_Qagent_1D/model_26thJune1_{100,200,400}.pt`, `model_29thJune1_N20.pt` | 16.0 MB | N=100 / N=20 DQN weights | W pattern `main.py:57-75`; no R of these names | CONFIRM |
| `RL_Qagent_1D/models/*.pt` (44) | 15.0 MB | historical DQN checkpoints | none at HEAD (`RL_DQN_position/main.py:60` R points to a different path) | CONFIRM |
| `RL_pos_PPO_1D/models/ppo_trainer__1__{1690198587,1690282116}.pt` | 348 KB | own PPO weights (5 and 3 Gaussians) | W `ppo_trainer.py:325`; no R | CONFIRM |
| `runs/ppo_trainer__1__*` (9) | 177 KB | own runs; 3 pair with saved models, 6 aborted | W `ppo_trainer.py:159`; no R | CONFIRM |
| `data_1D_20N_10Gaussian.pt` (root) | 75.2 MB | different from the `data/` copy. It is the only copy committed code reproduces (`data_gen.py` is unseeded). Actions aliased (B14), rewards suspect (B15). Superseded by `data/` [Inferred] (both added in 71ce56f) | W `data_gen.py:47`; no R | CONFIRM |
| `RL_matlab/ReinforcementLearningDesignerSession_{1,TW_1,TW_2}.mat` | 27.3 MB | DDPG / TD3 Designer sessions | none | CONFIRM |
| `RL_Qagent_1D/figs/*` (16), `total_rewards.txt` | 476 KB | result figures and training curve | W `main.py:82,115,124,158`; no R | CONFIRM |
| `figures.pptx` | 3.9 MB | 4-slide deck ("1D", "2D traj1 … Generated bias"), 2023-09-20; about the 2D DHAM work, not RL | none | CONFIRM |
| `fes_visualization.png` (root) | 24.7 KB | `apply_fes(plot=True)` default output, committed "for student meeting" (c93d7f8) | W default `plot_path` in 6 `util.py`, e.g. `1D_langevin_sim/util.py:330` | CONFIRM |

The task asked about other byte-identical Python files: within this area there are **none** besides `mingpt.py`/`transformer_agent.py` (per `duplicate_groups.txt`) [Verified].
Near-duplicates are listed under Lineage.

## Open questions for the authors

1. Which `data_gen.py` variant produced `data/data_1D_20N_10Gaussian.pt`? Should B15 (rate-matrix vs Markov-matrix MFPT) be fixed and the data regenerated?
2. Which posM/PPO checkpoints, if any, back a reported result? Can the rest go? The filename N labels are wrong (B19, 13July_2_N100).
3. How were the PPO `.pt` files saved, given that HEAD saves only after training completes (B24)? Which gym version was used (B25)?
4. Is the decision transformer for 2D (README "unfinished") still planned? If not, `mingpt*` and `util.sample` can go.
5. Are the MATLAB Designer sessions still needed, or were their DDPG/TD3 results captured elsewhere?
6. Is `figures.pptx` (2D DHAM figures) still wanted at the repo root, or should it move to `paper_publication/`?

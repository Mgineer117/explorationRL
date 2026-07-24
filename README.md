# Intrinsic Reward Policy Optimization
[![arXiv](https://img.shields.io/badge/arXiv-2601.21391-b31b1b.svg)](https://arxiv.org/abs/2601.21391)
<img width="1918" height="626" alt="IRPO" src="https://github.com/user-attachments/assets/ac129e10-304a-4b40-8361-8e82844f5f11" />

## Authors
* **Minjae Cho** - _The Grainger College of Engineering, University of Illinois Urbana-Champaign_ (Correspondance)
* **Huy T. Tran** - _The Grainger College of Engineering, University of Illinois Urbana-Champaign_

## Citation
Please cite our paper if you use this code or algorithm for any part of your research or work:
```
@misc{cho2026intrinsicrewardpolicyoptimization,
      title={Intrinsic Reward Policy Optimization for Sparse-Reward Environments},
      author={Minjae Cho and Huy Trong Tran},
      year={2026},
      eprint={2601.21391},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.21391},
}
```

---

This repository runs entirely on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) +
[skrl](https://skrl.readthedocs.io). Everything lives in the Isaac Lab extension
`source/explorationRL/`; there is no separate legacy training pipeline.

## Installation

Requires Isaac Sim / Isaac Lab and skrl >= 2.1 in the active environment:

```bash
conda activate env_isaaclab
python -m pip install -e source/explorationRL
```

## Environments

| task id | kind |
|---|---|
| `PointMaze-v1` | Isaac Lab `DirectRLEnv` — force-actuated point mass in a maze |

`PointMaze-v1` reproduces the gymnasium-robotics MuJoCo `PointMaze` it replaces:
the same force balance (`m·a = gear·u − c·v`, gear 100, damping 1, mass
4.18879 kg), `dt = 0.01`, the pre-step velocity clamp to ±5, the v1 maze layout,
and the sparse goal reward / termination at 0.45 m. `tests/test_pointmaze_env.py`
asserts all of it against the simulator (needs Isaac Sim):

```bash
python tests/test_pointmaze_env.py
```

## Algorithms

Selected with `--algorithm`; each has a config in
`source/explorationRL/explorationRL/tasks/direct/pointmaze/agents/`.

| `--algorithm` | base | what it adds |
|---|---|---|
| `ppo`, `sac`, `trpo` | — | stock skrl baselines |
| `drnd` | PPO | Distributional RND novelty bonus folded into the reward |
| `psne` | TRPO | adaptive parameter-space noise (Plappert et al.) — acts through a perturbed policy copy |
| `htrpo` | TRPO | hindsight goal relabelling (HGF) + eq.-79 weighted importance sampling |
| `irpo` | PPO | **flagship** — persistent per-option policies driven by ALLO Laplacian eigenvector intrinsic rewards |
| `hrl` | PPO | categorical controller sequencing those options, semi-MDP execution |
| `maml` | PPO | FOMAML inner/outer loop over a support/query env split |

`irpo` and `hrl` share the representation stack in
`explorationRL/extractors/` (ALLO, an augmented-Lagrangian Laplacian objective)
and `explorationRL/agents/skrl/intrinsic.py` (signed eigenvector-direction
rewards). Each agent's module docstring states precisely how it maps onto the
original formulation, and where it deliberately differs.

## Train

```bash
python scripts/skrl/train.py --task PointMaze-v1 --algorithm irpo
python scripts/skrl/train.py --task PointMaze-v1 --algorithm sac --num_envs 64
python scripts/skrl/play.py  --task PointMaze-v1 --checkpoint <ckpt> --headon
```

Isaac Sim runs headless by default; `--headon` renders a window. Every run logs
to W&B unless `--no_wandb`.

> `irpo` and `hrl` partition the parallel envs across their option policies, so
> they need at least `num_options + 1` envs (they raise with a clear message
> otherwise).

## Multi-seed runs — `./run.sh`

Interactive launcher: asks **local vs SLURM cluster**, then algorithm, env, seed
count, sequential-vs-parallel concurrency, and GPUs (or partition / GPUs-per-job
/ wall-time / account on the cluster). Local runs are handed to `./run_seeds.sh`
(detached via `nohup`); cluster runs generate and submit a self-contained sbatch
job. Both stamp every run with the same `EXP_RUN_TAG` and aggregate into:

```
results/<tag>_runs.csv   one row per seed
results/<tag>.csv        one row per (env, algorithm): mean/std/95% CI
```

## Hyperparameter search — `./search/search.sh`

Interactive W&B sweep launcher: asks **local vs cluster**, algorithm, env, and
how many self-restarting search agents (plus partition / GPUs-per-job /
agents-per-GPU on the cluster). Local spawns detached `wandb agent` workers;
cluster creates the sweep once on the login node and submits self-resubmitting
sbatch workers that renew before the wall-time kill (`--stop <tag>` halts them).

The searched space is **not** in the script — it lives in
[`search/configs/`](search/configs/README.md), one yaml per algorithm declaring
the parameters, ranges, objective metric and search method.

```bash
./search/search.sh     # fully interactive
./run.sh               # fully interactive
```

## Layout

```
source/explorationRL/      Isaac Lab extension (envs, agents, extractors)
scripts/skrl/              train.py / play.py / train_utils.py
scripts/aggregate_seeds.py per-seed CSV -> mean/std/CI
search/                    sweep configs + build_sweep.py + search.sh
run.sh, run_seeds.sh       multi-seed launchers
tests/                     PointMaze env correctness checks
model/                     pretrained ALLO option checkpoints
```

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

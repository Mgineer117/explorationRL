# explorationRL — Isaac Lab extension

Exploration-driven reinforcement learning on a unified [skrl](https://skrl.readthedocs.io)
backend, for [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) and lightweight /
discrete environments.

This package hosts:

* **Environments** — `tasks/direct/`, e.g. `PointMaze-v1`, a true Isaac Lab
  `DirectRLEnv` whose point-mass dynamics are matched to the gymnasium-robotics
  MuJoCo `PointMaze`.
* **Agents** — `agents/skrl/`: the research algorithms DRND, PSNE, HTRPO, IRPO,
  HRL and MAML, alongside skrl's stock PPO / TRPO / SAC.
* **Extractors** — `extractors/`: ALLO, an augmented-Lagrangian Laplacian
  representation learner whose eigenvector directions define the intrinsic
  rewards IRPO and HRL explore along.
* **Runner** — `agents/skrl/runner.py` (`ExplorationRunner`), a thin
  `skrl.utils.runner.torch.Runner` subclass that resolves custom agent classes
  and model factories while falling back to skrl's stock components.

Train from the repository root:

```bash
python scripts/skrl/train.py --task PointMaze-v1 --algorithm ppo
```

See the repo-root `run.sh` (multi-seed launcher) and `search/search.sh`
(W&B hyperparameter sweep launcher).

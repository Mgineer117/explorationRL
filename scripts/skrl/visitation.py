"""Progressive state-visitation maps: roll out each training checkpoint of one or
more runs and plot where the policy actually goes in the maze.

Each checkpoint is a snapshot of the policy at that point in training, so the
row of heatmaps reads left-to-right as "how the occupancy of the learned policy
evolved" — the direct check on whether AGA's preconditioning buys exploration
over plain PPO.

Positions come from the agent state: ``obs[:, :2]`` is the ball's local ``(x, y)``
(see ``pointmaze_env._get_observations``).

Example
-------
    python scripts/skrl/visitation.py --task PointMaze-v1 \
        --run "PPO=logs/skrl/pointmaze/<ppo_run>" \
        --run "AGA+DRND=logs/skrl/pointmaze/<aga_run>" \
        --out logs/skrl/visitation.png

Writes the figure and a ``.npz`` of the raw histograms next to it.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Plot progressive visitation maps from checkpoints.")
parser.add_argument("--task", type=str, default="PointMaze-v1")
parser.add_argument("--run", action="append", default=[], metavar="LABEL=RUN_DIR",
                    help="Training run to map; repeatable. LABEL is the plot row title.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=1, help="Episodes per checkpoint.")
parser.add_argument("--bins_per_cell", type=int, default=10)
parser.add_argument("--max_columns", type=int, default=6, help="Checkpoints plotted per row.")
parser.add_argument("--traj_envs", type=int, default=12,
                    help="Envs whose full xy path is kept and drawn on the trajectory figure.")
parser.add_argument("--out", type=str, default="logs/skrl/visitation.png")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import glob  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402

import explorationRL.tasks  # noqa: F401,E402
from explorationRL.agents.skrl.runner import ExplorationRunner  # noqa: E402
from explorationRL.tasks.direct.pointmaze.pointmaze_env_cfg import (  # noqa: E402
    CELL_SIZE, POINTMAZE_V1_MAP, PointMazeEnvCfg,
)

ROWS, COLS = len(POINTMAZE_V1_MAP), len(POINTMAZE_V1_MAP[0])
X_MAX, Y_MAX = COLS / 2.0 * CELL_SIZE, ROWS / 2.0 * CELL_SIZE
EXTENT = (-X_MAX, X_MAX, -Y_MAX, Y_MAX)


def _checkpoints(run_dir: str) -> list[tuple[int, str]]:
    """``[(timestep, path), ...]`` sorted by timestep (``best_agent.pt`` skipped)."""
    out = []
    for path in glob.glob(os.path.join(run_dir, "checkpoints", "agent_*.pt")):
        m = re.search(r"agent_(\d+)\.pt$", path)
        if m:
            out.append((int(m.group(1)), path))
    return sorted(out)


def _thin(items: list, k: int) -> list:
    """At most ``k`` items, evenly spaced, always keeping the first and last."""
    if len(items) <= k:
        return items
    idx = np.unique(np.linspace(0, len(items) - 1, k).round().astype(int))
    return [items[i] for i in idx]


@torch.no_grad()
def rollout_histogram(env, agent, steps: int, bins: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Roll out ``agent`` once; return its occupancy histogram and the raw
    ``(steps, traj_envs, 2)`` xy paths of the first few envs."""
    counts = np.zeros(bins, dtype=np.float64)
    observations, _ = env.reset()
    states = env.state()
    n_traj = min(args_cli.traj_envs, observations.shape[0])
    paths = np.empty((steps, n_traj, 2), dtype=np.float32)
    for t in range(steps):
        xy = observations[:, :2].detach().cpu().numpy()
        paths[t] = xy[:n_traj]
        h, _, _ = np.histogram2d(
            xy[:, 1], xy[:, 0], bins=bins,  # (row=y, col=x)
            range=[[-Y_MAX, Y_MAX], [-X_MAX, X_MAX]],
        )
        counts += h
        # Actions are sampled, not the distribution mean: the question is where the
        # policy *explores*, and the mean collapses the map to a single path.
        actions, _ = agent.act(observations, states, timestep=t, timesteps=steps)
        observations, _, _, _, _ = env.step(actions)
        states = env.state()
    return counts, paths


def build_agent(env, run_dir: str):
    """Rebuild the run's agent from its dumped ``params/agent.yaml``."""
    with open(os.path.join(run_dir, "params", "agent.yaml")) as f:
        agent_cfg = yaml.unsafe_load(f)
    experiment = agent_cfg["agent"].setdefault("experiment", {})
    experiment.update({"wandb": False, "write_interval": 0, "checkpoint_interval": 0})
    runner = ExplorationRunner(env, agent_cfg)
    return runner.agent


def maze_overlay(ax) -> None:
    """Draw walls, start ('r') and goal ('g') on top of a heatmap."""
    for i in range(ROWS):
        for j in range(COLS):
            cell = POINTMAZE_V1_MAP[i][j]
            x = -X_MAX + j * CELL_SIZE
            y = Y_MAX - (i + 1) * CELL_SIZE
            if cell == 1:
                ax.add_patch(plt.Rectangle((x, y), CELL_SIZE, CELL_SIZE,
                                           facecolor="0.25", edgecolor="none", zorder=3))
            elif cell in ("r", "g"):
                ax.plot(x + CELL_SIZE / 2, y + CELL_SIZE / 2,
                        marker="o" if cell == "r" else "*",
                        color="white" if cell == "r" else "gold",
                        markersize=7 if cell == "r" else 12,
                        markeredgecolor="black", markeredgewidth=0.6, zorder=4)


def main() -> None:
    global reachable_bins
    runs = []
    for spec in args_cli.run:
        label, _, run_dir = spec.partition("=")
        if not run_dir:
            raise SystemExit(f"--run expects LABEL=RUN_DIR, got {spec!r}")
        ckpts = _thin(_checkpoints(run_dir), args_cli.max_columns)
        if not ckpts:
            raise SystemExit(f"No agent_*.pt checkpoints under {run_dir}/checkpoints/")
        runs.append((label, run_dir, ckpts))
    if not runs:
        raise SystemExit("Pass at least one --run LABEL=RUN_DIR")

    env_cfg = PointMazeEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    env = gym.make(args_cli.task, cfg=env_cfg)
    max_len = int(getattr(env.unwrapped, "max_episode_length", 500))
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    bins = (ROWS * args_cli.bins_per_cell, COLS * args_cli.bins_per_cell)
    # Coverage is quoted against the free cells only — the walls are not reachable,
    # so counting them in the denominator would flatter every policy equally.
    open_cells = sum(c != 1 for row in POINTMAZE_V1_MAP for c in row)
    reachable_bins = open_cells * args_cli.bins_per_cell**2
    steps = max_len * max(1, args_cli.episodes)

    maps: dict[str, list[np.ndarray]] = {}
    paths: dict[str, list[np.ndarray]] = {}
    for label, run_dir, ckpts in runs:
        agent = build_agent(env, run_dir)
        if hasattr(agent, "enable_training_mode"):
            agent.enable_training_mode(False)
        elif hasattr(agent, "set_running_mode"):
            agent.set_running_mode("eval")
        maps[label], paths[label] = [], []
        for timestep, path in ckpts:
            agent.load(path)
            counts, xy_paths = rollout_histogram(env, agent, steps, bins)
            visited = int((counts > 0).sum())
            print(f"[INFO] {label} @ {timestep:>7d} steps: {visited} / {reachable_bins} "
                  f"reachable bins visited ({100 * visited / reachable_bins:.1f}%)", flush=True)
            maps[label].append(counts)
            paths[label].append(xy_paths)
    env.close()

    # ── plot ──────────────────────────────────────────────────────────────── #
    n_rows, n_cols = len(runs), max(len(c) for _, _, c in runs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.4 * n_rows),
                             squeeze=False, constrained_layout=True)
    for r, (label, _, ckpts) in enumerate(runs):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.set_xticks([]), ax.set_yticks([])
            if c >= len(ckpts):
                ax.axis("off")
                continue
            counts = maps[label][c]
            # log1p: occupancy is heavily peaked at the start cell, and the
            # question here is *coverage*, not how long it lingered.
            # origin="lower": histogram row 0 is the *lowest* y bin, so it belongs
            # at the bottom of the axes — "upper" mirrors the occupancy against the
            # (data-coordinate, hence unflipped) maze overlay.
            ax.imshow(np.log1p(counts), origin="lower", extent=EXTENT,
                      cmap="magma", interpolation="nearest")
            maze_overlay(ax)
            # Coverage of the *reachable* area — the headline number, on the panel.
            ax.text(0.03, 0.04, f"{100 * (counts > 0).sum() / reachable_bins:.0f}% covered",
                    transform=ax.transAxes, fontsize=8, color="white", zorder=5)
            if r == 0:
                ax.set_title(f"{ckpts[c][0] // 1000}k steps", fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=11)
    fig.suptitle("Progressive state visitation (policy rollouts at each checkpoint)", fontsize=12)

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    fig.savefig(args_cli.out, dpi=150)

    # ── trajectories ──────────────────────────────────────────────────────── #
    # Same grid, but the actual paths: one line per env, colored by time within
    # the episode, so a policy that goes somewhere and a policy that circles in
    # place are distinguishable (the heatmap alone cannot tell them apart).
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.4 * n_rows),
                               squeeze=False, constrained_layout=True)
    for r, (label, _, ckpts) in enumerate(runs):
        for c in range(n_cols):
            ax = axes2[r][c]
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_xlim(-X_MAX, X_MAX), ax.set_ylim(-Y_MAX, Y_MAX)
            if c >= len(ckpts):
                ax.axis("off")
                continue
            ax.set_facecolor("black")
            xy = paths[label][c]  # (steps, envs, 2)
            time_color = plt.cm.viridis(np.linspace(0, 1, xy.shape[0] - 1))
            for e in range(xy.shape[1]):
                seg_x, seg_y = xy[:, e, 0], xy[:, e, 1]
                ax.scatter(seg_x[:-1], seg_y[:-1], c=time_color, s=0.7, linewidths=0, zorder=2)
                ax.plot(seg_x[0], seg_y[0], marker="o", color="white", markersize=2.5, zorder=4)
            maze_overlay(ax)
            if r == 0:
                ax.set_title(f"{ckpts[c][0] // 1000}k steps", fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=11)
    fig2.suptitle("Trajectories over one episode (dark = episode start, bright = end)", fontsize=12)
    traj_out = os.path.splitext(args_cli.out)[0] + "_trajectories.png"
    fig2.savefig(traj_out, dpi=150)

    npz = os.path.splitext(args_cli.out)[0] + ".npz"
    arrays = {f"counts_{label}_{t}": m for label, _, ckpts in runs
              for (t, _), m in zip(ckpts, maps[label])}
    arrays.update({f"xy_{label}_{t}": p for label, _, ckpts in runs
                   for (t, _), p in zip(ckpts, paths[label])})
    np.savez_compressed(npz, **arrays)
    print(f"[INFO] Wrote {args_cli.out}, {traj_out} and {npz}")


if __name__ == "__main__":
    main()
    simulation_app.close()
    os._exit(0)

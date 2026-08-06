"""ALLOGoalKernelRewards potential Φ(s)=K(φ(s),φ(g)) for several goals, across
every kernel mode, on the hard serpentine maze (see hard_maze_intrinsic.py).

No Isaac Sim / GPU needed: trains ALLO the same way hard_maze_intrinsic.py
does (a vectorized reject-if-wall random walk, not a real physics rollout),
then reuses that representation for every (goal, kernel mode) pair.

Example
-------
    python scripts/skrl/hard_maze_kernel_reward.py --out logs/skrl/hard_maze_kernel_reward.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from explorationRL.agents.skrl.intrinsic import _apply_kernel
from explorationRL.extractors import ALLO, ALLO_CFG

# Same 11x11 serpentine maze as hard_maze_intrinsic.py.
HARD_MAZE_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, "r", 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, "g", 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]
CELL_SIZE = 1.0

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--feature_dim", type=int, default=10, help="Useful ALLO dims (index 0 always dropped).")
parser.add_argument("--walkers", type=int, default=256)
parser.add_argument("--steps", type=int, default=3000, help="Random-walk buffer length per walker.")
parser.add_argument("--episode_len", type=int, default=200)
parser.add_argument("--step_size", type=float, default=0.3, help="Random-walk step size, in cells.")
parser.add_argument("--pretrain_epochs", type=int, default=5000)
parser.add_argument("--goal_cells", type=int, nargs="+", default=None,
                    help="Flat list of maze (row, col) pairs, e.g. 9 9 1 1. Default: the maze's own "
                         "goal cell, the start cell, and two far corners.")
parser.add_argument("--kernel_modes", type=str, nargs="+",
                    default=["rbf", "laplacian", "cauchy", "cosine"])
parser.add_argument("--bins_per_cell", type=int, default=20)
parser.add_argument("--out", type=str, default="logs/skrl/hard_maze_kernel_reward.png")
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

ROWS, COLS = len(HARD_MAZE_MAP), len(HARD_MAZE_MAP[0])
X_MAX, Y_MAX = COLS / 2.0 * CELL_SIZE, ROWS / 2.0 * CELL_SIZE
EXTENT = (-X_MAX, X_MAX, -Y_MAX, Y_MAX)
WALL_GRID = torch.tensor([[HARD_MAZE_MAP[i][j] == 1 for j in range(COLS)] for i in range(ROWS)])
FREE_CELLS = torch.tensor([
    (-X_MAX + (j + 0.5) * CELL_SIZE, Y_MAX - (i + 0.5) * CELL_SIZE)
    for i in range(ROWS) for j in range(COLS) if HARD_MAZE_MAP[i][j] != 1
], dtype=torch.float32)


def _cell_center(i: int, j: int) -> tuple[float, float]:
    return -X_MAX + (j + 0.5) * CELL_SIZE, Y_MAX - (i + 0.5) * CELL_SIZE


def _find(char: str) -> tuple[int, int]:
    for i in range(ROWS):
        for j in range(COLS):
            if HARD_MAZE_MAP[i][j] == char:
                return i, j
    raise ValueError(f"maze_map has no {char!r} cell")


def default_goal_cells() -> list[tuple[int, int]]:
    free = [(i, j) for i in range(ROWS) for j in range(COLS) if HARD_MAZE_MAP[i][j] != 1]
    goal, start = _find("g"), _find("r")
    far1 = max(free, key=lambda c: c[0] + c[1])
    far2 = max(free, key=lambda c: c[0] - c[1])
    cells, seen, unique = [goal, start, far1, far2], set(), []
    for c in cells:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def cell_of(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    j = torch.floor((pos[:, 0] + X_MAX) / CELL_SIZE).long().clamp(0, COLS - 1)
    i = torch.floor((Y_MAX - pos[:, 1]) / CELL_SIZE).long().clamp(0, ROWS - 1)
    return i, j


def sample_uniform(n: int, device) -> torch.Tensor:
    idx = torch.randint(FREE_CELLS.shape[0], (n,), device=device)
    return FREE_CELLS.to(device)[idx] + (torch.rand(n, 2, device=device) * 2 - 1) * (CELL_SIZE * 0.3)


@torch.no_grad()
def random_walk_buffer(n: int, steps: int, episode_len: int, step_size: float, device):
    wall = WALL_GRID.to(device)
    pos = sample_uniform(n, device)
    obs = torch.empty(steps, n, 2, device=device)
    dones = torch.zeros(steps, n, dtype=torch.bool, device=device)
    for t in range(steps):
        obs[t] = pos
        proposal = pos + (torch.rand(n, 2, device=device) * 2 - 1) * (step_size * CELL_SIZE)
        i, j = cell_of(proposal)
        accept = ~wall[i, j]
        pos = torch.where(accept.unsqueeze(-1), proposal, pos)
        if (t + 1) % episode_len == 0:
            dones[t] = True
            pos = sample_uniform(n, device)
    return obs, dones


def build_grid(bins_per_cell: int):
    R, C = ROWS * bins_per_cell, COLS * bins_per_cell
    cell_w, cell_h = (2 * X_MAX) / C, (2 * Y_MAX) / R
    ii, jj = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
    x = -X_MAX + (jj + 0.5) * cell_w
    y = Y_MAX - (ii + 0.5) * cell_h
    wall_mask = np.array([[HARD_MAZE_MAP[i // bins_per_cell][j // bins_per_cell] == 1
                           for j in range(C)] for i in range(R)])
    obs = torch.stack([torch.from_numpy(x.reshape(-1)), torch.from_numpy(y.reshape(-1))], dim=-1).float()
    return obs, wall_mask, (R, C)


def maze_overlay(ax) -> None:
    rx, ry = _cell_center(*_find("r"))
    ax.plot(rx, ry, "o", color="white", markersize=6, markeredgecolor="black", markeredgewidth=0.6, zorder=4)


def mark_goal(ax, gx: float, gy: float) -> None:
    ax.plot(gx, gy, "P", color="red", markersize=11, markeredgecolor="black", markeredgewidth=0.8, zorder=5)


def phi(extractor, observations: torch.Tensor) -> torch.Tensor:
    return extractor(observations)[:, 1:]  # drop the trivial constant eigenfunction


def main() -> None:
    device = args.device
    print(f"[INFO] Collecting a {args.steps}x{args.walkers} random-walk buffer on the hard maze.")
    obs, dones = random_walk_buffer(args.walkers, args.steps, args.episode_len, args.step_size, device)

    internal_dim = args.feature_dim + 1  # +1: reserved index 0 (see aga.py / pretrain_allo.py convention)
    extractor = ALLO(observation_dim=2, cfg=ALLO_CFG(feature_dim=internal_dim, positional_indices=None),
                     device=device)
    print(f"[INFO] Training ALLO ({args.feature_dim} useful dims + 1 reserved) "
          f"for {args.pretrain_epochs} epochs.")
    log_every = max(1, args.pretrain_epochs // 10)
    for epoch in range(args.pretrain_epochs):
        metrics = extractor.learn(obs, dones)
        if epoch % log_every == 0 or epoch == args.pretrain_epochs - 1:
            print(f"[INFO]   epoch {epoch:5d}  loss={metrics['ALLO / loss']:.4f}  "
                  f"graph={metrics['ALLO / graph loss']:.4f}  dual_norm={metrics['ALLO / dual norm']:.4f}")
    extractor.eval()

    grid_obs, wall_mask, (R, C) = build_grid(args.bins_per_cell)
    with torch.no_grad():
        phi_grid = phi(extractor, grid_obs)

    goal_cells = (
        [(args.goal_cells[k], args.goal_cells[k + 1]) for k in range(0, len(args.goal_cells), 2)]
        if args.goal_cells else default_goal_cells()
    )
    modes = args.kernel_modes

    fig, axes = plt.subplots(len(goal_cells), len(modes),
                             figsize=(3.4 * len(modes), 3.0 * len(goal_cells)), squeeze=False)
    for row, (gi, gj) in enumerate(goal_cells):
        gx, gy = _cell_center(gi, gj)
        goal_state = torch.tensor([[gx, gy]], device=device)
        with torch.no_grad():
            phi_g = phi(extractor, goal_state)

        for col, mode in enumerate(modes):
            with torch.no_grad():
                k = _apply_kernel(phi_grid, phi_g.expand_as(phi_grid), mode).squeeze(-1).numpy()
            field = np.where(wall_mask.reshape(-1), np.nan, k).reshape(R, C)

            ax = axes[row][col]
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad("0.25")
            im = ax.imshow(field, extent=EXTENT, cmap=cmap, vmin=0.0, vmax=1.0)
            maze_overlay(ax)
            mark_goal(ax, gx, gy)
            ax.set_title(f"goal=({gi},{gj})  {mode}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("ALLOGoalKernelRewards potential Φ(s)=K(φ(s),φ(g)) — hard serpentine maze", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"[INFO] Wrote {args.out}")


if __name__ == "__main__":
    main()

"""Visualize ALLOGoalKernelRewards's potential Φ(s) = K(φ(s), φ(g)) for several
goal cells, across every kernel mode (rbf, laplacian, cauchy, cosine).

Reuses the pretrained ALLO checkpoint from pretrain_allo.py exactly as
IntrinsicRewardWrapper(int_reward="allo_kernel") would: φ(g) is computed by
embedding "goal-as-state" (a state vector with the position dims overwritten
by the goal xy) through the same extractor, then φ(s) for every grid cell is
compared to it via _apply_kernel. Rows = goal cells, columns = kernel modes,
so a row shows how differently each kernel shapes the same goal's field, and
a column shows how one kernel's field moves as the goal moves.

Example
-------
    python scripts/skrl/plot_kernel_reward.py --task PointMaze-v1 \
        --allo_checkpoint model/allo/PointMaze-v1_dim10.pth --allo_feature_dim 11 \
        --out logs/skrl/kernel_reward_maps.png
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="PointMaze-v1")
parser.add_argument("--allo_checkpoint", type=str, default="model/allo/PointMaze-v1_dim10.pth")
parser.add_argument("--allo_feature_dim", type=int, default=11,
                    help="Checkpoint's raw ALLO width, i.e. useful dims + 1.")
parser.add_argument("--allo_hidden_dim", type=int, nargs="+", default=[256, 256])
parser.add_argument("--allo_positional_indices", type=int, nargs="+", default=[0, 1])
parser.add_argument("--goal_cells", type=int, nargs="+", default=None,
                    help="Flat list of maze (row, col) pairs to use as goals, e.g. "
                         "1 3 3 1 3 4 (three goals). Default: the env's own goal cell, "
                         "the start cell, and two far corners of the maze.")
parser.add_argument("--kernel_modes", type=str, nargs="+",
                    default=["rbf", "laplacian", "cauchy", "cosine"])
parser.add_argument("--bins_per_cell", type=int, default=20, help="Plot grid resolution per maze cell.")
parser.add_argument("--out", type=str, default="logs/skrl/kernel_reward_maps.png")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── Post-app imports ─────────────────────────────────────────────────────── #
import os  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explorationRL.agents.skrl.intrinsic import _apply_kernel  # noqa: E402
from explorationRL.extractors import ALLO, ALLO_CFG  # noqa: E402
from explorationRL.tasks.direct.pointmaze.pointmaze_env_cfg import (  # noqa: E402
    CELL_SIZE, POINTMAZE_V1_MAP,
)

ROWS, COLS = len(POINTMAZE_V1_MAP), len(POINTMAZE_V1_MAP[0])
X_MAX, Y_MAX = COLS / 2.0 * CELL_SIZE, ROWS / 2.0 * CELL_SIZE
EXTENT = (-X_MAX, X_MAX, -Y_MAX, Y_MAX)


def _cell_center(i: int, j: int) -> tuple[float, float]:
    return -X_MAX + (j + 0.5) * CELL_SIZE, Y_MAX - (i + 0.5) * CELL_SIZE


def _find(char: str) -> tuple[int, int]:
    for i in range(ROWS):
        for j in range(COLS):
            if POINTMAZE_V1_MAP[i][j] == char:
                return i, j
    raise ValueError(f"maze_map has no {char!r} cell")


def default_goal_cells() -> list[tuple[int, int]]:
    free = [(i, j) for i in range(ROWS) for j in range(COLS) if POINTMAZE_V1_MAP[i][j] != 1]
    goal, start = _find("g"), _find("r")
    far1 = max(free, key=lambda c: c[0] + c[1])  # bottom-right-most free cell
    far2 = max(free, key=lambda c: c[0] - c[1])  # bottom-left-most free cell
    cells = [goal, start, far1, far2]
    seen, unique = set(), []
    for c in cells:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def build_grid(bins_per_cell: int):
    """``(obs, wall_mask)`` for an ``(R, C)`` pixel grid over the whole maze."""
    R, C = ROWS * bins_per_cell, COLS * bins_per_cell
    cell_w, cell_h = (2 * X_MAX) / C, (2 * Y_MAX) / R
    ii, jj = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
    x = -X_MAX + (jj + 0.5) * cell_w
    y = Y_MAX - (ii + 0.5) * cell_h
    wall_mask = np.array([[POINTMAZE_V1_MAP[i // bins_per_cell][j // bins_per_cell] == 1
                           for j in range(C)] for i in range(R)])
    obs = torch.zeros(R * C, 8)
    obs[:, 0] = torch.from_numpy(x.reshape(-1))
    obs[:, 1] = torch.from_numpy(y.reshape(-1))
    obs[:, 4] = obs[:, 0]
    obs[:, 5] = obs[:, 1]
    return obs, wall_mask, (R, C)


def maze_overlay(ax) -> None:
    rx, ry = _cell_center(*_find("r"))
    ax.plot(rx, ry, "o", color="white", markersize=6, markeredgecolor="black", markeredgewidth=0.6, zorder=4)


def mark_goal(ax, gx: float, gy: float) -> None:
    ax.plot(gx, gy, "P", color="red", markersize=11, markeredgecolor="black", markeredgewidth=0.8, zorder=5)


def phi(extractor, observations: torch.Tensor) -> torch.Tensor:
    return extractor(observations)[:, 1:]  # drop the trivial constant eigenfunction


def main() -> None:
    device = "cpu"
    extractor = ALLO(
        observation_dim=8,
        cfg=ALLO_CFG(
            feature_dim=args_cli.allo_feature_dim,
            hidden_dim=list(args_cli.allo_hidden_dim),
            positional_indices=list(args_cli.allo_positional_indices),
        ),
        device=device,
    )
    extractor.load_state_dict(torch.load(args_cli.allo_checkpoint, map_location=device))
    extractor.eval()

    grid_obs, wall_mask, (R, C) = build_grid(args_cli.bins_per_cell)
    with torch.no_grad():
        phi_grid = phi(extractor, grid_obs)  # (R*C, d)

    goal_cells = (
        [(args_cli.goal_cells[k], args_cli.goal_cells[k + 1]) for k in range(0, len(args_cli.goal_cells), 2)]
        if args_cli.goal_cells else default_goal_cells()
    )
    modes = args_cli.kernel_modes

    fig, axes = plt.subplots(len(goal_cells), len(modes),
                             figsize=(3.4 * len(modes), 3.0 * len(goal_cells)), squeeze=False)
    for row, (gi, gj) in enumerate(goal_cells):
        gx, gy = _cell_center(gi, gj)
        goal_state = torch.zeros(1, 8)
        goal_state[0, 0], goal_state[0, 1] = gx, gy
        with torch.no_grad():
            phi_g = phi(extractor, goal_state)  # (1, d)

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

    fig.suptitle(f"ALLOGoalKernelRewards potential Φ(s)=K(φ(s),φ(g)) — {args_cli.task}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(args_cli.out) or ".", exist_ok=True)
    fig.savefig(args_cli.out, dpi=150)
    print(f"[INFO] Wrote {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()

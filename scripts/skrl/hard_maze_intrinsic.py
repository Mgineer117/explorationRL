"""ALLO + geodesic-guidance intrinsic reward fields on a harder, hand-built maze.

A longer serpentine layout (11x11, BFS start->goal distance 48 cells vs. the
6x5 PointMaze-v1's ~7) used purely to see how these two intrinsic signals
structure a maze with many more turns and a much longer single corridor.

No Isaac Sim / GPU physics needed: both signals only depend on (x, y)
transitions, not realistic dynamics, so the representation-learning buffer
here is a plain vectorized random walk (reject-if-wall) over the maze graph
instead of a real physics rollout — the same trick ``pretrain_allo.py`` uses,
minus the simulator. GeodesicGuidance/ALLO themselves are unchanged imports
from the real pipeline.

Example
-------
    python scripts/skrl/hard_maze_intrinsic.py --out logs/skrl/hard_maze_intrinsic.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from explorationRL.agents.skrl.intrinsic import GeodesicGuidance
from explorationRL.extractors import ALLO, ALLO_CFG

# ── a harder maze: 11x11 serpentine, 4 full corridor traversals ───────────── #
# 1 = wall, 0 = free, "r" = start, "g" = goal. BFS start->goal distance: 48
# cells (vs. ~7 for POINTMAZE_V1_MAP) — the path must run the full width of
# the maze five times, reversing direction at alternating ends each time.
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
parser.add_argument("--feature_dim", type=int, default=10)
parser.add_argument("--walkers", type=int, default=256)
parser.add_argument("--steps", type=int, default=3000, help="Random-walk buffer length per walker.")
parser.add_argument("--episode_len", type=int, default=200,
                    help="Walkers reset to a fresh uniform free cell every this many steps.")
parser.add_argument("--step_size", type=float, default=0.3, help="Random-walk step size, in cells.")
parser.add_argument("--pretrain_epochs", type=int, default=5000)
parser.add_argument("--bins_per_cell", type=int, default=20, help="Plot grid resolution per maze cell.")
parser.add_argument("--out", type=str, default="logs/skrl/hard_maze_intrinsic.png")
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


def _find(char: str) -> tuple[float, float]:
    for i in range(ROWS):
        for j in range(COLS):
            if HARD_MAZE_MAP[i][j] == char:
                return _cell_center(i, j)
    raise ValueError(f"maze_map has no {char!r} cell")


def cell_of(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    j = torch.floor((pos[:, 0] + X_MAX) / CELL_SIZE).long().clamp(0, COLS - 1)
    i = torch.floor((Y_MAX - pos[:, 1]) / CELL_SIZE).long().clamp(0, ROWS - 1)
    return i, j


def sample_uniform(n: int, device) -> torch.Tensor:
    idx = torch.randint(FREE_CELLS.shape[0], (n,), device=device)
    return FREE_CELLS.to(device)[idx] + (torch.rand(n, 2, device=device) * 2 - 1) * (CELL_SIZE * 0.3)


@torch.no_grad()
def random_walk_buffer(n: int, steps: int, episode_len: int, step_size: float, device):
    """``(obs, dones)`` of shape ``(steps, n, 2)`` / ``(steps, n)`` from a
    reject-if-wall random walk: propose a random step, keep it only if the
    destination cell is free."""
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
    """``(obs, wall_mask)`` for an ``(R, C)`` pixel grid over the whole maze."""
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
    rx, ry = _find("r")
    gx, gy = _find("g")
    ax.plot(rx, ry, "o", color="white", markersize=6, markeredgecolor="black", markeredgewidth=0.6, zorder=4)
    ax.plot(gx, gy, "*", color="gold", markersize=10, markeredgecolor="black", markeredgewidth=0.6, zorder=4)


def plot_field(ax, field: np.ndarray, wall_mask: np.ndarray, title: str, *, diverging: bool) -> None:
    field = np.where(wall_mask, np.nan, field)
    cmap = plt.get_cmap("RdBu_r" if diverging else "viridis").copy()
    cmap.set_bad("0.25")
    if diverging:
        vmax = np.nanmax(np.abs(field)) or 1.0
        im = ax.imshow(field, extent=EXTENT, cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(field, extent=EXTENT, cmap=cmap)
    maze_overlay(ax)

    # Highest-reward point (argmax over non-wall cells).
    R, C = field.shape
    row, col = np.unravel_index(np.nanargmax(field), field.shape)
    x = -X_MAX + (col + 0.5) * (2 * X_MAX) / C
    y = Y_MAX - (row + 0.5) * (2 * Y_MAX) / R
    ax.plot(x, y, marker="X", markersize=9, color="lime",
            markeredgecolor="black", markeredgewidth=0.8, zorder=5)

    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def argmax_xy(field: np.ndarray) -> tuple[float, float]:
    R, C = field.shape
    row, col = np.unravel_index(np.nanargmax(field), field.shape)
    x = -X_MAX + (col + 0.5) * (2 * X_MAX) / C
    y = Y_MAX - (row + 0.5) * (2 * Y_MAX) / R
    return x, y


def plot_argmax_summary(panels: list[tuple[str, np.ndarray, bool, str]], wall_mask: np.ndarray,
                        out: str, suptitle: str) -> None:
    """One map with every option's highest-reward point, labelled by name/sign."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(np.where(wall_mask, 0.0, 1.0), extent=EXTENT, cmap="gray", vmin=0, vmax=1)
    maze_overlay(ax)
    colors = plt.cm.hsv(np.linspace(0, 1, len(panels), endpoint=False))
    for (_, field, _, label), color in zip(panels, colors):
        # Mask walls before arg-maxing: an untrained-region extrapolation spike
        # under a wall would otherwise win over every real reachable-cell value.
        x, y = argmax_xy(np.where(wall_mask, np.nan, field))
        ax.scatter(x, y, color=color, edgecolor="black", linewidth=0.6, s=50, zorder=5)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4),
                    fontsize=6.5, zorder=6)
    ax.set_title(suptitle, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[INFO] Wrote {out}")


def main() -> None:
    device = args.device
    print(f"[INFO] Collecting a {args.steps}x{args.walkers} random-walk buffer on the hard maze.")
    obs, dones = random_walk_buffer(args.walkers, args.steps, args.episode_len, args.step_size, device)

    # feature_dim 0 is the constant Laplacian eigenfunction (no directional
    # information; ALLOIntrinsicRewards's option_directions never emits it
    # either) — always train one extra dim and drop index 0, so --feature_dim N
    # means N *useful* directions, not N total.
    internal_dim = args.feature_dim + 1
    extractor = ALLO(
        observation_dim=2,
        cfg=ALLO_CFG(feature_dim=internal_dim, positional_indices=None),
        device=device,
    )
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
        phi = extractor(grid_obs.to(device)).cpu().numpy()

    # Each dim is TWO intrinsic-reward options (ALLOIntrinsicRewards: sign_n *
    # phi_idx_n) — the reward *potential* each option ascends, not one field.
    # Index 0 dropped (see above); "dim i" here means eigenvector index i+1.
    panels: list[tuple[str, np.ndarray, bool, str]] = []
    for i in range(1, phi.shape[1]):
        field = phi[:, i].reshape(R, C)
        panels.append((f"ALLO dim {i - 1}  (+)", field, True, f"{i - 1}+"))
        panels.append((f"ALLO dim {i - 1}  (-)", -field, True, f"{i - 1}-"))

    guidance = GeodesicGuidance(HARD_MAZE_MAP, CELL_SIZE, device=device)
    with torch.no_grad():
        potential = -guidance.potential(grid_obs.to(device)).cpu().numpy()
    panels.append(("best (GeodesicGuidance): Phi = -d_geo", potential.reshape(R, C), False, "best"))

    n = len(panels)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), squeeze=False)
    for k, (title, field, diverging, _label) in enumerate(panels):
        plot_field(axes[k // ncols][k % ncols], field, wall_mask, title, diverging=diverging)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.suptitle("Intrinsic reward potential fields — hard serpentine maze", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"[INFO] Wrote {args.out}")

    root, ext = os.path.splitext(args.out)
    plot_argmax_summary(panels, wall_mask, f"{root}_argmax{ext}",
                        "Highest-reward point per option — hard serpentine maze")


if __name__ == "__main__":
    main()

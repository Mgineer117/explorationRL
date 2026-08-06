"""Plot the ALLO eigenfunctions and the geodesic-guidance potential over the maze.

Both intrinsic rewards in ``agents/skrl/intrinsic.py`` are literally the
discrete derivative of a scalar potential field over agent position:

    ALLO option n (sign_n, idx_n):  r_n(s, s') = sign_n * (phi_idx_n(s') - phi_idx_n(s))
    GeodesicGuidance ("best"):      r(s, s')   = d_geo(s) - d_geo(s')

so the complete spatial picture is the potential itself: phi_i(x, y) for each
ALLO dimension i (a diverging colormap directly shows the +/- structure a
signed option reads off it: sign=+1 rewards moving toward red, sign=-1 toward
blue) and -d_geo(x, y) for the geodesic-guidance ("best") reward.

No Isaac Sim rollout needed — only the maze layout (for GeodesicGuidance's
Dijkstra and the plot overlay) and, if plotting ALLO, a checkpoint from
``scripts/skrl/pretrain_allo.py``. Isaac Sim is still launched because
importing the maze layout constants pulls in ``isaaclab.sim`` transitively.

Example
-------
    python scripts/skrl/plot_intrinsic_reward.py --task PointMaze-v1 \
        --allo_checkpoint model/allo/PointMaze-v1_dim10.pth --allo_feature_dim 10 \
        --out logs/skrl/intrinsic_reward_maps.png
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="PointMaze-v1")
parser.add_argument("--allo_checkpoint", type=str, default=None,
                    help="Checkpoint from pretrain_allo.py. Omit to skip the ALLO panels.")
parser.add_argument("--allo_feature_dim", type=int, default=11,
                    help="Checkpoint's raw ALLO width, i.e. useful dims + 1 (pretrain_allo.py's "
                         "--feature_dim N saves an (N+1)-wide checkpoint; pass N+1 here).")
parser.add_argument("--allo_hidden_dim", type=int, nargs="+", default=[256, 256])
parser.add_argument("--allo_positional_indices", type=int, nargs="+", default=[0, 1])
parser.add_argument("--bins_per_cell", type=int, default=20, help="Plot grid resolution per maze cell.")
parser.add_argument("--out", type=str, default="logs/skrl/intrinsic_reward_maps.png")
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

from explorationRL.agents.skrl.intrinsic import GeodesicGuidance  # noqa: E402
from explorationRL.extractors import ALLO, ALLO_CFG  # noqa: E402
from explorationRL.tasks.direct.pointmaze.pointmaze_env_cfg import (  # noqa: E402
    CELL_SIZE, POINTMAZE_V1_MAP,
)

ROWS, COLS = len(POINTMAZE_V1_MAP), len(POINTMAZE_V1_MAP[0])
X_MAX, Y_MAX = COLS / 2.0 * CELL_SIZE, ROWS / 2.0 * CELL_SIZE
EXTENT = (-X_MAX, X_MAX, -Y_MAX, Y_MAX)


def _cell_center(i: int, j: int) -> tuple[float, float]:
    return -X_MAX + (j + 0.5) * CELL_SIZE, Y_MAX - (i + 0.5) * CELL_SIZE


def _find(char: str) -> tuple[float, float]:
    for i in range(ROWS):
        for j in range(COLS):
            if POINTMAZE_V1_MAP[i][j] == char:
                return _cell_center(i, j)
    raise ValueError(f"maze_map has no {char!r} cell")


def build_grid(bins_per_cell: int, device):
    """``(obs, wall_mask)`` for an ``(R, C)`` pixel grid over the whole maze,
    ``obs`` shaped ``(R*C, 8)`` matching pointmaze's
    ``[x, y, vx, vy, achieved_x, achieved_y, goal_x, goal_y]`` observation."""
    R, C = ROWS * bins_per_cell, COLS * bins_per_cell
    cell_w, cell_h = (2 * X_MAX) / C, (2 * Y_MAX) / R
    ii, jj = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
    x = -X_MAX + (jj + 0.5) * cell_w
    y = Y_MAX - (ii + 0.5) * cell_h
    wall_mask = np.array([[POINTMAZE_V1_MAP[i // bins_per_cell][j // bins_per_cell] == 1
                           for j in range(C)] for i in range(R)])

    goal_x, goal_y = _find("g")
    n = R * C
    obs = torch.zeros(n, 8, device=device)
    obs[:, 0] = torch.from_numpy(x.reshape(-1)).to(device)
    obs[:, 1] = torch.from_numpy(y.reshape(-1)).to(device)
    obs[:, 4] = obs[:, 0]
    obs[:, 5] = obs[:, 1]
    obs[:, 6] = goal_x
    obs[:, 7] = goal_y
    return obs, wall_mask, (R, C)


def maze_overlay(ax) -> None:
    """Draw walls (already masked in the heatmap) plus start/goal markers."""
    rx, ry = _find("r")
    gx, gy = _find("g")
    ax.plot(rx, ry, "o", color="white", markersize=7, markeredgecolor="black", markeredgewidth=0.6, zorder=4)
    ax.plot(gx, gy, "*", color="gold", markersize=12, markeredgecolor="black", markeredgewidth=0.6, zorder=4)


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
    device = "cpu"
    obs, wall_mask, (R, C) = build_grid(args_cli.bins_per_cell, device)

    panels: list[tuple[str, np.ndarray, bool, str]] = []

    if args_cli.allo_checkpoint:
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
        with torch.no_grad():
            phi = extractor(obs).cpu().numpy()  # (R*C, feature_dim)
        # Each dim is TWO intrinsic-reward options (ALLOIntrinsicRewards: sign_n *
        # phi_idx_n) — the reward *potential* each option ascends, not one field.
        # Index 0 is the constant Laplacian eigenfunction (no directional
        # information; ALLOIntrinsicRewards's option_directions never emits it
        # either) — always dropped, so "dim i" here means eigenvector index i+1.
        for i in range(1, phi.shape[1]):
            field = phi[:, i].reshape(R, C)
            panels.append((f"ALLO dim {i - 1}  (+)", field, True, f"{i - 1}+"))
            panels.append((f"ALLO dim {i - 1}  (-)", -field, True, f"{i - 1}-"))
    else:
        print("[INFO] --allo_checkpoint not given, skipping ALLO panels.")

    guidance = GeodesicGuidance(POINTMAZE_V1_MAP, CELL_SIZE, device=device)
    with torch.no_grad():
        potential = -guidance.potential(obs).cpu().numpy()  # Phi = -d_geo
    panels.append(("best (GeodesicGuidance): Phi = -d_geo", potential.reshape(R, C), False, "best"))

    n = len(panels)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), squeeze=False)
    for k, (title, field, diverging, _label) in enumerate(panels):
        plot_field(axes[k // ncols][k % ncols], field, wall_mask, title, diverging=diverging)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.suptitle(f"Intrinsic reward potential fields — {args_cli.task}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(args_cli.out) or ".", exist_ok=True)
    fig.savefig(args_cli.out, dpi=150)
    print(f"[INFO] Wrote {args_cli.out}")

    root, ext = os.path.splitext(args_cli.out)
    plot_argmax_summary(panels, wall_mask, f"{root}_argmax{ext}",
                        f"Highest-reward point per option — {args_cli.task}")


if __name__ == "__main__":
    main()
    simulation_app.close()

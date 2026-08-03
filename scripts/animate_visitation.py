"""Animate the per-update visitation recorded during training.

``train.py --record_visitation`` writes one occupancy histogram per policy update
to ``<log_dir>/visitation.npz``. This turns one or more of those into a video, so
exploration can be watched as it evolved *during* training rather than inferred
from checkpoint replays.

No Isaac Sim needed: the maze layout travels inside the npz.

Example
-------
    python scripts/animate_visitation.py \
        "PPO=logs/skrl/pointmaze/<ppo_run>/visitation.npz" \
        "AGA+DRND=logs/skrl/pointmaze/<aga_run>/visitation.npz" \
        --out logs/skrl/visitation.mp4
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Animate recorded visitation histograms.")
parser.add_argument("runs", nargs="+", metavar="LABEL=NPZ",
                    help="Recordings to animate side by side.")
parser.add_argument("--out", type=str, default="logs/skrl/visitation.mp4")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--stride", type=int, default=1, help="Keep every Nth update.")
parser.add_argument("--trail", type=int, default=1,
                    help="Updates blended into each frame (>1 smooths the sparse "
                         "per-update occupancy without making it cumulative).")
args = parser.parse_args()


def load(spec: str):
    label, _, path = spec.partition("=")
    if not path:
        raise SystemExit(f"Expected LABEL=NPZ, got {spec!r}")
    d = np.load(path)
    return label, d["frames"], d["extent"], d.get("maze_map"), int(d["steps_per_update"])


def maze_overlay(ax, maze_map, extent) -> None:
    """1 = wall, 2 = start, 3 = goal (codes written by VisitationRecorder)."""
    if maze_map is None:
        return
    rows, cols = maze_map.shape
    x_min, x_max, y_min, y_max = extent
    cell_w, cell_h = (x_max - x_min) / cols, (y_max - y_min) / rows
    for i in range(rows):
        for j in range(cols):
            x, y = x_min + j * cell_w, y_max - (i + 1) * cell_h
            if maze_map[i, j] == 1:
                ax.add_patch(plt.Rectangle((x, y), cell_w, cell_h, facecolor="0.25",
                                           edgecolor="none", zorder=3))
            elif maze_map[i, j] in (2, 3):
                ax.plot(x + cell_w / 2, y + cell_h / 2,
                        marker="o" if maze_map[i, j] == 2 else "*",
                        color="white" if maze_map[i, j] == 2 else "gold",
                        markersize=7 if maze_map[i, j] == 2 else 12,
                        markeredgecolor="black", markeredgewidth=0.6, zorder=4)


def main() -> None:
    runs = [load(spec) for spec in args.runs]
    n_frames = min(len(f) for _, f, _, _, _ in runs) // args.stride
    if n_frames < 2:
        raise SystemExit("Need at least two frames to animate.")

    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.6),
                             squeeze=False, constrained_layout=True)
    images, coverages = [], []
    for ax, (label, frames, extent, maze_map, _) in zip(axes[0], runs):
        # Fixed colour scale across the whole video, so brightness changes mean
        # occupancy changes rather than per-frame renormalisation.
        vmax = float(np.log1p(frames).max())
        im = ax.imshow(np.log1p(frames[0]), origin="lower", extent=tuple(extent),
                       cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest")
        maze_overlay(ax, maze_map, extent)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(label, fontsize=12)
        images.append(im)
        coverages.append(ax.text(0.03, 0.04, "", transform=ax.transAxes, fontsize=9,
                                 color="white", zorder=5))
    title = fig.suptitle("", fontsize=12)

    reachable = None
    if runs[0][3] is not None:
        maze_map = runs[0][3]
        bins_per_cell = runs[0][1].shape[1] // maze_map.shape[0]
        reachable = int((maze_map != 1).sum()) * bins_per_cell**2

    def draw(k: int):
        lo = max(0, (k + 1) * args.stride - args.trail)
        hi = (k + 1) * args.stride
        for im, cov, (_, frames, _, _, steps_per_update) in zip(images, coverages, runs):
            window = frames[lo:hi].sum(axis=0)
            im.set_data(np.log1p(window))
            if reachable:
                cov.set_text(f"{100 * (window > 0).sum() / reachable:.0f}% covered")
        update = hi
        title.set_text(f"policy update {update}  (env step {update * runs[0][4]:,})")
        return images + coverages + [title]

    anim = animation.FuncAnimation(fig, draw, frames=n_frames, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = ("ffmpeg" if args.out.endswith((".mp4", ".mkv", ".webm")) else "pillow")
    anim.save(args.out, writer=writer, fps=args.fps, dpi=120)
    print(f"[INFO] Wrote {args.out} ({n_frames} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()

"""Plot mean +/- 95% CI learning curves over seeds, straight from the tensorboard
event files each run already writes.

    python scripts/plot_learning_curves.py \
        "PPO=logs/skrl/pointmaze/<run>" "PPO=logs/skrl/pointmaze/<run2>" \
        "AGA+best=logs/skrl/pointmaze/<run3>" --out logs/skrl/curves.png

Repeat a label once per seed; runs sharing a label are aggregated. Curves are
interpolated onto a common step grid first, so runs need not be logged in
lockstep.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# two-sided t_{0.975} by degrees of freedom (n-1); n>=11 is close enough to 1.96
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("runs", nargs="+", metavar="LABEL=RUN_DIR")
parser.add_argument("--tag", default="Reward / Total reward (mean)")
parser.add_argument("--out", default="logs/skrl/learning_curves.png")
parser.add_argument("--smooth", type=int, default=25, help="Moving-average window (updates).")
parser.add_argument("--ylabel", default=None)
args = parser.parse_args()


def scalars(run_dir: str, tag: str):
    ea = EventAccumulator(run_dir)
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        raise SystemExit(f"{run_dir}: no scalar {tag!r} (have {ea.Tags()['scalars']})")
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev], float), np.array([e.value for e in ev], float)


def smooth(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return y
    k = np.ones(w) / w
    return np.convolve(np.pad(y, (w - 1, 0), mode="edge"), k, mode="valid")


groups: dict[str, list[str]] = {}
for spec in args.runs:
    label, _, path = spec.partition("=")
    if not path:
        raise SystemExit(f"Expected LABEL=RUN_DIR, got {spec!r}")
    groups.setdefault(label, []).append(path)

fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
for label, dirs in groups.items():
    curves = [scalars(d, args.tag) for d in dirs]
    grid = np.linspace(max(x[0] for x, _ in curves), min(x[-1] for x, _ in curves), 400)
    ys = np.stack([np.interp(grid, x, smooth(y, args.smooth)) for x, y in curves])
    mean = ys.mean(0)
    n = len(ys)
    # 95% CI of the mean across seeds; a single seed gets no band.
    half = T975.get(n - 1, 1.96) * ys.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    line, = ax.plot(grid, mean, label=f"{label} (n={n})", linewidth=1.8)
    ax.fill_between(grid, mean - half, mean + half, color=line.get_color(), alpha=0.2, linewidth=0)

ax.set_xlabel("env steps")
ax.set_ylabel(args.ylabel or args.tag)
ax.grid(alpha=0.25)
ax.legend()
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
fig.savefig(args.out, dpi=150)
print(f"[INFO] Wrote {args.out}")

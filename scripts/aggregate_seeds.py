#!/usr/bin/env python3
"""Aggregate per-seed result CSVs written by scripts/skrl/train.py.

Each finished run drops one row at ``results/<run-tag>/<task>_<algo>_seed<seed>.csv``
(see ``train_utils.write_seed_result``). This collects every row for a run tag and
writes:

    results/<tag>_runs.csv   one row per seed
    results/<tag>.csv        one row per (task, algorithm): mean/std/95% CI over seeds

Usage:
    python scripts/aggregate_seeds.py --run-tag run_20260723_120000
    python scripts/aggregate_seeds.py --run-tag run_... --out results/run_...
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics
import sys

METRICS = ("return_mean", "success_rate")


def _read_rows(run_dir: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*.csv"))):
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    rows.append(row)
        except Exception as exc:  # noqa: BLE001
            print(f"[aggregate] skipping {path}: {exc}", file=sys.stderr)
    return rows


def _summarize(values: list[float]) -> tuple[float, float, float]:
    """(mean, sample-std, 95% CI half-width). std/CI are 0 for a single seed."""
    n = len(values)
    mean = statistics.fmean(values)
    if n < 2:
        return mean, 0.0, 0.0
    std = statistics.stdev(values)
    ci = 1.96 * std / math.sqrt(n)
    return mean, std, ci


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-tag", required=True, help="Run tag (the results/<tag>/ directory).")
    p.add_argument("--results-dir", default="results", help="Root results directory.")
    p.add_argument("--out", default=None, help="Output basename (default results/<tag>).")
    args = p.parse_args()

    run_dir = os.path.join(args.results_dir, args.run_tag)
    if not os.path.isdir(run_dir):
        print(f"[aggregate] no such directory: {run_dir}", file=sys.stderr)
        return 1

    rows = _read_rows(run_dir)
    if not rows:
        print(f"[aggregate] no per-seed CSVs found in {run_dir}", file=sys.stderr)
        return 1

    out_base = args.out or os.path.join(args.results_dir, args.run_tag)
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)

    # ── per-seed rows ─────────────────────────────────────────────────────── #
    runs_path = f"{out_base}_runs.csv"
    fields = ["task", "algorithm", "seed", *METRICS]
    with open(runs_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in sorted(rows, key=lambda r: (r.get("task", ""), r.get("algorithm", ""),
                                               int(r.get("seed", 0)))):
            w.writerow(row)

    # ── aggregate over seeds ──────────────────────────────────────────────── #
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row.get("task", ""), row.get("algorithm", "")), []).append(row)

    summary_path = f"{out_base}.csv"
    header = ["task", "algorithm", "num_seeds"]
    for m in METRICS:
        header += [f"{m}_mean", f"{m}_std", f"{m}_ci95"]
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for (task, algo), grows in sorted(groups.items()):
            out = [task, algo, len(grows)]
            for m in METRICS:
                vals = []
                for r in grows:
                    try:
                        vals.append(float(r[m]))
                    except (KeyError, TypeError, ValueError):
                        pass
                if vals:
                    mean, std, ci = _summarize(vals)
                    out += [f"{mean:.6f}", f"{std:.6f}", f"{ci:.6f}"]
                else:
                    out += ["", "", ""]
            w.writerow(out)

    print(f"[aggregate] {len(rows)} run(s) -> {runs_path}")
    print(f"[aggregate] {len(groups)} group(s) -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

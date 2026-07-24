#!/usr/bin/env python3
"""Emit a W&B sweep yaml from ``search/configs/<algorithm>.yaml`` + an env.

The searched space lives in ``search/configs/`` (one file per algorithm, applied
to every env). This script turns one of those into what ``wandb sweep`` wants: it
copies the config's ``metric:``/``parameters:`` blocks through verbatim and
synthesizes the ``program:``/``project:``/``name:``/``command:`` lines around them.

Every sweep goes into the SAME W&B project (``explorationRL-Search``) and is
identified by its ``name:`` = ``<env>-<algorithm>`` — so all runs for a given
env+algorithm accumulate in one place across relaunches. Trials invoke
``scripts/skrl/train.py`` directly.

Usage:
    python search/build_sweep.py --algorithm ppo --env PointMaze-v1 \
        [--method bayes] [--project NAME] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS = Path(__file__).resolve().parent / "configs"

# The single W&B project every sweep lands in (set explicitly so wandb doesn't
# auto-derive one from the launch path and split runs).
DEFAULT_PROJECT = "explorationRL-Search"

# Isaac Lab task ids that can be swept. Extend as new envs are added.
ISAACLAB_ENVS = ("PointMaze-v1",)


def available_algorithms() -> list[str]:
    """Algorithm names discoverable in configs/ — the file stem IS the name."""
    return sorted(p.stem for p in _CONFIGS.glob("*.yaml"))


def load_config(algorithm: str) -> dict:
    path = _CONFIGS / f"{algorithm}.yaml"
    if not path.exists():
        raise SystemExit(
            f"No search config for '{algorithm}'. Available: {', '.join(available_algorithms())}"
        )
    cfg = yaml.safe_load(path.read_text())
    for required in ("algorithm", "num_envs", "metric", "parameters"):
        if required not in cfg:
            raise SystemExit(f"{path} is missing required key '{required}'.")
    for required in ("name", "goal"):
        if required not in (cfg.get("metric") or {}):
            raise SystemExit(f"{path}: metric is missing required key '{required}'.")
    assert_method_compatible(cfg, path)
    return cfg


def assert_method_compatible(cfg: dict, path) -> None:
    """Reject a ``method: grid`` config that still declares a continuous range.

    W&B's grid search enumerates the cross product of discrete value lists; it
    cannot enumerate a ``distribution:`` and silently drops it. Catch it here.
    """
    if str(cfg.get("method", "")).lower() != "grid":
        return
    continuous = sorted(
        name for name, spec in (cfg.get("parameters") or {}).items()
        if isinstance(spec, dict) and "distribution" in spec
    )
    if continuous:
        raise SystemExit(
            f"{path}: method is 'grid' but these parameters declare a continuous "
            f"distribution grid search cannot enumerate: {continuous}. Give each an "
            f"explicit 'values:' list, or switch the config to method: bayes."
        )


def grid_size(cfg: dict) -> int:
    """Number of trials a ``method: grid`` sweep enumerates (0 if not grid)."""
    if str(cfg.get("method", "")).lower() != "grid":
        return 0
    total = 1
    for spec in (cfg.get("parameters") or {}).values():
        if isinstance(spec, dict) and "values" in spec:
            total *= max(1, len(spec["values"]))
    return total


def build(algorithm: str, env: str, *, method: str, project: str) -> dict:
    cfg = load_config(algorithm)

    command = [
        "${env}", "python", "scripts/skrl/train.py",
        "--task", env,
        "--algorithm", str(cfg["algorithm"]),
        "--num_envs", str(cfg["num_envs"]),
    ]
    command.append("${args}")

    return {
        "program": "scripts/skrl/train.py",
        # ONE project; sweeps are told apart by NAME (env-algorithm).
        "project": project,
        "name": f"{env}-{algorithm}",
        # The config's own `method:` wins over the CLI default.
        "method": str(cfg.get("method", method)),
        "metric": cfg["metric"],
        "parameters": cfg["parameters"],
        "command": command,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--algorithm", required=True, choices=available_algorithms())
    parser.add_argument("--env", required=True, choices=list(ISAACLAB_ENVS))
    parser.add_argument("--method", default="bayes", choices=["bayes", "grid", "random"],
                        help="Fallback only — a config that sets its own `method:` overrides this.")
    parser.add_argument("--count", action="store_true",
                        help="Print the number of trials a grid sweep enumerates, then exit.")
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help=f"W&B project for this sweep (default {DEFAULT_PROJECT}).")
    parser.add_argument("--out", default="-", help="Output path, or '-' for stdout.")
    args = parser.parse_args()

    if args.count:
        size = grid_size(load_config(args.algorithm))
        print(f"{args.algorithm}: {size} grid trials" if size
              else f"{args.algorithm}: not a grid sweep (method is not 'grid')")
        return 0

    sweep = build(args.algorithm, args.env, method=args.method, project=args.project)
    text = yaml.safe_dump(sweep, sort_keys=False, default_flow_style=False, width=100)
    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text)
        print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

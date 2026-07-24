"""List the environments registered by explorationRL (and, optionally, Isaac Lab's own).

Examples
--------
    python scripts/list_envs.py
    python scripts/list_envs.py --keyword Maze
    python scripts/list_envs.py --all          # also include isaaclab_tasks' Isaac-* envs

Registration happens as an import side effect, so this launches Isaac Sim headless
first (same as Isaac Lab's own ``scripts/environments/list_envs.py``) before importing
the task packages.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="List explorationRL environments.")
parser.add_argument("--keyword", type=str, default=None, help="Only show tasks whose id contains this.")
parser.add_argument("--all", action="store_true", default=False,
                    help="Also list the Isaac-* envs shipped with isaaclab_tasks.")
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
from prettytable import PrettyTable  # noqa: E402

import explorationRL.tasks  # noqa: F401,E402 — registers PointMaze-v1 etc.

if args_cli.all:
    import isaaclab_tasks  # noqa: F401,E402


def _algorithms(kwargs: dict) -> str:
    """Algorithm names implied by the ``skrl_<algo>_cfg_entry_point`` kwargs."""
    # The bare "skrl_cfg_entry_point" (the default agent) yields an empty name — drop it.
    algos = sorted(
        name
        for key in kwargs
        if key.startswith("skrl_") and key.endswith("_cfg_entry_point")
        and (name := key[len("skrl_"):-len("_cfg_entry_point")])
    )
    return ", ".join(algos) if algos else "-"


def main():
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config", "Algorithms"])
    table.title = "Available Environments"
    for column in ("Task Name", "Entry Point", "Config", "Algorithms"):
        table.align[column] = "l"

    index = 0
    for task_spec in gym.registry.values():
        # explorationRL tasks live under the explorationRL package; Isaac Lab's own
        # are the Isaac-* ids, only shown with --all.
        entry_point = str(task_spec.entry_point)
        is_ours = entry_point.startswith("explorationRL")
        if not (is_ours or (args_cli.all and "Isaac" in task_spec.id)):
            continue
        if args_cli.keyword is not None and args_cli.keyword not in task_spec.id:
            continue
        index += 1
        table.add_row([
            index,
            task_spec.id,
            entry_point,
            task_spec.kwargs.get("env_cfg_entry_point", "-"),
            _algorithms(task_spec.kwargs),
        ])

    # Flush explicitly: simulation_app.close() can tear the process down before
    # a buffered stdout (e.g. when piped to a file) is written out.
    print("No matching environments found." if index == 0 else table, flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

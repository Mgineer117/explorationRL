"""Helpers shared by scripts/skrl/train.py and play.py.

Deliberately small: skrl's own experiment layer already handles TensorBoard /
W&B logging and checkpointing, so this only adds the glue that layer does not:
resolving W&B sweep overrides into the agent config, an algorithm-aware default
for the number of parallel envs, and a clean W&B teardown.
"""

from __future__ import annotations

import os

# Off-policy / replay-based algorithms want far fewer parallel envs than the
# on-policy PPO/TRPO family (a big replay buffer >> many envs). Mirrors the
# split contractionRL's train.py uses.
_SAC_LIKE_ALGOS = {"sac", "ddpg", "td3"}
_DEFAULT_NUM_ENVS_SAC = 64
_DEFAULT_NUM_ENVS_ONPOLICY = 4096


def default_num_envs(algorithm: str) -> int:
    """Algorithm-aware default for ``--num_envs`` (used when not given)."""
    return _DEFAULT_NUM_ENVS_SAC if algorithm.lower() in _SAC_LIKE_ALGOS else _DEFAULT_NUM_ENVS_ONPOLICY


def _set_dotted(cfg: dict, dotted_key: str, value) -> None:
    """Set ``cfg["a"]["b"]["c"] = value`` from a ``"a.b.c"`` key, creating dicts."""
    parts = dotted_key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def apply_wandb_sweep_overrides(agent_cfg: dict) -> None:
    """Write every ``wandb.config`` entry into ``agent_cfg`` by dotted key.

    A W&B sweep sets its sampled hyperparameters on ``wandb.config`` with keys
    like ``agent.discount_factor`` or ``models.policy.network`` (see
    ``search/configs/``). They only exist once the run is live, and must reach
    ``agent_cfg`` before any model is built from it — train.py calls this right
    after ``wandb.init`` for a sweep.
    """
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    for key, value in dict(wandb.config).items():
        if "." in key:  # only dotted keys target agent_cfg; scalars like `seed` handled elsewhere
            _set_dotted(agent_cfg, key, value)
        elif key in ("seed",):
            agent_cfg[key] = value


def install_wandb_scalar_hook() -> None:
    """Best-effort: make skrl's ``track_data`` scalars also reach ``wandb.log``.

    skrl already mirrors its writer to W&B when ``experiment.wandb`` is true, so
    this is a no-op unless that path is unavailable. Kept as a named hook so the
    call sites read the same as contractionRL's train.py.
    """
    return None


def finish_wandb(args_cli) -> None:
    """Close the active W&B run (no-op if W&B is off / not running)."""
    if getattr(args_cli, "no_wandb", False):
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:  # noqa: BLE001 — teardown must never raise
        pass


def is_sweep() -> bool:
    """True when running under a ``wandb agent`` (a sweep worker)."""
    return "WANDB_SWEEP_ID" in os.environ


def write_seed_result(skrl_env, agent, *, task: str, algorithm: str, seed: int,
                      max_len: int, num_envs: int) -> None:
    """Deterministically roll out one episode and write a one-row result CSV.

    Produces ``results/<EXP_RUN_TAG>/<task>_<algorithm>_seed<seed>.csv`` with the
    mean first-episode return and success rate (an env "succeeds" if it earns any
    positive reward before its episode ends — exact for PointMaze's sparse goal
    reward). ``scripts/aggregate_seeds.py`` collects these across seeds.

    Best-effort and fully guarded: a failure here must never fail a training run
    that already completed. Skipped by ``--skip_final_eval`` (sweeps read the
    trainer scalars instead).
    """
    import csv

    import torch

    try:
        # skrl 2.x: enable_training_mode(False); 1.x used set_running_mode("eval").
        if hasattr(agent, "enable_training_mode"):
            agent.enable_training_mode(False)
        elif hasattr(agent, "set_running_mode"):
            agent.set_running_mode("eval")
        observations, _ = skrl_env.reset()
        states = skrl_env.state()
        device = observations.device
        returns = torch.zeros(num_envs, device=device)
        success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        alive = torch.ones(num_envs, dtype=torch.bool, device=device)

        with torch.no_grad():
            for t in range(int(max_len)):
                actions, outputs = agent.act(observations, states, timestep=t, timesteps=int(max_len))
                if isinstance(outputs, dict) and outputs.get("mean_actions") is not None:
                    actions = outputs["mean_actions"]
                observations, rewards, terminated, truncated, _ = skrl_env.step(actions)
                states = skrl_env.state()
                r = rewards.reshape(-1)
                returns += r * alive.float()
                success |= (r > 0) & alive
                alive &= ~(terminated | truncated).reshape(-1).bool()
                if not bool(alive.any()):
                    break

        return_mean = float(returns.mean().item())
        success_rate = float(success.float().mean().item())
    except Exception as exc:  # noqa: BLE001 — never fail a finished run
        print(f"[train] final eval skipped ({type(exc).__name__}: {exc})")
        return

    run_tag = os.environ.get("EXP_RUN_TAG", "adhoc")
    out_dir = os.path.join("results", run_tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task}_{algorithm}_seed{seed}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "algorithm", "seed", "return_mean", "success_rate"])
        w.writerow([task, algorithm, seed, f"{return_mean:.6f}", f"{success_rate:.6f}"])
    print(f"[train] eval: return_mean={return_mean:.4f} success_rate={success_rate:.4f} -> {out_path}")

    try:
        import wandb

        if wandb.run is not None:
            wandb.run.summary["eval/return_mean"] = return_mean
            wandb.run.summary["eval/success_rate"] = success_rate
    except Exception:  # noqa: BLE001
        pass

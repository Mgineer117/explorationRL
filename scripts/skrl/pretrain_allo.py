"""Offline-pretrain a frozen ALLO Laplacian representation for an environment.

Standalone counterpart to IRPO's built-in extractor pretraining
(``agents/skrl/irpo.py``'s ``pretrain_extractor``): produces a reusable ALLO
checkpoint for a given task, independent of any particular RL agent/run, so
other consumers of ``IntrinsicRewardWrapper(int_reward="allo")``
(``agents/skrl/intrinsic.py``) can load a fitted representation via
``allo_pretrained_path`` instead of training one online from scratch.

The representation is fit under a uniform-random policy, with uniform-cell
resets if the env supports them (``env.unwrapped._uniform_reset``), so it sees
a near-uniform state distribution rather than any one policy's occupancy —
the measure ALLO's orthonormality objective assumes.

Example
-------
    python scripts/skrl/pretrain_allo.py --task PointMaze-v1 --feature_dim 10
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, required=True, help="Environment / task id, e.g. PointMaze-v1.")
parser.add_argument("--feature_dim", type=int, default=10,
                    help="ALLO embedding width (number of eigenvector directions learned).")
parser.add_argument("--hidden_dim", type=int, nargs="+", default=[256, 256])
parser.add_argument("--positional_indices", type=int, nargs="+", default=None,
                    help="Observation slice ALLO learns over (e.g. 0 1 for xy). Default: whole observation.")
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--discount", type=float, default=0.9,
                    help="Discount of the geometric distribution the future state s' is drawn from.")
parser.add_argument("--pretrain_epochs", type=int, default=5000)
parser.add_argument("--collect_steps", type=int, default=256,
                    help="Per-env length of the uniform-random buffer ALLO pretrains on.")
parser.add_argument("--no_uniform_reset", action="store_true", default=False,
                    help="Don't force uniform-cell resets while collecting (uses the env's normal reset).")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default=None,
                    help="Checkpoint path (default: model/allo/<task>_dim<feature_dim>.pth).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── Post-app imports ─────────────────────────────────────────────────────── #
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import explorationRL.tasks  # noqa: F401,E402 — registers PointMaze-v1 etc.
from explorationRL.extractors import ALLO, ALLO_CFG  # noqa: E402


@torch.no_grad()
def collect_uniform_buffer(env, steps: int, uniform_reset: bool, device):
    """Roll out a uniform-random policy to a ``(T, num_envs, obs_dim)``/dones buffer."""
    base = getattr(env, "unwrapped", env)
    prev = getattr(base, "_uniform_reset", None)
    if uniform_reset and hasattr(base, "_uniform_reset"):
        base._uniform_reset = True

    act_dim = int(env.action_space.shape[0])
    observations, _ = env.reset()
    n = observations.shape[0]
    obs_buf = torch.empty(steps, n, observations.shape[-1], device=device)
    done_buf = torch.empty(steps, n, dtype=torch.bool, device=device)
    for t in range(steps):
        actions = torch.rand(n, act_dim, device=device) * 2.0 - 1.0
        next_obs, _, terminated, truncated, _ = env.step(actions)
        obs_buf[t] = observations
        done_buf[t] = (terminated.bool() | truncated.bool()).reshape(n)
        observations = next_obs

    if prev is not None:
        base._uniform_reset = prev
    env.reset()
    return obs_buf, done_buf


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    device = env.device

    print(f"[INFO] Collecting a {args_cli.collect_steps}x{args_cli.num_envs} uniform buffer on {args_cli.task}.")
    observations, dones = collect_uniform_buffer(
        env, args_cli.collect_steps, not args_cli.no_uniform_reset, device)

    # feature_dim 0 is the constant Laplacian eigenfunction (no directional
    # information, see extractors/allo.py's module docstring and
    # ALLOIntrinsicRewards's `option_directions`, which already never emits
    # it) — always train one extra dim and reserve index 0 as dead, so
    # --feature_dim N means N *useful* directions, not N total.
    internal_dim = args_cli.feature_dim + 1
    obs_dim = int(env.observation_space.shape[0])
    extractor = ALLO(
        observation_dim=obs_dim,
        cfg=ALLO_CFG(
            feature_dim=internal_dim,
            hidden_dim=list(args_cli.hidden_dim),
            positional_indices=list(args_cli.positional_indices) if args_cli.positional_indices else None,
            learning_rate=args_cli.learning_rate,
            batch_size=args_cli.batch_size,
            discount=args_cli.discount,
        ),
        device=device,
    )

    print(f"[INFO] Pretraining ALLO ({args_cli.feature_dim} useful dims + 1 reserved) "
          f"for {args_cli.pretrain_epochs} epochs.")
    log_every = max(1, args_cli.pretrain_epochs // 20)
    metrics: dict = {}
    for epoch in range(args_cli.pretrain_epochs):
        metrics = extractor.learn(observations, dones)
        if epoch % log_every == 0 or epoch == args_cli.pretrain_epochs - 1:
            print(f"[INFO]   epoch {epoch:5d}  loss={metrics['ALLO / loss']:.4f}  "
                  f"graph={metrics['ALLO / graph loss']:.4f}  dual_norm={metrics['ALLO / dual norm']:.4f}")

    out = args_cli.out or os.path.join("model", "allo", f"{args_cli.task}_dim{args_cli.feature_dim}.pth")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(extractor.state_dict(), out)
    print(f"[INFO] Saved pretrained ALLO to {out}.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

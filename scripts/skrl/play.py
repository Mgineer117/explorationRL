"""Roll out a trained skrl checkpoint on an explorationRL Isaac Lab environment.

Examples
--------
    python scripts/skrl/play.py --task PointMaze-v1 --algorithm ppo \
        --checkpoint logs/skrl/pointmaze/<run>/checkpoints/best_agent.pt --headon

    # record a video instead of opening a window
    python scripts/skrl/play.py --task PointMaze-v1 --checkpoint <ckpt> --video

With no --checkpoint the newest ``*.pt`` under logs/skrl/<experiment>/ is used.
"""

from __future__ import annotations

import argparse
import sys

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--task", type=str, default="")
_pre_args, _ = _pre.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Play a trained skrl agent (explorationRL).")
parser.add_argument("--headon", action="store_true", default=False, help="Render the Isaac Sim GUI.")
parser.add_argument("--task", type=str, default=None, help="Task id, e.g. PointMaze-v1.")
parser.add_argument("--algorithm", "--algo", type=str, default="ppo")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a .pt checkpoint.")
parser.add_argument("--num_episodes", type=int, default=5, help="Episodes to roll out.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])
parser.add_argument("--video", action="store_true", default=False, help="Record a video.")
parser.add_argument("--video_length", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = not args_cli.headon
if args_cli.video:
    args_cli.enable_cameras = True
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")

hydra_args = [a for a in hydra_args if not (a.startswith("--") and ("=" in a or "." in a))]
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import glob  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import explorationRL.tasks  # noqa: F401,E402
from explorationRL.agents.skrl.runner import ExplorationRunner  # noqa: E402

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = f"skrl_{algorithm.replace('-', '_')}_cfg_entry_point"


def _latest_checkpoint(experiment_dir: str) -> str | None:
    """Newest ``*.pt`` under ``logs/skrl/<experiment_dir>/`` (any run)."""
    root = os.path.abspath(os.path.join("logs", "skrl", experiment_dir))
    candidates = glob.glob(os.path.join(root, "**", "*.pt"), recursive=True)
    return max(candidates, key=os.path.getmtime) if candidates else None


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
        env_cfg.seed = args_cli.seed

    # Never start a W&B run just to replay a policy.
    agent_cfg["agent"].setdefault("experiment", {})["wandb"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    checkpoint = args_cli.checkpoint or _latest_checkpoint(
        agent_cfg["agent"]["experiment"].get("directory", "")
    )
    if not checkpoint or not os.path.isfile(checkpoint):
        raise SystemExit(
            f"No checkpoint found (--checkpoint not given and nothing under "
            f"logs/skrl/{agent_cfg['agent']['experiment'].get('directory', '')}/)."
        )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    max_len = int(getattr(env.unwrapped, "max_episode_length", 1000))
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join("logs", "skrl", "play_videos"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    runner = ExplorationRunner(env, agent_cfg)
    print(f"[INFO] Loading checkpoint: {checkpoint}")
    runner.agent.load(checkpoint)
    # skrl 2.x: enable_training_mode(False); 1.x used set_running_mode("eval").
    if hasattr(runner.agent, "enable_training_mode"):
        runner.agent.enable_training_mode(False)
    elif hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")

    total_steps = max_len * max(1, args_cli.num_episodes)
    observations, _ = env.reset()
    states = env.state()
    returns = torch.zeros(args_cli.num_envs, device=observations.device)
    episode_returns: list[float] = []

    with torch.no_grad():
        for t in range(total_steps):
            # Never collapse to the distribution mean: policy gradient is derived for
            # the stochastic policy, so that's the only one it's valid to evaluate.
            actions, _ = runner.agent.act(observations, states, timestep=t, timesteps=total_steps)
            observations, rewards, terminated, truncated, _ = env.step(actions)
            states = env.state()
            returns += rewards.reshape(-1)
            done = (terminated | truncated).reshape(-1).bool()
            if bool(done.any()):
                episode_returns.extend(returns[done].tolist())
                returns[done] = 0.0

    if episode_returns:
        mean_return = sum(episode_returns) / len(episode_returns)
        print(f"[INFO] {len(episode_returns)} episode(s) — mean return {mean_return:.4f}")
    else:
        print("[INFO] No episode completed within the rollout budget.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    os._exit(0)

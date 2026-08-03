"""Train RL agents with skrl on explorationRL Isaac Lab environments.

Examples
--------
    python scripts/skrl/train.py --task PointMaze-v1 --algorithm ppo
    python scripts/skrl/train.py --task PointMaze-v1 --algorithm sac --num_envs 64
    python scripts/skrl/train.py --task PointMaze-v1 --algorithm trpo --seed 3 --headon

Isaac Sim runs headless by default; pass ``--headon`` to render a window.

W&B
---
By default every run logs to W&B (``--no_wandb`` to disable). Under a
``wandb agent`` (a sweep), the sampled hyperparameters on ``wandb.config`` are
applied to the agent config *before* any model is built, and skrl's TensorBoard
scalars are synced to the active sweep run.
"""

from __future__ import annotations

import argparse
import sys

# ─── Pre-parse: know --classic BEFORE importing Isaac Sim ─────────────────── #
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--classic", action="store_true", default=False)
_pre.add_argument("--task", type=str, default="")
_pre_args, _ = _pre.parse_known_args()
_is_classic = _pre_args.classic or _pre_args.task.startswith("classic")

if not _is_classic:
    from isaaclab.app import AppLauncher

# ─── Argument parser ──────────────────────────────────────────────────────── #
parser = argparse.ArgumentParser(description="Train an RL agent with skrl (explorationRL).")
parser.add_argument("--classic", action="store_true", default=False,
                    help="Classic (non-Isaac) route — reserved for the ported custom algorithms.")
parser.add_argument("--headon", action="store_true", default=False,
                    help="Render the Isaac Sim GUI (headless by default).")
parser.add_argument("--task", type=str, default=None, help="Environment / task id, e.g. PointMaze-v1.")
parser.add_argument("--algorithm", "--algo", type=str, default="ppo",
                    help="Algorithm: ppo | sac | trpo | ... (resolves skrl_<algo>_cfg_entry_point).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed (default: from the yaml).")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume from.")
parser.add_argument("--num_timesteps", "--num-timesteps", type=int, default=None,
                    help="Total training timesteps (overrides trainer.timesteps).")
parser.add_argument("--max_iterations", type=int, default=None,
                    help="Training iterations (= max_iterations * agent.rollouts timesteps).")
parser.add_argument("--skip_final_eval", "--skip-final-eval", action="store_true", default=False,
                    help="Skip the post-training evaluation rollout (sweeps read the trainer scalars).")
parser.add_argument("--agent", type=str, default=None, help="Explicit skrl cfg entry-point key override.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])
parser.add_argument("--int_reward", "--int-reward", type=str, default=None,
                    choices=["drnd", "allo", "best"],
                    help="Wrap the env with an intrinsic reward model; it is exposed per step "
                         "as infos['int_reward'] (default: none).")
parser.add_argument("--int_reward_option", "--int-reward-option", type=int, default=0,
                    help="Which ALLO eigenvector direction to use (--int_reward allo only).")
parser.add_argument("--int_reward_num_options", "--int-reward-num-options", type=int, default=2,
                    help="Number of ALLO directions the extractor provides.")
parser.add_argument("--record_visitation", "--record-visitation", action="store_true", default=False,
                    help="Record one occupancy histogram per policy update to "
                         "<log_dir>/visitation.npz (animate it with scripts/skrl/visitation.py --animate).")
parser.add_argument("--visitation_bins_per_cell", type=int, default=10)
parser.add_argument("--distributed", action="store_true", default=False)

# Common agent-config overrides (flag name == YAML key under `agent:`).
parser.add_argument("--learning_rate", "--learning-rate", "--lr", type=float, default=None)
parser.add_argument("--discount_factor", "--discount-factor", type=float, default=None)
parser.add_argument("--rollouts", type=int, default=None)
parser.add_argument("--learning_epochs", "--learning-epochs", type=int, default=None)
parser.add_argument("--mini_batches", "--mini-batches", type=int, default=None)
parser.add_argument("--batch_size", "--batch-size", type=int, default=None)
parser.add_argument("--entropy_loss_scale", "--entropy-loss-scale", type=float, default=None)

# W&B
parser.add_argument("--no_wandb", "--no-wandb", action="store_true", default=False)
parser.add_argument("--wandb_project", "--wandb-project", type=str, default="explorationRL")
parser.add_argument("--wandb_run_name", "--wandb-run-name", type=str, default=None)

# Video (Isaac only)
parser.add_argument("--video", action="store_true", default=False, help="Record training videos.")
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)

if not _is_classic:
    AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

if _is_classic:
    raise SystemExit(
        "The --classic (non-Isaac) route is reserved for the ported custom algorithms "
        "(Phase F). For the discrete/Mujoco envs use the legacy main.py pipeline; for "
        "PointMaze-v1 drop --classic (it is a true Isaac Lab env)."
    )

args_cli.headless = not args_cli.headon
if args_cli.video:
    args_cli.enable_cameras = True
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")

# hydra needs a clean argv (strip our dotted/`=` passthrough) before launching.
args_cli.kit_args = (getattr(args_cli, "kit_args", "") or "") + " --/app/hangDetector/enabled=false"
hydra_args = [a for a in hydra_args if not (a.startswith("--") and ("=" in a or "." in a))]

# `agent.<key>=<value>` overrides are applied by _apply_agent_overrides below, not
# by hydra: the agent node hydra composes for this task is not the yaml's, so it
# rejects *every* agent key ("Key '<key>' is not in struct") — including plain
# ones like agent.discount_factor. This is the same dotted dialect the W&B sweeps
# use (search/configs/*.yaml). `env.<key>=<value>` still goes to hydra as usual.
agent_overrides = [a for a in hydra_args if a.startswith("agent.") and "=" in a]
hydra_args = [a for a in hydra_args if a not in agent_overrides]
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402

carb.settings.get_settings().set("/log/logger/channelFilter", "-omni.hydra")

# ─── Post-app imports ─────────────────────────────────────────────────────── #
import os  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import skrl  # noqa: E402
import yaml  # noqa: E402
from packaging import version  # noqa: E402

sys.path.append(os.path.dirname(__file__))
from train_utils import (  # noqa: E402
    _set_dotted, apply_wandb_sweep_overrides, default_num_envs, finish_wandb,
    install_wandb_scalar_hook, is_sweep, upload_videos_to_wandb, warmup_video_pipeline,
    write_seed_result,
)

SKRL_VERSION = "2.1.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(f"Unsupported skrl {skrl.__version__}; need >= {SKRL_VERSION}.")
    sys.exit(1)

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import explorationRL.tasks  # noqa: F401,E402 — registers PointMaze-v1 etc.
from explorationRL.agents.skrl.intrinsic import wrap_intrinsic  # noqa: E402
from explorationRL.agents.skrl.runner import ExplorationRunner  # noqa: E402

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = args_cli.agent or f"skrl_{algorithm.replace('-', '_')}_cfg_entry_point"


def _apply_agent_overrides(agent_cfg: dict) -> None:
    a = agent_cfg["agent"]
    for key in ("learning_rate", "discount_factor", "rollouts", "learning_epochs",
                "mini_batches", "batch_size", "entropy_loss_scale"):
        val = getattr(args_cli, key, None)
        if val is not None:
            a[key] = val
    for override in agent_overrides:
        path, _, value = override.partition("=")
        *parents, leaf = path.split(".")[1:]  # drop the leading "agent"
        node = a
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"'{path}' is not a key of {agent_cfg_entry_point}")
        # yaml.safe_load: "0.98" -> float, "true" -> bool, "[1, 2]" -> list.
        _set_dotted(a, ".".join(parents + [leaf]), yaml.safe_load(value))
        print(f"[INFO] Agent config override: {path} = {node[leaf]!r}")


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    # ── num_envs (algorithm-aware default) / device ───────────────────────── #
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    else:
        env_cfg.scene.num_envs = default_num_envs(algorithm)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # ── timesteps / seed ──────────────────────────────────────────────────── #
    if args_cli.max_iterations is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    if args_cli.num_timesteps is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.num_timesteps
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    seed = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 42)
    random.seed(seed)
    agent_cfg["seed"] = seed
    env_cfg.seed = seed

    _apply_agent_overrides(agent_cfg)

    # ── log dir ───────────────────────────────────────────────────────────── #
    log_root = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_seed{seed}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root
    agent_cfg["agent"]["experiment"]["experiment_name"] = run_stamp
    log_dir = os.path.join(log_root, run_stamp)
    print(f"[INFO] Logging experiment in: {log_dir}")

    # ── W&B ───────────────────────────────────────────────────────────────── #
    experiment = agent_cfg["agent"].setdefault("experiment", {})
    wkw = experiment.setdefault("wandb_kwargs", {})
    wkw["project"] = args_cli.wandb_project
    wkw["name"] = args_cli.wandb_run_name or run_stamp
    if args_cli.no_wandb:
        experiment["wandb"] = False
    elif is_sweep():
        # A sweep worker: init W&B early so wandb.config (the sampled trial) is
        # applied to agent_cfg BEFORE models are built, and sync skrl's
        # TensorBoard scalars into the trial run. skrl's own wandb is then off
        # (avoids a second wandb.init).
        import wandb
        if wandb.run is None:
            wandb.init(project=args_cli.wandb_project, sync_tensorboard=True)
        apply_wandb_sweep_overrides(agent_cfg)
        _apply_agent_overrides(agent_cfg)  # CLI still wins over the sampled trial
        experiment["wandb"] = False
        install_wandb_scalar_hook()
    else:
        experiment["wandb"] = True
        # NOT sync_tensorboard: it only patches torch's SummaryWriter, which skrl
        # 2.x no longer uses, so it syncs nothing while still seizing the step
        # axis ("Step cannot be set when using tensorboard syncing" — which then
        # drops the step on our own scalar/video logs). install_wandb_scalar_hook
        # forwards skrl's scalars directly instead.
        wkw["sync_tensorboard"] = False
        install_wandb_scalar_hook()

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # ── env ───────────────────────────────────────────────────────────────── #
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        # Isaac Lab builds the rgb annotator lazily inside the first render().
        # Doing that on training step 0 — before the RTX renderer has produced a
        # single frame — makes omni.syntheticdata activate its SDG graph against
        # empty buffers and die with "Unable to write from unknown dtype, kind=f,
        # size=0". Pump a few renders and build the annotator here instead, while
        # nothing is consuming its output yet.
        warmup_video_pipeline(env)
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            # Skip step 0: the trainer's very first step is also the noisiest
            # moment for the renderer, and the warm-up above already covers it.
            "step_trigger": lambda step: step > 0 and step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording training videos.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        if not args_cli.no_wandb:
            env = upload_videos_to_wandb(env, video_kwargs["video_folder"])

    # Episode length is read BEFORE the skrl wrapper (which does not forward it)
    # and drives the one-episode final eval below.
    max_len = int(getattr(env.unwrapped, "max_episode_length", 1000))
    num_envs = int(env_cfg.scene.num_envs)

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    env = wrap_intrinsic(env, args_cli.int_reward, option=args_cli.int_reward_option,
                         num_options=args_cli.int_reward_num_options)

    if args_cli.record_visitation:
        # One frame per policy update: the trainer collects `rollouts` steps, then
        # updates, so that is the natural frame boundary.
        from explorationRL.agents.skrl.wrappers import VisitationRecorder
        from explorationRL.tasks.direct.pointmaze.pointmaze_env_cfg import CELL_SIZE, POINTMAZE_V1_MAP

        rows, cols = len(POINTMAZE_V1_MAP), len(POINTMAZE_V1_MAP[0])
        x_max, y_max = cols / 2.0 * CELL_SIZE, rows / 2.0 * CELL_SIZE
        b = int(args_cli.visitation_bins_per_cell)
        env = VisitationRecorder(
            env, path=os.path.join(log_dir, "visitation.npz"),
            extent=(-x_max, x_max, -y_max, y_max), bins=(rows * b, cols * b),
            steps_per_update=int(agent_cfg["agent"]["rollouts"]),
            maze_map=POINTMAZE_V1_MAP,
        )
        print(f"[INFO] Recording visitation every {agent_cfg['agent']['rollouts']} steps.")

    # ── train ─────────────────────────────────────────────────────────────── #
    runner = ExplorationRunner(env, agent_cfg)
    if args_cli.checkpoint:
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
        runner.agent.load(args_cli.checkpoint)

    # Agents with an offline representation-learning phase (IRPO's ALLO) pretrain
    # it here, before the trainer takes over the env loop. No-op for other agents.
    if hasattr(runner.agent, "pretrain_extractor") and not args_cli.checkpoint:
        runner.agent.pretrain_extractor(env, tag=f"{args_cli.task}_seed{seed}")

    start = time.time()
    runner.run()
    print(f"[INFO] Training time: {round(time.time() - start, 2)} s")

    # One-episode deterministic eval → results/<EXP_RUN_TAG>/... for
    # scripts/aggregate_seeds.py. Sweeps skip it (they read trainer scalars).
    if not args_cli.skip_final_eval:
        write_seed_result(env, runner.agent, task=args_cli.task, algorithm=algorithm,
                          seed=seed, max_len=max_len, num_envs=num_envs)

    env.close()
    finish_wandb(args_cli)


if __name__ == "__main__":
    main()
    simulation_app.close()
    os._exit(0)

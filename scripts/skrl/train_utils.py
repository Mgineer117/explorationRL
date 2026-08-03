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
    """Mirror skrl's TensorBoard scalars into ``wandb.log``.

    ``wandb.init(sync_tensorboard=True)`` works by patching *torch's* /
    tensorboardX's ``SummaryWriter``. skrl >= 2.0 ships its own minimal writer
    (``skrl.utils.tensorboard.SummaryWriter``, note the keyword-only
    ``add_scalar(tag=, value=, timestep=)`` signature), which that patch never
    touches — so every scalar lands in the event file and *nothing* reaches W&B,
    leaving runs that show Media and no charts. Patch skrl's writer directly.

    Idempotent, and logs at the trainer timestep so scalars line up with the
    videos ``upload_videos_to_wandb`` logs at the same step.
    """
    try:
        from skrl.utils.tensorboard import SummaryWriter
    except ImportError:
        return
    if getattr(SummaryWriter, "_wandb_hooked", False):
        return
    original_add_scalar = SummaryWriter.add_scalar

    def add_scalar(self, *, tag: str, value: float, timestep: int) -> None:
        original_add_scalar(self, tag=tag, value=value, timestep=timestep)
        try:
            import wandb

            if wandb.run is not None:
                wandb.log({tag: value}, step=int(timestep))
        except Exception:  # noqa: BLE001 — logging must never fail training
            pass

    SummaryWriter.add_scalar = add_scalar
    SummaryWriter._wandb_hooked = True


def warmup_video_pipeline(env, num_renders: int = 16, max_polls: int = 32) -> None:
    """Build the rgb annotator against a warm renderer, and prove it produces pixels.

    ``DirectRLEnv.render()`` creates the render product and the ``rgb`` annotator
    together, on the first call. Both failure modes we hit come from that timing:

    * created at training step 0, before the RTX renderer emitted anything, the
      SDG graph is wired against empty buffers and ``omni.syntheticdata`` raises
      ``TypeError: Unable to write from unknown dtype, kind=f, size=0``;
    * created after only a couple of renders, the graph comes up *without* a
      valid ``LdrColorSD`` input (the ``SdRenderVarPtr missing valid input
      renderVar LdrColorSDhost`` warning). ``get_data()`` then returns an empty
      array forever, and Isaac Lab quietly substitutes zeros — every recorded
      frame is black, with no error anywhere.

    So: make the render product first, render enough frames that ``LdrColorSD``
    actually exists, only then attach the annotator, and poll until real pixels
    come out. If they never do, say so loudly — silent black videos are the whole
    problem being fixed here.
    """
    unwrapped = getattr(env, "unwrapped", env)
    try:
        import numpy as np
        import omni.replicator.core as rep

        resolution = unwrapped.cfg.viewer.resolution

        def build(camera, label: str):
            """Attach an rgb annotator to ``camera`` and report whether it yields pixels."""
            for _ in range(num_renders):
                unwrapped.sim.render()
            render_product = rep.create.render_product(camera, resolution)
            for _ in range(num_renders):
                unwrapped.sim.render()
            annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            annotator.attach([render_product])
            for _ in range(max_polls):
                unwrapped.sim.render()
                data = np.frombuffer(annotator.get_data(), dtype=np.uint8)
                if data.size and data.any():
                    print(f"[INFO] Video pipeline live on {label}.")
                    return render_product, annotator, True
            return render_product, annotator, False

        # The viewport camera is the cheap path, but headless runs have no
        # viewport driving /OmniverseKit_Persp, so its render var often stays
        # empty and every frame comes out black.
        render_product, annotator, ok = build(unwrapped.cfg.viewer.cam_prim_path, "the viewport camera")
        if not ok:
            print("[INFO] Viewport camera yielded no pixels (headless has no viewport) — "
                  "falling back to a dedicated recording camera.")
            camera = rep.create.camera(
                position=tuple(unwrapped.cfg.viewer.eye), look_at=tuple(unwrapped.cfg.viewer.lookat)
            )
            render_product, annotator, ok = build(camera, "a dedicated recording camera")
        if not ok:
            print("[WARN] No camera produced pixels — videos will be black. Verify the run has "
                  "--enable_cameras and that the GPU supports RTX rendering.")

        # Hand the warmed pair to Isaac Lab so render() reuses them instead of
        # lazily building its own cold ones.
        unwrapped._render_product = render_product
        unwrapped._rgb_annotator = annotator
    except Exception as exc:  # noqa: BLE001 — warm-up is advisory, not required
        print(f"[WARN] Video pipeline warm-up failed ({exc}); recording may be unstable.")
    make_render_robust(env)


def make_render_robust(env) -> None:
    """Keep ``render()`` alive across the long gaps between recorded clips.

    Replicator tears the rgb annotator off its render product when thousands of
    training steps pass with nothing rendering (``SdRenderVarPtr missing valid
    input renderVar LdrColorSDhost``), so the next capture dies with
    ``AnnotatorError: annotator is not attached to any render products``. Before
    each frame we re-attach a detached annotator, and if even that fails we drop
    the cached objects so Isaac Lab rebuilds them from scratch. A capture that
    still cannot produce a frame yields a black one — a missing video must never
    take down a training run that is otherwise fine.
    """
    unwrapped = getattr(env, "unwrapped", env)
    if getattr(unwrapped, "_render_is_robust", False):
        return
    original_render = unwrapped.render
    unwrapped._black_frame_warned = False

    def render(recompute: bool = False):
        annotator = getattr(unwrapped, "_rgb_annotator", None)
        if annotator is not None and not annotator.is_attached:
            try:
                annotator.attach([unwrapped._render_product])
            except Exception:  # noqa: BLE001 — fall back to a full rebuild
                for attr in ("_rgb_annotator", "_render_product"):
                    if hasattr(unwrapped, attr):
                        delattr(unwrapped, attr)
        import numpy as np

        try:
            frame = original_render(recompute=recompute)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Frame capture failed ({exc}); emitting a black frame.")
            width, height = unwrapped.cfg.viewer.resolution
            return np.zeros((height, width, 3), dtype=np.uint8)
        # Isaac Lab returns zeros instead of raising when the annotator has no
        # data, which is how a whole run of black videos goes unnoticed. Say it
        # once.
        if frame is not None and not unwrapped._black_frame_warned and not np.any(frame):
            unwrapped._black_frame_warned = True
            print("[WARN] Captured an all-black frame — the rgb annotator is not producing pixels.")
        return frame

    unwrapped.render = render
    unwrapped._render_is_robust = True


def upload_videos_to_wandb(env, video_folder: str):
    """Make every clip ``gym.wrappers.RecordVideo`` writes show up in W&B.

    W&B's own ``monitor_gym`` hook patches the recorder at ``wandb.init`` time,
    which is too late here: skrl only starts the run when the agent is built,
    long after the wrapper exists. So we hook ``stop_recording`` instead and log
    whatever new ``.mp4`` landed in ``video_folder``. Best-effort — a failed
    upload must never kill a training run.
    """
    import glob

    uploaded: set[str] = set()
    original_stop = env.stop_recording

    def stop_recording():
        original_stop()
        try:
            import wandb

            if wandb.run is None:
                return
            for path in sorted(glob.glob(os.path.join(video_folder, "*.mp4"))):
                if path in uploaded or os.path.getsize(path) == 0:
                    continue
                uploaded.add(path)
                # Same step axis as install_wandb_scalar_hook's scalars: W&B
                # requires monotonically increasing steps across ALL log calls,
                # and a video logged on its own counter would fight the charts.
                wandb.log({"video": wandb.Video(path, format="mp4")},
                          step=int(getattr(env, "step_id", 0)))
        except Exception:  # noqa: BLE001 — logging must never fail training
            pass

    env.stop_recording = stop_recording
    return env


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

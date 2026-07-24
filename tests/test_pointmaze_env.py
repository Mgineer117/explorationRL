"""Correctness checks for the PointMaze-v1 Isaac Lab environment.

Verifies that the Isaac port reproduces the gymnasium-robotics MuJoCo
``PointMaze`` it replaces: the point-mass force balance, the velocity clamp,
the sparse goal reward / termination, the observation layout, and that maze
walls actually block motion.

Run (needs Isaac Sim):
    python tests/test_pointmaze_env.py
Exits non-zero if any check fails.
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import explorationRL.tasks  # noqa: F401,E402
from explorationRL.tasks.direct.pointmaze.pointmaze_env_cfg import PointMazeEnvCfg  # noqa: E402

N = 4
cfg = PointMazeEnvCfg()
cfg.scene.num_envs = N
env = gym.make("PointMaze-v1", cfg=cfg).unwrapped

MASS, GEAR, DAMP, DT = cfg.point_mass, cfg.control_gear, cfg.joint_damping, cfg.sim.dt
VEL_CLIP = cfg.vel_clip
fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def act_dir(dx: float, dy: float) -> torch.Tensor:
    a = torch.zeros(N, 2, device=env.device)
    a[:, 0], a[:, 1] = dx, dy
    return a


# ── observation layout ────────────────────────────────────────────────────── #
obs, _ = env.reset()
o = obs["policy"]
check("obs shape == (N, 8)", tuple(o.shape) == (N, 8), f"got {tuple(o.shape)}")
check("action space == (2,)", env.action_space.shape[-1] == 2)
# layout [x, y, vx, vy, achieved_x, achieved_y, goal_x, goal_y]
check("obs achieved_goal mirrors position",
      torch.allclose(o[:, 0:2], o[:, 4:6], atol=1e-5))

# ── force balance: m*a = gear*u - c*v (one step from rest) ────────────────── #
env.reset()
obs, _, _, _, _ = env.step(act_dir(1.0, 0.0))
vx = obs["policy"][:, 2]
expected_v1 = (GEAR * 1.0 - DAMP * 0.0) / MASS * DT
check("v after 1 step matches MuJoCo force balance",
      torch.allclose(vx, torch.full_like(vx, expected_v1), rtol=0.02),
      f"expected≈{expected_v1:.5f}, got {vx[0].item():.5f}")

# ── velocity clamp — push -y, which is open space from the reset cell ─────── #
# The reset cell "r" is at (-1.5, 1.0); cells below it (rows 2 and 3 of the v1
# map) are free, so -y gives ~2.4 m of travel. Max accel is gear/mass ≈ 23.9
# m/s^2, so the clamp is reached in ~21 steps; 40 steps stays clear of the wall.
env.reset()
a = act_dir(0.0, -1.0)
for _ in range(40):
    obs, _, _, _, _ = env.step(a)
vy = obs["policy"][:, 3]
# MuJoCo's PointEnv.step clips the velocity BEFORE integrating, so the velocity
# OBSERVED after the step may exceed the clip by exactly one step of
# acceleration under the post-clip force: v <= clip + (gear - c*clip)/m * dt.
# (Measured: 5.0 + (100 - 1*5)/4.18879*0.01 = 5.2268.) Reproducing that overshoot
# is the point — it is what the MuJoCo env does.
overshoot = (GEAR - DAMP * VEL_CLIP) / MASS * DT
check("|v| respects the pre-step clamp (MuJoCo semantics)",
      bool((vy.abs() <= VEL_CLIP + overshoot + 1e-3).all()),
      f"max |vy| = {vy.abs().max().item():.4f} <= {VEL_CLIP + overshoot:.4f}")
check("sustained max force saturates at the clamp",
      bool((vy.abs() > 0.9 * VEL_CLIP).all()),
      f"|vy| = {vy.abs()[0].item():.4f} (clamp {VEL_CLIP})")

# ── sparse goal reward + termination (radius 0.45) ────────────────────────── #
env.reset()
pose = torch.zeros(N, 7, device=env.device)
pose[:, 0:2] = env._origin_xy + env.goal          # teleport onto the goal
pose[:, 2] = env._z0
pose[:, 3] = 1.0
env.point.write_root_pose_to_sim(pose)
env.point.write_root_velocity_to_sim(torch.zeros(N, 6, device=env.device))
obs, rew, term, trunc, _ = env.step(act_dir(0.0, 0.0))
check("reward == 1 at the goal", bool((rew > 0.5).all()), f"rew = {rew.tolist()}")
check("terminated at the goal", bool(term.all()))

env.reset()
obs, rew, term, trunc, _ = env.step(act_dir(0.0, 0.0))
check("no reward at the reset cell", bool((rew < 0.5).all()))
check("not terminated at the reset cell", bool((~term).all()))

# ── walls block motion ────────────────────────────────────────────────────── #
# From the reset cell, +x runs into the wall at map cell (1,2), whose near face
# is x = -1.0; the sphere (r=0.1) must stop at ~-1.1 rather than pass through.
env.reset()
a = act_dir(1.0, 0.0)
for _ in range(200):
    env.step(a)
end_x = env._pos_xy()[:, 0]
check("wall blocks motion in +x", bool((end_x < -1.0).all()),
      f"end x = {end_x[0].item():.3f} (wall face at -1.0)")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"), flush=True)
env.close()
simulation_app.close()
os._exit(0 if not fails else 1)

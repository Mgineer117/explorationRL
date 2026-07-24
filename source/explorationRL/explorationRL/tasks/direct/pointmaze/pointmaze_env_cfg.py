# Copyright (c) 2026, Minjae Cho.
# SPDX-License-Identifier: Apache-2.0

"""Config for ``PointMaze-v1`` — a force-actuated point mass in a maze.

Dynamics are matched to the gymnasium-robotics MuJoCo ``PointMaze`` (the env this
repo used previously via ``get_env`` + ``MazeWrapper``):

    MuJoCo point.xml                         →  this Isaac env
    ─────────────────────────────────────────────────────────────────────────
    timestep 0.01, frame_skip 1              →  sim.dt = 0.01, decimation = 1
    gravity 0                                →  point rigid body: disable_gravity
    2 slide joints, damping 1                →  viscous force  F_damp = -c * v
    motor gear 100, ctrlrange [-1, 1]        →  F_ctrl = 100 * clip(action, -1, 1)
    sphere size 0.1, density 1000            →  mass = 1000 * (4/3)π(0.1)^3 ≈ 4.18879 kg
    _clip_velocity to [-5, 5]                →  |v| clamped to 5 every control step
    maze_size_scaling 1, maze_height 0.4     →  cell_size 1.0, wall height 0.4

The reward / termination match ``MazeEnv`` (sparse): reward 1.0 and termination
when the ball is within 0.45 m of the goal (``continuing_task=False``).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

# ── PointMaze-v1 layout (identical to POINTMAZE_MAPS["pointmaze-v1"]) ──────── #
# 1 = wall, 0 = free, "r" = agent reset cell, "g" = goal cell.
POINTMAZE_V1_MAP = [
    [1, 1, 1, 1, 1, 1],
    [1, "r", 1, "g", 0, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1],
]

# Physical constants shared with the env (see the module docstring for the
# MuJoCo correspondence).
POINT_MASS = 4.18879  # 1000 * (4/3) * pi * 0.1**3
POINT_RADIUS = 0.1
CONTROL_GEAR = 100.0  # MuJoCo motor gear -> force = gear * ctrl
JOINT_DAMPING = 1.0  # MuJoCo slide-joint damping coefficient c
VEL_CLIP = 5.0  # MuJoCo _clip_velocity range [-5, 5]
CELL_SIZE = 1.0  # maze_size_scaling
MAZE_HEIGHT = 0.4
GOAL_THRESHOLD = 0.45  # MazeEnv success / termination radius
RESET_NOISE = 0.25  # uniform [-0.25, 0.25] added to reset & goal cell centers


@configclass
class PointMazeEnvCfg(DirectRLEnvCfg):
    # ── env ───────────────────────────────────────────────────────────────── #
    decimation = 1
    # 500 control steps * 0.01 s  = 5.0 s  (EPI_LENGTH["pointmaze-v1"] = 500).
    episode_length_s = 5.0
    # action = normalised force in [-1, 1] per axis, scaled by CONTROL_GEAR [N].
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    # obs layout matches the old MazeWrapper: [x, y, vx, vy, x, y, gx, gy].
    observation_space = 8
    state_space = 0

    # ── simulation ────────────────────────────────────────────────────────── #
    # dt = 0.01 & decimation = 1 mirror MuJoCo's timestep 0.01, frame_skip 1.
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, 0.0),  # MuJoCo point.xml sets gravity to 0
        physx=PhysxCfg(enable_external_forces_every_iteration=True),
    )

    # ── scene ─────────────────────────────────────────────────────────────── #
    # env_spacing must clear the maze footprint (cols * cell) with margin so
    # neighbouring mazes never overlap.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=10.0, replicate_physics=True
    )

    # ── point mass (rigid body) ───────────────────────────────────────────── #
    # A single sphere rigid body actuated by external XY force. Gravity and
    # damping are disabled on the body — damping is applied explicitly as
    # F_damp = -c*v in the env so the MuJoCo force balance is transparent.
    point_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Point",
        spawn=sim_utils.SphereCfg(
            radius=POINT_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=100.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=POINT_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.2)),
        ),
        # Spawn at the v1 reset ("r") cell centre so the prototype body is not
        # created inside a wall; the real per-env placement happens in _reset_idx.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.5, 1.0, MAZE_HEIGHT / 2.0)),
    )

    # ── maze / task parameters ────────────────────────────────────────────── #
    maze_map = POINTMAZE_V1_MAP
    cell_size = CELL_SIZE
    maze_height = MAZE_HEIGHT
    point_mass = POINT_MASS
    control_gear = CONTROL_GEAR
    joint_damping = JOINT_DAMPING
    vel_clip = VEL_CLIP
    goal_threshold = GOAL_THRESHOLD
    reset_noise = RESET_NOISE
    reward_type = "sparse"  # "sparse" | "dense" (matches MazeEnv.compute_reward)

# Copyright (c) 2026, Minjae Cho.
# SPDX-License-Identifier: Apache-2.0

"""``PointMaze-v1`` — a force-actuated point mass in a maze (Isaac Lab DirectRLEnv).

See ``pointmaze_env_cfg.py`` for the point-by-point correspondence to the
gymnasium-robotics MuJoCo ``PointMaze`` this reproduces. In short, each control
step (matching ``PointEnv.step``):

1. clip the action to [-1, 1];
2. clamp the current velocity to [-5, 5]  (MuJoCo ``_clip_velocity``);
3. apply force  F = gear * action - c * v  (motor gear + slide-joint damping);
4. integrate one 0.01 s physics step under that force, resolving wall contacts.

Observation, reward and termination follow the old ``MazeWrapper`` / ``MazeEnv``:
obs ``[x, y, vx, vy, x, y, gx, gy]``, sparse reward + termination within 0.45 m
of the goal.
"""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .pointmaze_env_cfg import PointMazeEnvCfg


def _maze_geometry(maze_map, cell_size):
    """Return (wall_centers, reset_cell_xy, goal_cell_xy) in local maze coords.

    Uses the exact cell→(x,y) mapping of gymnasium-robotics ``Maze``:
        x = (j + 0.5) * scaling - x_map_center
        y = y_map_center - (i + 0.5) * scaling
    with x_map_center = cols/2 * scaling, y_map_center = rows/2 * scaling.
    """
    rows = len(maze_map)
    cols = len(maze_map[0])
    x_map_center = cols / 2.0 * cell_size
    y_map_center = rows / 2.0 * cell_size

    walls: list[tuple[float, float]] = []
    reset_xy: tuple[float, float] | None = None
    goal_xy: tuple[float, float] | None = None
    for i in range(rows):
        for j in range(cols):
            val = maze_map[i][j]
            x = (j + 0.5) * cell_size - x_map_center
            y = y_map_center - (i + 0.5) * cell_size
            if val == 1:
                walls.append((x, y))
            elif val == "r":
                reset_xy = (x, y)
            elif val == "g":
                goal_xy = (x, y)
    if reset_xy is None or goal_xy is None:
        raise ValueError("PointMaze map must contain exactly one 'r' and one 'g' cell.")
    return walls, reset_xy, goal_xy


class PointMazeEnv(DirectRLEnv):
    cfg: PointMazeEnvCfg

    def __init__(self, cfg: PointMazeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Maze geometry (local coordinates, shared across envs).
        walls, reset_xy, goal_xy = _maze_geometry(cfg.maze_map, cfg.cell_size)
        self._reset_xy = torch.tensor(reset_xy, device=self.device, dtype=torch.float32)
        self._goal_cell_xy = torch.tensor(goal_xy, device=self.device, dtype=torch.float32)
        self._z0 = float(cfg.maze_height / 2.0)

        # Per-env goal (local coords), sampled every reset.
        self.goal = self._goal_cell_xy.unsqueeze(0).repeat(self.num_envs, 1).clone()
        # Buffered clipped action.
        self.actions = torch.zeros(self.num_envs, 2, device=self.device)
        # env origin (xy) to convert world <-> local coordinates.
        self._origin_xy = self.scene.env_origins[:, :2]

        # Dynamics constants as tensors/scalars.
        self._gear = float(cfg.control_gear)
        self._damping = float(cfg.joint_damping)
        self._vel_clip = float(cfg.vel_clip)
        self._goal_thresh = float(cfg.goal_threshold)
        self._reset_noise = float(cfg.reset_noise)

        # No wrapping-angle dimensions in this env (consumed by train_utils'
        # angle-embedding injection for the ported algorithms).
        self.angle_idx: list[int] = []

    # ── scene ─────────────────────────────────────────────────────────────── #
    def _setup_scene(self):
        self.point = RigidObject(self.cfg.point_cfg)

        # Ground plane (shared).
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Maze walls: static colliders spawned under the env-0 template, then
        # cloned to every env by clone_environments below.
        walls, _, _ = _maze_geometry(self.cfg.maze_map, self.cfg.cell_size)
        wall_cfg = sim_utils.CuboidCfg(
            size=(self.cfg.cell_size, self.cfg.cell_size, self.cfg.maze_height),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.35, 0.2)),
        )
        for k, (wx, wy) in enumerate(walls):
            wall_cfg.func(
                f"/World/envs/env_0/Maze/Wall_{k}",
                wall_cfg,
                translation=(wx, wy, self.cfg.maze_height / 2.0),
            )

        # Clone & (on CPU) filter cross-env collisions.
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.rigid_objects["point"] = self.point

        light_cfg = sim_utils.DistantLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/Light", light_cfg)

    # ── action → force ────────────────────────────────────────────────────── #
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # MuJoCo PointEnv.step clips the action to [-1, 1] first.
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        vel = self.point.data.root_lin_vel_w  # (N, 3), world == local (origins fixed)
        v_xy = vel[:, :2]
        # MuJoCo _clip_velocity: clamp current velocity BEFORE stepping.
        v_xy_clipped = torch.clamp(v_xy, -self._vel_clip, self._vel_clip)

        # Write back the clamped, planar velocity (zero vz + angular) so the body
        # stays a true 2-DoF point mass, then let the solver integrate contacts.
        root_vel = torch.zeros(self.num_envs, 6, device=self.device)
        root_vel[:, 0:2] = v_xy_clipped
        self.point.write_root_velocity_to_sim(root_vel)

        # Force balance identical to the MuJoCo slide joint:
        #   m * a = gear * ctrl - c * v
        force_xy = self._gear * self.actions - self._damping * v_xy_clipped
        forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        forces[:, 0, 0:2] = force_xy
        torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.point.set_external_force_and_torque(forces, torques)

    # ── observation / reward / done ───────────────────────────────────────── #
    def _pos_xy(self) -> torch.Tensor:
        return self.point.data.root_pos_w[:, :2] - self._origin_xy

    def _get_observations(self) -> dict:
        pos = self._pos_xy()
        vel = self.point.data.root_lin_vel_w[:, :2]
        # [x, y, vx, vy, achieved_x, achieved_y, goal_x, goal_y] — matches the old
        # MazeWrapper obs (achieved_goal == ball xy).
        obs = torch.cat([pos, vel, pos, self.goal], dim=-1)
        return {"policy": obs}

    def _goal_distance(self) -> torch.Tensor:
        return torch.linalg.norm(self._pos_xy() - self.goal, dim=-1)

    def _get_rewards(self) -> torch.Tensor:
        dist = self._goal_distance()
        if self.cfg.reward_type == "dense":
            return torch.exp(-dist)
        return (dist <= self._goal_thresh).float()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        reached = self._goal_distance() <= self._goal_thresh  # continuing_task=False
        return reached, time_out

    # ── reset ─────────────────────────────────────────────────────────────── #
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)

        n = len(env_ids)
        noise = lambda: (torch.rand(n, 2, device=self.device) * 2.0 - 1.0) * self._reset_noise

        start_xy = self._reset_xy.unsqueeze(0) + noise()
        self.goal[env_ids] = self._goal_cell_xy.unsqueeze(0) + noise()

        # World pose: env origin + local start, identity orientation, z = z0.
        pose = torch.zeros(n, 7, device=self.device)
        pose[:, 0:2] = self._origin_xy[env_ids] + start_xy
        pose[:, 2] = self._z0
        pose[:, 3] = 1.0  # quat w (identity)
        self.point.write_root_pose_to_sim(pose, env_ids)
        self.point.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)

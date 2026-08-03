"""Environment wrappers shared by the training scripts.

``PassthroughWrapper`` exists because skrl's base ``Wrapper`` resolves
``observation_space``/``num_envs``/... against ``env.unwrapped``. For an Isaac Lab
env that is the *batched* space (``num_envs * obs_dim``) — precisely what the
inner ``SkrlVecEnvWrapper`` normalizes away — so a wrapper stacked on top of it
must delegate to the env it wraps, not to the raw env underneath.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from skrl.envs.wrappers.torch.base import Wrapper


class PassthroughWrapper(Wrapper):
    """skrl env wrapper that is transparent: spaces and counts come from the
    wrapped env. Subclasses override ``reset``/``step``."""

    @property
    def device(self):
        return self._env.device

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def num_agents(self) -> int:
        return self._env.num_agents

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def state_space(self):
        return self._env.state_space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        return self._env.reset()

    def step(self, actions):
        return self._env.step(actions)

    def state(self):
        return self._env.state()

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        self._env.close()


class VisitationRecorder(PassthroughWrapper):
    """Records where the policy goes, one occupancy histogram per policy update.

    Every step the agent's 2D position (``observations[:, pos_idx]``) is binned
    into a ``(rows, cols)`` grid; every ``steps_per_update`` steps that histogram
    is closed off as one frame and a fresh one starts. The result is a
    ``(num_updates, rows, cols)`` stack — the raw material for an animation of how
    exploration evolves *during* training, as opposed to replaying checkpoints
    afterwards.

    Frames are the occupancy of a single update, not cumulative, so a policy that
    stops exploring shows a shrinking frame rather than a frozen inherited one.
    """

    def __init__(self, env, *, path: str, extent: tuple[float, float, float, float],
                 bins: tuple[int, int], steps_per_update: int, pos_idx: tuple[int, int] = (0, 1),
                 maze_map: list | None = None):
        super().__init__(env)
        self.path = path
        # Stored alongside the frames so the animation script needs nothing from
        # Isaac Lab: 1 = wall, 0 = free, 2 = start ("r"), 3 = goal ("g").
        codes = {1: 1, 0: 0, "r": 2, "g": 3}
        self.maze_map = (np.asarray([[codes.get(c, 0) for c in row] for row in maze_map],
                                    dtype=np.int8) if maze_map is not None else None)
        self.extent = extent  # (x_min, x_max, y_min, y_max)
        self.bins = bins
        self.steps_per_update = max(1, int(steps_per_update))
        self.pos_idx = pos_idx

        self._frames: list[np.ndarray] = []
        self._current = np.zeros(bins, dtype=np.float32)
        self._steps = 0

    def _accumulate(self, observations: torch.Tensor) -> None:
        xy = observations[:, self.pos_idx].detach().cpu().numpy()
        x_min, x_max, y_min, y_max = self.extent
        h, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=self.bins,  # (row=y, col=x)
                                 range=[[y_min, y_max], [x_min, x_max]])
        self._current += h.astype(np.float32)

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self._env.step(actions)
        self._accumulate(observations)
        self._steps += 1
        if self._steps % self.steps_per_update == 0:
            self._frames.append(self._current)
            self._current = np.zeros(self.bins, dtype=np.float32)
        return observations, rewards, terminated, truncated, infos

    def save(self) -> str | None:
        """Write the frame stack to ``path`` (no-op if nothing was recorded)."""
        if not self._frames:
            return None
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        arrays = {
            "frames": np.stack(self._frames),
            "extent": np.asarray(self.extent, dtype=np.float32),
            "steps_per_update": np.asarray(self.steps_per_update),
        }
        if self.maze_map is not None:
            arrays["maze_map"] = self.maze_map
        np.savez_compressed(self.path, **arrays)
        print(f"[INFO] Visitation: {len(self._frames)} frames -> {self.path}")
        return self.path

    def close(self) -> None:
        self.save()
        self._env.close()

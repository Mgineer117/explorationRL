"""Intrinsic reward functions for the option-based agents (IRPO, HRL).

Port of the ``*IntRewardFunctions`` family in ``utils/intrinsic_rewards.py``.

Each *option* is a signed direction in the learned representation. For option
``n`` the direction is ``(n // 2 + 1, 2 * (n % 2) - 1)`` — eigenvector index and
sign — so options come in ``+/-`` pairs over eigenvector 1, 2, ... Index 0 is
skipped because the first Laplacian eigenfunction is constant (it carries no
directional information; the ALLO unit test shows its feature std collapsing to
~0 while the others orthonormalise).

The reward for option ``n`` on a transition is the signed change of that
coordinate,

    r_n(s, s') = sign_n * ( phi_{idx_n}(s') - phi_{idx_n}(s) ),

normalised by a running standard deviation (variance only — the sign carries the
direction and must not be centred away).
"""

from __future__ import annotations

from collections import deque

import gymnasium
import torch
import torch.nn as nn

from explorationRL.agents.skrl.wrappers import PassthroughWrapper


class RunningVariance:
    """Running variance used to scale intrinsic rewards (no mean subtraction)."""

    def __init__(self, shape, device, epsilon: float = 1e-4):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch_mean, batch_var, batch_count = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = total

    def normalize_var_only(self, x: torch.Tensor) -> torch.Tensor:
        return x / (torch.sqrt(self.var) + 1e-8)


def option_directions(num_options: int) -> list[tuple[int, int]]:
    """``[(eigenvector index, sign), ...]`` for ``num_options`` options."""
    return [(n // 2 + 1, 2 * (n % 2) - 1) for n in range(num_options)]


class ALLOIntrinsicRewards(nn.Module):
    """Signed eigenvector-direction rewards from an :class:`~..extractors.ALLO`."""

    def __init__(self, extractor, num_options: int, *, use_difference: bool = True,
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.extractor = extractor
        self.num_options = int(num_options)
        self.use_difference = use_difference
        self.directions = option_directions(self.num_options)
        self._indices = [d[0] for d in self.directions]
        self._signs = torch.tensor([float(d[1]) for d in self.directions], device=device)
        self.reward_rms = RunningVariance((self.num_options,), device=device)
        self.device = device

        feature_dim = int(getattr(extractor, "d", 0))
        if feature_dim and max(self._indices) >= feature_dim:
            raise ValueError(
                f"num_options={num_options} needs eigenvector index {max(self._indices)}, "
                f"but the extractor only has feature_dim={feature_dim}. Increase "
                f"extractor feature_dim to at least {max(self._indices) + 1}."
            )

    @torch.no_grad()
    def forward(self, observations: torch.Tensor, next_observations: torch.Tensor,
                update_stats: bool = True) -> torch.Tensor:
        """Return ``(batch, num_options)`` intrinsic rewards. ``update_stats=False``
        leaves the normalizer untouched, so off-policy/diagnostic calls neither
        move nor are moved by the training-time scale."""
        if self.use_difference:
            delta = self.extractor(next_observations) - self.extractor(observations)
        else:
            delta = self.extractor(observations)
        rewards = delta[:, self._indices] * self._signs
        if update_stats:
            self.reward_rms.update(rewards)
        return self.reward_rms.normalize_var_only(rewards)


class RandomIntrinsicRewards(ALLOIntrinsicRewards):
    """Ablation: identical machinery over a FROZEN random encoder.

    This is the ``RandomIntRewardFunctions`` baseline — it isolates how much of
    IRPO's benefit comes from the *learned* Laplacian structure versus merely
    having several distinct directions to explore along.
    """

    def __init__(self, extractor, num_options: int, **kwargs):
        super().__init__(extractor, num_options, **kwargs)
        for p in self.extractor.parameters():
            p.requires_grad_(False)


def _mlp(in_dim: int, hidden: list[int], out_dim: int, activation) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, int(h)), activation()]
        d = int(h)
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class DRNDNovelty(nn.Module):
    """DRND novelty bonus: a predictor distilled toward an ensemble of frozen
    random targets. See ``..drnd`` for the formulation; this holds only the
    networks, the bonus, and the distillation step, so both the DRND *agent* and
    the environment wrapper below can share one implementation.
    """

    def __init__(self, observation_dim: int, *, num_targets: int = 5, embedding_dim: int = 64,
                 predictor_hidden: list | None = None, target_hidden: list | None = None,
                 alpha: float = 0.9, update_proportion: float = 0.25,
                 learning_rate: float = 3e-4, device: str | torch.device = "cpu"):
        super().__init__()
        self.alpha = float(alpha)
        self.update_proportion = float(update_proportion)
        self.num_targets = int(num_targets)
        self.predictor = _mlp(observation_dim, list(predictor_hidden or [512, 512, 512]),
                              embedding_dim, nn.ReLU).to(device)
        self.targets = nn.ModuleList([
            _mlp(observation_dim, list(target_hidden or [128, 128]), embedding_dim, nn.Tanh)
            for _ in range(self.num_targets)
        ]).to(device)
        for p in self.targets.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        self.reward_rms = RunningVariance((1,), device=device)
        self.to(device)

    @torch.no_grad()
    def forward(self, observations: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        """``(batch, 1)`` normalized novelty of ``observations``. ``update_stats=False``
        leaves the normalizer untouched (see :meth:`ALLOIntrinsicRewards.forward`)."""
        pred = self.predictor(observations)
        targets = torch.stack([t(observations) for t in self.targets], dim=0)

        mu = targets.mean(dim=0)
        b2_mom = (targets**2).mean(dim=0)

        b1 = self.alpha * ((pred - mu) ** 2).sum(dim=1, keepdim=True)
        var = torch.clamp(b2_mom - mu**2, min=1e-6)
        b2 = (1.0 - self.alpha) * torch.sqrt(
            torch.clamp(torch.abs(pred**2 - mu**2) / var, 1e-6, 1.0)
        ).sum(dim=-1, keepdim=True)

        novelty = b1 + b2
        if update_stats:
            self.reward_rms.update(novelty)
        return self.reward_rms.normalize_var_only(novelty)

    def loss(self, observations: torch.Tensor) -> torch.Tensor:
        """Distillation loss of the predictor onto a randomly chosen target, over a
        random ``update_proportion`` subset of the batch."""
        pred = self.predictor(observations)
        n = observations.shape[0]
        with torch.no_grad():
            targets = torch.stack([t(observations) for t in self.targets], dim=0)
            idx = torch.randint(high=self.num_targets, size=(n,), device=observations.device)
            target = targets[idx, torch.arange(n, device=observations.device), :]
        per_sample = ((pred - target) ** 2).mean(dim=-1)
        mask = (torch.rand(n, device=observations.device) < self.update_proportion).float()
        return (per_sample * mask).sum() / torch.clamp(mask.sum(), min=1.0)

    def learn(self, observations: torch.Tensor) -> float:
        """One optimizer step on :meth:`loss`; returns the loss value."""
        loss = self.loss(observations)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class GeodesicGuidance(nn.Module):
    """Geometry-aware guidance toward the goal (``int_reward="best"``).

    The potential is the *geodesic* distance to the goal — shortest path through
    the corridors, computed once by Dijkstra on a fine grid of the maze — not the
    Euclidean distance, which points straight through walls and is actively
    misleading in a maze (in PointMaze-v1 the start is 2 cells from the goal in
    a straight line but 8 cells away around the corridor).

    The reward is the potential difference

        r(s, s') = d_geo(s) - d_geo(s')          [metres of progress]

    i.e. potential-based shaping with ``Phi = -d_geo``: it is positive exactly
    when the step made real progress along the corridor, so it gradually guides
    the agent to the goal without inventing an optimum of its own.
    """

    def __init__(self, maze_map, cell_size: float, *, bins_per_cell: int = 16,
                 pos_idx: tuple[int, int] = (0, 1), device: str | torch.device = "cpu"):
        super().__init__()
        import heapq
        import math

        rows, cols = len(maze_map), len(maze_map[0])
        b = int(bins_per_cell)
        h = float(cell_size) / b
        R, C = rows * b, cols * b
        self.pos_idx = pos_idx
        self.device = device
        self.h = h
        self.x0 = -cols / 2.0 * float(cell_size)  # left edge, local coords
        self.y1 = rows / 2.0 * float(cell_size)  # top edge

        free = [[maze_map[I // b][J // b] != 1 for J in range(C)] for I in range(R)]
        goal = [(i, j) for i in range(rows) for j in range(cols) if maze_map[i][j] == "g"]
        if not goal:
            raise ValueError("maze_map has no 'g' cell — cannot build a goal potential.")
        gi, gj = goal[0]

        INF = float("inf")
        dist = [[INF] * C for _ in range(R)]
        heap = []
        for I in range(gi * b, (gi + 1) * b):  # the whole goal cell is "arrived"
            for J in range(gj * b, (gj + 1) * b):
                dist[I][J] = 0.0
                heap.append((0.0, I, J))
        heapq.heapify(heap)
        nbrs = [(di, dj, math.hypot(di, dj) * h) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                if (di, dj) != (0, 0)]
        while heap:
            d, I, J = heapq.heappop(heap)
            if d > dist[I][J]:
                continue
            for di, dj, w in nbrs:
                I2, J2 = I + di, J + dj
                if not (0 <= I2 < R and 0 <= J2 < C) or not free[I2][J2]:
                    continue
                if di and dj and not (free[I][J2] and free[I2][J]):
                    continue  # no cutting diagonally through a wall corner
                if d + w < dist[I2][J2]:
                    dist[I2][J2] = d + w
                    heapq.heappush(heap, (d + w, I2, J2))

        # ponytail: walls are not inflated by the ball radius (0.1 m vs 1 m cells);
        # inflate `free` if the agent ever clips a corner and the potential jumps.
        finite = [d for row in dist for d in row if d < INF]
        fill = max(finite) * 1.5 if finite else 0.0
        self.register_buffer("dist", torch.tensor(
            [[fill if d == INF else d for d in row] for row in dist],
            dtype=torch.float32, device=device))
        self.reward_rms = RunningVariance((1,), device=device)
        self.to(device)

    def potential(self, observations: torch.Tensor) -> torch.Tensor:
        """``(batch,)`` geodesic distance-to-goal at the agent's xy."""
        xy = observations[:, self.pos_idx]
        R, C = self.dist.shape
        J = ((xy[:, 0] - self.x0) / self.h).long().clamp_(0, C - 1)
        I = ((self.y1 - xy[:, 1]) / self.h).long().clamp_(0, R - 1)
        return self.dist[I, J]

    @torch.no_grad()
    def forward(self, observations: torch.Tensor, next_observations: torch.Tensor,
                update_stats: bool = True) -> torch.Tensor:
        progress = (self.potential(observations) - self.potential(next_observations)).unsqueeze(-1)
        if update_stats:
            self.reward_rms.update(progress)
        return self.reward_rms.normalize_var_only(progress)


class IntrinsicRewardWrapper(PassthroughWrapper):
    """Environment wrapper that computes an intrinsic reward per step and hands it
    back in ``infos["int_reward"]`` (shape ``(num_envs, 1)``), alongside the
    unchanged environment reward. The reward model lives here and is trained here,
    on the same stream the agent is consuming — agents only read the number.

    ``int_reward``:

    * ``None``  — no model, no wrapping (see :func:`wrap_intrinsic`).
    * ``drnd``  — :class:`DRNDNovelty`; the predictor is distilled every
      ``train_every`` steps on the observations seen since the last update.
    * ``best``  — :class:`GeodesicGuidance`: analytic geometry-aware progress
      toward the goal (no model, nothing to train).
    * ``allo``  — an online-trained :class:`~...extractors.ALLO` representation;
      ``option`` selects which of the ``num_options`` signed eigenvector
      directions is returned.

    ALLO is trained *online* here (``allo_epochs`` updates every ``train_every``
    steps on a rolling ``buffer_steps`` window), unlike IRPO's offline-pretrained,
    then frozen extractor — a wrapper has no separate collection phase. Early
    intrinsic rewards are therefore from a barely-trained representation.
    """

    def __init__(self, env, int_reward: str, *, option: int = 0, num_options: int = 2,
                 train_every: int = 64, buffer_steps: int = 256, allo_epochs: int = 8,
                 allo_cfg=None, drnd_kwargs: dict | None = None, best_kwargs: dict | None = None):
        super().__init__(env)
        self.int_reward = str(int_reward).lower()
        self.option = int(option)
        self.train_every = int(train_every)
        self.allo_epochs = int(allo_epochs)
        obs_dim = int(gymnasium.spaces.flatdim(env.observation_space))

        if self.int_reward == "drnd":
            self.model = DRNDNovelty(obs_dim, device=self.device, **(drnd_kwargs or {}))
        elif self.int_reward == "best":
            kwargs = dict(best_kwargs or {})
            if "maze_map" not in kwargs:  # read the layout off the env being wrapped
                cfg = getattr(getattr(env, "unwrapped", env), "cfg", None)
                kwargs["maze_map"] = getattr(cfg, "maze_map")
                kwargs.setdefault("cell_size", float(getattr(cfg, "cell_size", 1.0)))
            self.model = GeodesicGuidance(device=self.device, **kwargs)
        elif self.int_reward == "allo":
            from explorationRL.extractors import ALLO, ALLO_CFG

            if not 0 <= self.option < int(num_options):
                raise ValueError(f"option {self.option} outside [0, {num_options})")
            cfg = allo_cfg or ALLO_CFG()
            cfg.feature_dim = max(int(cfg.feature_dim), int(num_options) // 2 + 2)
            self.extractor = ALLO(observation_dim=obs_dim, cfg=cfg, device=self.device)
            self.model = ALLOIntrinsicRewards(self.extractor, int(num_options), device=self.device)
        else:
            raise ValueError(
                f"unknown int_reward {int_reward!r} (use None | 'drnd' | 'allo' | 'best')")

        self._obs_buf: deque = deque(maxlen=int(buffer_steps))
        self._done_buf: deque = deque(maxlen=int(buffer_steps))
        self._prev_observations: torch.Tensor | None = None
        self._steps = 0

    # ── intrinsic reward / model training ─────────────────────────────────── #
    @torch.no_grad()
    def _intrinsic(self, observations: torch.Tensor, next_observations: torch.Tensor) -> torch.Tensor:
        if self.int_reward == "drnd":
            return self.model(next_observations)
        if self.int_reward == "best":
            return self.model(observations, next_observations)  # (N, 1)
        rewards = self.model(observations, next_observations)  # (N, num_options)
        return rewards[:, self.option: self.option + 1]

    @torch.enable_grad()
    def _train(self) -> None:
        # enable_grad: skrl's trainer runs the whole env-interaction loop inside
        # `torch.no_grad()`, and this trains a model from inside `step()`.
        if not self._obs_buf or self.int_reward == "best":  # "best" is analytic, nothing to fit
            return
        if self.int_reward == "drnd":
            # Only the steps since the last update: the predictor should chase what
            # was *just* visited, and re-distilling on old states flattens the bonus.
            recent = list(self._obs_buf)[-self.train_every:]
            self.model.learn(torch.cat(recent, dim=0))
            return
        observations = torch.stack(list(self._obs_buf), dim=0)  # (T, N, obs_dim)
        dones = torch.stack(list(self._done_buf), dim=0)  # (T, N)
        for _ in range(self.allo_epochs):
            self.extractor.learn(observations, dones)

    # ── skrl env API ──────────────────────────────────────────────────────── #
    def _with_int_reward(self, infos, int_reward: torch.Tensor):
        if not isinstance(infos, dict):
            infos = {}
        infos["int_reward"] = int_reward
        return infos

    def reset(self):
        observations, infos = self._env.reset()
        self._prev_observations = observations
        zeros = torch.zeros(observations.shape[0], 1, device=self.device)
        return observations, self._with_int_reward(infos, zeros)

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self._env.step(actions)
        if self._prev_observations is None:
            self._prev_observations = observations

        int_reward = self._intrinsic(self._prev_observations, observations)
        self._obs_buf.append(self._prev_observations)
        self._done_buf.append((terminated.bool() | truncated.bool()).reshape(-1))
        self._prev_observations = observations

        self._steps += 1
        if self._steps % self.train_every == 0:
            self._train()

        return observations, rewards, terminated, truncated, self._with_int_reward(infos, int_reward)


def wrap_intrinsic(env, int_reward: str | None, **kwargs):
    """``env`` unchanged when ``int_reward`` is ``None``, else wrapped."""
    if int_reward is None or str(int_reward).lower() in ("", "none"):
        return env
    return IntrinsicRewardWrapper(env, int_reward, **kwargs)

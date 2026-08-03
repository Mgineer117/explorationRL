"""AGA — Asymmetric Geometry-Adaptive preconditioning, as a skrl agent.

Steers exploration by *reshaping* the realized PPO update rather than adding an
intrinsic reward: every update, a real multi-epoch PPO step is taken (delegated
to skrl's own ``PPO.update``, so the inner loop is bit-for-bit the same as plain
PPO), then rolled back to a diff ``delta_ppo``. A single-pass REINFORCE gradient
on the same on-policy batch gives a unit exploration direction ``v_hat`` (target:
maze doorway cells by default). ``delta_ppo`` is reweighted by a rank-1 positive
definite map whose scale flips sign with ``<v_hat, delta_ppo>`` (asymmetric —
this is what gives a systematic directional bias while staying strictly inside
the PD cone, so every task fixed point stays a fixed point), then applied with a
linearly annealed step size.

Only ``precond_mode="direct"`` is implemented — the hypernetwork variant's
meta-gradient regression (predicting P from a state-independent input against a
fresh-noise-every-update target) was measured to be ill-posed upstream and never
learns past an isotropic rescale.
"""

from __future__ import annotations

import dataclasses

import torch

from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG

from explorationRL.agents.skrl.common import flat_params, set_flat_params
from explorationRL.tasks.direct.pointmaze.pointmaze_env_cfg import CELL_SIZE, POINTMAZE_V1_MAP


@dataclasses.dataclass(kw_only=True)
class AGA_CFG(PPO_CFG):
    """PPO config + the AGA preconditioning knobs."""

    precond_beta: float = 2.0
    """Rank-1 scale applied when ``<v_hat, delta_ppo> > 0``."""

    precond_damp: float = 0.9
    """Rank-1 scale magnitude applied when ``<v_hat, delta_ppo> < 0`` (``[0, 1)``,
    so ``P = I - damp * v_hat v_hat^T`` stays strictly positive definite)."""

    alpha_start: float = 1.0
    alpha_end: float = 0.3
    alpha_anneal_updates: int = 300
    """Linear anneal of the preconditioned step-size multiplier, in units of
    ``update()`` calls."""

    target_dist: str = "bottleneck"
    """Exploration target ``v_hat`` is fit to: ``bottleneck`` | ``uniform`` |
    ``far_from_start`` (per-cell, maze-specific) or ``intrinsic`` — the per-step
    intrinsic reward supplied by :class:`~.intrinsic.IntrinsicRewardWrapper` in
    ``infos["int_reward"]`` (DRND novelty or an ALLO eigenvector direction), which
    needs no maze layout."""

    meta_gamma: float = 0.99
    """Discount for the ``v_hat`` REINFORCE return-to-go (independent of the task
    discount, ``discount_factor``)."""

    novelty_coef: float = 0.0
    """Count-based novelty bonus (``1/sqrt(visit_count)``, centered) folded into
    the PPO inner-loop reward. ``0`` (the validated setting) means all exploration
    pressure comes from the preconditioner, not the task objective."""

    dense_coef: float = 0.0
    """Potential-based dense shaping (Ng, Harada & Russell 1999), ``Phi(s) =
    -||achieved_goal - goal||``, folded into the PPO inner-loop reward on top of
    the true sparse task reward. ``0`` disables it — the task's true (eval/logged)
    return stays the sparse env reward either way, this only shapes what the PPO
    inner loop trains on."""

    maze_map: list = dataclasses.field(default_factory=lambda: POINTMAZE_V1_MAP)
    cell_size: float = CELL_SIZE
    """Maze layout the per-cell target reward is computed over."""


class _MazeCells:
    """Grid <-> reachable-cell-index bookkeeping, plus the two static per-cell
    target rewards (``bottleneck``, ``far_from_start``). Local-coordinate formulas
    match ``pointmaze_env._maze_geometry``."""

    def __init__(self, maze_map: list, cell_size: float, device):
        self.rows, self.cols = len(maze_map), len(maze_map[0])
        self.cell_size = float(cell_size)
        self.x_map_center = self.cols / 2.0 * self.cell_size
        self.y_map_center = self.rows / 2.0 * self.cell_size

        open_grid = [[maze_map[i][j] != 1 for j in range(self.cols)] for i in range(self.rows)]

        def is_open(i, j):
            return 0 <= i < self.rows and 0 <= j < self.cols and open_grid[i][j]

        cell_to_idx = torch.full((self.rows, self.cols), -1, dtype=torch.long)
        reachable, start_ij = [], None
        bottleneck, far = [], []
        for i in range(self.rows):
            for j in range(self.cols):
                if not open_grid[i][j]:
                    continue
                cell_to_idx[i, j] = len(reachable)
                reachable.append((i, j))
                if maze_map[i][j] == "r":
                    start_ij = (i, j)
                horiz = is_open(i, j - 1) and is_open(i, j + 1) and not is_open(i - 1, j) and not is_open(i + 1, j)
                vert = is_open(i - 1, j) and is_open(i + 1, j) and not is_open(i, j - 1) and not is_open(i, j + 1)
                bottleneck.append(1.0 if (horiz or vert) else 0.0)

        assert start_ij is not None, "maze_map must contain a 'r' cell"
        dist = self._bfs(open_grid, start_ij)
        max_dist = max(dist[c] for c in reachable) or 1
        far = [dist[c] / max_dist for c in reachable]

        self.n_cells = len(reachable)
        self.cell_to_idx = cell_to_idx.to(device)
        self.bottleneck_reward = torch.tensor(bottleneck, dtype=torch.float32, device=device)
        self.far_from_start_reward = torch.tensor(far, dtype=torch.float32, device=device)

    @staticmethod
    def _bfs(open_grid, start):
        rows, cols = len(open_grid), len(open_grid[0])
        dist = {start: 0}
        frontier = [start]
        while frontier:
            nxt = []
            for (i, j) in frontier:
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (i + di, j + dj)
                    if 0 <= n[0] < rows and 0 <= n[1] < cols and open_grid[n[0]][n[1]] and n not in dist:
                        dist[n] = dist[(i, j)] + 1
                        nxt.append(n)
            frontier = nxt
        return dist

    def cell_index(self, xy: torch.Tensor) -> torch.Tensor:
        """``xy``: ``(N, 2)`` raw local ``(x, y)`` -> ``(N,)`` reachable-cell index."""
        j = torch.floor((xy[:, 0] + self.x_map_center) / self.cell_size).long().clamp(0, self.cols - 1)
        i = torch.floor((self.y_map_center - xy[:, 1]) / self.cell_size).long().clamp(0, self.rows - 1)
        return self.cell_to_idx[i, j].clamp(min=0)  # off-map/wall samples fall back to cell 0


def _return_to_go(b_t: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """Discounted return-to-go over a ``(T, N)`` batch, reset at episode ends
    (``dones[t]`` marks the terminal step of an episode, matching the convention
    of ``estimate_advantages``/``compute_gae`` in ``common.py``)."""
    T = b_t.shape[0]
    G = torch.zeros_like(b_t)
    running = torch.zeros(b_t.shape[1], device=b_t.device)
    for t in reversed(range(T)):
        running = b_t[t] + gamma * running * (~dones[t]).float()
        G[t] = running
    return G


class AGA(PPO):
    """PPO whose realized update is asymmetrically, PD-preconditioned toward an
    exploration direction, instead of task reward being augmented."""

    def __init__(self, *, models, memory=None, observation_space=None, state_space=None,
                 action_space=None, device=None, cfg=None):
        super().__init__(
            models=models, memory=memory, observation_space=observation_space,
            state_space=state_space, action_space=action_space, device=device, cfg=cfg,
        )
        c: AGA_CFG = self.cfg
        assert 0.0 <= c.precond_damp < 1.0, "precond_damp must be in [0, 1)"
        self.maze = _MazeCells(c.maze_map, c.cell_size, self.device)
        self._visit_counts = torch.ones(self.maze.n_cells, device=self.device)  # Laplace-smoothed
        self._update_count = 0
        # Intrinsic rewards of the current rollout (one (num_envs, 1) entry per
        # step), collected from the env wrapper's infos and consumed by update().
        self._rollout_int_rewards: list[torch.Tensor] = []

    def record_transition(self, *, observations, states, actions, rewards, next_observations,
                          next_states, terminated, truncated, infos, timestep, timesteps) -> None:
        if (self.training and self.cfg.target_dist == "intrinsic"
                and isinstance(infos, dict) and "int_reward" in infos):
            self._rollout_int_rewards.append(infos["int_reward"].detach())
        super().record_transition(
            observations=observations, states=states, actions=actions, rewards=rewards,
            next_observations=next_observations, next_states=next_states,
            terminated=terminated, truncated=truncated, infos=infos,
            timestep=timestep, timesteps=timesteps,
        )

    def _int_rewards(self, T: int, N: int) -> torch.Tensor:
        """The rollout's ``(T, N)`` intrinsic rewards; cleared as it is consumed."""
        if len(self._rollout_int_rewards) < T:
            raise RuntimeError(
                f"target_dist='intrinsic' needs an intrinsic reward per step, but only "
                f"{len(self._rollout_int_rewards)}/{T} were recorded. Wrap the env with "
                "IntrinsicRewardWrapper (train.py --int_reward drnd|allo)."
            )
        b_t = torch.stack(self._rollout_int_rewards[-T:], dim=0).reshape(T, N)
        self._rollout_int_rewards.clear()
        return b_t

    def _cell_target_reward(self) -> torch.Tensor:
        c: AGA_CFG = self.cfg
        if c.target_dist == "bottleneck":
            return self.maze.bottleneck_reward
        if c.target_dist == "uniform":
            return 1.0 / torch.sqrt(self._visit_counts)
        if c.target_dist == "far_from_start":
            return self.maze.far_from_start_reward
        raise ValueError(f"unknown target_dist {c.target_dist!r}")

    def _target_direction(self, observations, actions, dones) -> torch.Tensor:
        """Single-pass REINFORCE gradient (unit vector) at the current (pre-PPO)
        parameters, on the exact rollout batch the PPO update is about to consume."""
        c: AGA_CFG = self.cfg
        T, N = observations.shape[0], observations.shape[1]
        obs_flat = observations.reshape(T * N, -1)

        if c.target_dist == "intrinsic":
            b_t = self._int_rewards(T, N)
        else:
            idx = self.maze.cell_index(obs_flat[:, :2]).reshape(T, N)
            b_t = self._cell_target_reward()[idx]
        G_t = _return_to_go(b_t, dones, c.meta_gamma)
        G_t = (G_t - G_t.mean()) / (G_t.std() + 1e-8)

        obs_p = self._observation_preprocessor(obs_flat)
        actions_flat = actions.reshape(T * N, -1)
        _, out = self.policy.act(
            {"observations": obs_p, "states": None, "taken_actions": actions_flat}, role="policy")
        # Shared policy/value model (`models.separate: false`): a `role="policy"`
        # call caches this batch's shared-body output for the very next
        # `role="value"` call to reuse (`single_forward_pass`). With no matching
        # value call here, that stale (T*N,)-sized cache would otherwise get
        # consumed by PPO's `last_values` computation in `super().update()`,
        # corrupting its batch size (N,) and breaking `compute_gae`.
        if self.policy is self.value:
            self.policy._shared_output = None
        log_probs = out["log_prob"].reshape(-1)
        loss = -(G_t.reshape(-1) * log_probs).mean()

        params = list(self.policy.parameters())
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        v = torch.cat([
            (-g if g is not None else torch.zeros_like(p)).reshape(-1)
            for p, g in zip(params, grads)
        ])
        return v / (v.norm() + 1e-12)

    def update(self, *, timestep: int, timesteps: int) -> None:
        c: AGA_CFG = self.cfg
        try:
            observations = self.memory.get_tensor_by_name("observations")
            actions = self.memory.get_tensor_by_name("actions")
            terminated = self.memory.get_tensor_by_name("terminated")
            truncated = self.memory.get_tensor_by_name("truncated")
        except Exception:  # noqa: BLE001 — fall back to plain PPO
            super().update(timestep=timestep, timesteps=timesteps)
            return

        T, N = observations.shape[0], observations.shape[1]
        dones = (terminated.bool() | truncated.bool()).squeeze(-1)

        # Persistent per-cell visit counts (shared by the `uniform` target and the
        # optional novelty bonus), incremented once per batch.
        cell_idx_flat = self.maze.cell_index(observations.reshape(T * N, -1)[:, :2])
        self._visit_counts.scatter_add_(0, cell_idx_flat, torch.ones_like(cell_idx_flat, dtype=torch.float32))

        if c.novelty_coef != 0.0:
            bonus = 1.0 / torch.sqrt(self._visit_counts[cell_idx_flat])
            bonus = (bonus - bonus.mean()).reshape(T, N, 1)
            self.memory.get_tensor_by_name("rewards").add_(c.novelty_coef * bonus)

        if c.dense_coef != 0.0:
            # Potential-based shaping (Ng, Harada & Russell 1999): F(s,s') =
            # gamma*Phi(s') - Phi(s) with Phi(s) = -||achieved_goal - goal|| (obs
            # layout [x, y, vx, vy, achieved_x, achieved_y, goal_x, goal_y], see
            # pointmaze_env._get_observations). This exact difference form is what
            # the theorem needs to guarantee the shaped and true-reward optimal
            # policies coincide — summed over an episode it telescopes to a
            # bounded, path-independent term, so it can't reward looping/lingering
            # the way adding the raw potential every step would. Phi(s') is
            # dropped (treated as 0) on any step that ends an episode, since the
            # auto-reset "next" observation there belongs to a different episode.
            def _potential(obs: torch.Tensor) -> torch.Tensor:
                return -torch.linalg.norm(obs[..., 4:6] - obs[..., 6:8], dim=-1, keepdim=True)

            phi = _potential(observations)
            next_observations = torch.cat(
                [observations[1:], self._current_next_observations.unsqueeze(0)], dim=0)
            next_phi = _potential(next_observations)
            not_done = (~dones).float().unsqueeze(-1)
            shaping = c.discount_factor * next_phi * not_done - phi
            self.memory.get_tensor_by_name("rewards").add_(c.dense_coef * shaping)

        theta_before = flat_params(self.policy)
        v_hat = self._target_direction(observations, actions, dones)

        # 1. Real multi-epoch PPO inner loop (skrl's own implementation — actor
        #    AND critic, identical to plain PPOAgent) + 2. diff + rollback.
        super().update(timestep=timestep, timesteps=timesteps)
        delta_ppo = flat_params(self.policy) - theta_before
        set_flat_params(self.policy, theta_before)

        # 3. Asymmetric rank-1 PD preconditioner.
        proj = torch.dot(v_hat, delta_ppo)
        scale = c.precond_beta if proj.item() > 0 else -c.precond_damp
        g_tilde = delta_ppo + scale * proj * v_hat

        # 4. Annealed step onto the preconditioned update.
        self._update_count += 1
        progress = min(1.0, self._update_count / max(1, int(c.alpha_anneal_updates)))
        alpha_t = c.alpha_start + (c.alpha_end - c.alpha_start) * progress
        set_flat_params(self.policy, theta_before + alpha_t * g_tilde)

        self.track_data("AGA / proj", float(proj.item()))
        self.track_data("AGA / alpha", float(alpha_t))
        self.track_data("AGA / g_tilde_norm", float(g_tilde.norm().item()))
        self.track_data("AGA / delta_ppo_norm", float(delta_ppo.norm().item()))

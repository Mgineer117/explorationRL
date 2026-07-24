"""IRPO — Intrinsic Reward Policy Optimization, as a skrl agent.

Port of ``policy/irpo.py`` (the legacy ``IRPO_Learner``) onto skrl's PPO.

A Laplacian representation of the state space (``ALLO``) is learned online; its
eigenvector directions define ``num_options`` signed intrinsic rewards. IRPO
keeps one **persistent exploratory policy per option**, each trained on its own
intrinsic reward, so the agent explores several directions at once. The base
policy is trained by PPO on the environment reward over *all* of the resulting
experience.

Env partitioning. skrl's trainer collects a single rollout, so the parallel envs
are split into ``num_options + 1`` contiguous groups: group 0 is driven by the
base policy, group ``n+1`` by option policy ``n``. Every policy therefore gets
genuine on-policy data for its own objective within one rollout. This replaces
the legacy sequential "collect a fresh batch per option" loop, which cannot be
expressed inside skrl's update without desynchronising the trainer's cached
observations — and under Isaac's massive parallelism it is the natural form.

Off-policy correction is free: ``act`` records the log-prob of the *acting*
policy, so when PPO updates the base policy over the whole batch its ratio
``pi_base(a|s) / pi_behaviour(a|s)`` is exactly the importance weight for the
slices an option policy drove.

Why gradient aggregation was abandoned. An earlier version of this port scored
each option on the base policy's batch and summed the resulting gradients (the
legacy ``j == 0`` round, with mean/PCGrad aggregation). That is invalid here:
options are signed ``+/-`` pairs over the same eigenvector, so their intrinsic
rewards — and hence their gradients — are exact negatives and annihilate on
summation. Measured directly: individual option gradients had norm ~1.28 while
the aggregate collapsed to 5e-08. PCGrad does not help, since for antiparallel
``g_j = -g_i`` the projection ``g_i - (<g_i,g_j>/||g_j||^2) g_j`` is identically
zero. Keeping the option policies *separate* is what makes the ``+/-`` pairing
meaningful: the two halves of a pair explore opposite ends of an eigenvector
instead of cancelling.
"""

from __future__ import annotations

import copy
import dataclasses

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG

from explorationRL.agents.skrl.intrinsic import ALLOIntrinsicRewards, RandomIntrinsicRewards
from explorationRL.extractors import ALLO, ALLO_CFG


@dataclasses.dataclass(kw_only=True)
class IRPO_CFG(PPO_CFG):
    """PPO config + the IRPO knobs (names mirror ``policy/irpo.py``)."""

    num_options: int = 4
    """Number of signed eigenvector directions explored (``+/-`` pairs). The envs
    are split into ``num_options + 1`` groups, so this also sets how many envs
    the base policy keeps."""

    option_learning_rate: float = 3e-4
    """Adam learning rate for each exploratory option policy."""

    option_epochs: int = 4
    """Gradient epochs per option policy per update."""

    option_ratio_clip: float = 0.2
    """PPO-style clip used for the option policies' surrogate."""

    intrinsic_discount: float = 0.99
    """Discount for the per-option intrinsic return."""

    intrinsic_reward_type: str = "allo"
    """``allo`` (learned Laplacian) or ``random`` (frozen-encoder ablation)."""

    extractor_feature_dim: int = 8
    extractor_hidden_dim: list = dataclasses.field(default_factory=lambda: [256, 256])
    extractor_learning_rate: float = 1e-3
    extractor_batch_size: int = 512
    extractor_discount: float = 0.9
    extractor_updates_per_rollout: int = 1
    """ALLO gradient steps per policy update (0 freezes the representation)."""

    extractor_positional_indices: list | None = dataclasses.field(
        default_factory=lambda: [0, 1]
    )
    """Observation slice the representation is learned over (xy for PointMaze)."""


class IRPO(PPO):
    """PPO base policy + persistent per-option exploratory policies."""

    def __init__(self, *, models, memory=None, observation_space=None, state_space=None,
                 action_space=None, device=None, cfg=None):
        super().__init__(
            models=models, memory=memory, observation_space=observation_space,
            state_space=state_space, action_space=action_space, device=device, cfg=cfg,
        )
        c: IRPO_CFG = self.cfg
        self.num_options = int(c.num_options)

        # One persistent exploratory policy per option, initialised from the base.
        self.option_policies = nn.ModuleList(
            [copy.deepcopy(self.policy) for _ in range(self.num_options)]
        ).to(self.device)
        self.option_optimizers = [
            torch.optim.Adam(p.parameters(), lr=c.option_learning_rate)
            for p in self.option_policies
        ]

        obs_dim = int(observation_space.shape[0])
        needed = self.num_options // 2 + 2  # highest eigenvector index used + 1
        self.extractor = ALLO(
            observation_dim=obs_dim,
            cfg=ALLO_CFG(
                feature_dim=max(int(c.extractor_feature_dim), needed),
                hidden_dim=list(c.extractor_hidden_dim),
                positional_indices=c.extractor_positional_indices,
                learning_rate=c.extractor_learning_rate,
                batch_size=c.extractor_batch_size,
                discount=c.extractor_discount,
            ),
            device=self.device,
        )
        reward_cls = RandomIntrinsicRewards if c.intrinsic_reward_type == "random" else ALLOIntrinsicRewards
        self.intrinsic_rewards = reward_cls(self.extractor, self.num_options, device=self.device)

        self.checkpoint_modules["extractor"] = self.extractor
        self.checkpoint_modules["option_policies"] = self.option_policies
        self._slices: list[tuple[int, int]] | None = None

    # ── env partitioning ──────────────────────────────────────────────────── #
    def _group_slices(self, num_envs: int) -> list[tuple[int, int]]:
        """Contiguous env ranges: index 0 = base policy, 1..N = option policies."""
        if self._slices is not None:
            return self._slices
        groups = self.num_options + 1
        if num_envs < groups:
            raise ValueError(
                f"IRPO needs at least num_options + 1 = {groups} parallel envs "
                f"(one group per policy), got num_envs={num_envs}. Reduce "
                f"num_options or raise --num_envs."
            )
        size = num_envs // groups
        bounds = [(i * size, (i + 1) * size) for i in range(groups)]
        bounds[-1] = (bounds[-1][0], num_envs)  # last group absorbs the remainder
        self._slices = bounds
        return bounds

    def _policy_for(self, group: int):
        return self.policy if group == 0 else self.option_policies[group - 1]

    # ── acting ────────────────────────────────────────────────────────────── #
    def act(self, observations: torch.Tensor, states: torch.Tensor | None, *,
            timestep: int, timesteps: int):
        if not self.training or timestep < self.cfg.random_timesteps:
            return super().act(observations, states, timestep=timestep, timesteps=timesteps)

        obs_p = self._observation_preprocessor(observations)
        states_p = self._state_preprocessor(states)
        slices = self._group_slices(observations.shape[0])

        # Values are computed for ALL envs BEFORE the per-slice policy calls. A
        # shared policy/value network (models.separate: false) caches its forward
        # pass between role lookups, so evaluating the value head after a
        # per-slice policy call would return that slice's stale rows.
        values, _ = self.value.act({"observations": obs_p, "states": states_p}, role="value")
        self._current_values = self._value_preprocessor(values, inverse=True)

        actions = torch.zeros(observations.shape[0], self.action_space.shape[0],
                              device=observations.device)
        log_probs = torch.zeros(observations.shape[0], 1, device=observations.device)
        for gi, (lo, hi) in enumerate(slices):
            a, out = self._policy_for(gi).act({"observations": obs_p[lo:hi], "states": None},
                                              role="policy")
            actions[lo:hi] = a
            log_probs[lo:hi] = out["log_prob"]

        # Log-prob of the ACTING policy -> PPO's ratio becomes the importance
        # weight for the option-driven slices (see the module docstring).
        self._current_log_prob = log_probs
        return actions, {"log_prob": log_probs}

    # ── option updates ────────────────────────────────────────────────────── #
    def _intrinsic_returns(self, reward: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Normalised discounted return of one option's intrinsic reward."""
        T = reward.shape[0]
        returns = torch.zeros_like(reward)
        running = torch.zeros(reward.shape[1], device=reward.device)
        for t in reversed(range(T)):
            running = reward[t] + self.cfg.intrinsic_discount * running * (~dones[t]).float()
            returns[t] = running
        return (returns - returns.mean()) / (returns.std() + 1e-8)

    def _update_option(self, n: int, observations, actions, old_log_prob, advantages) -> float:
        """Clipped-surrogate update of option policy ``n`` on its own slice."""
        policy, optimizer = self.option_policies[n], self.option_optimizers[n]
        obs = self._observation_preprocessor(observations.reshape(-1, observations.shape[-1]))
        act = actions.reshape(-1, actions.shape[-1])
        old_lp = old_log_prob.reshape(-1)
        adv = advantages.reshape(-1)

        last = 0.0
        for _ in range(int(self.cfg.option_epochs)):
            _, out = policy.act({"observations": obs, "states": None, "taken_actions": act},
                                role="policy")
            ratio = torch.exp(out["log_prob"].reshape(-1) - old_lp)
            clip = self.cfg.option_ratio_clip
            loss = -torch.min(ratio * adv,
                              torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            last = float(loss.item())
        return last

    # ── skrl hook ─────────────────────────────────────────────────────────── #
    def update(self, *, timestep: int, timesteps: int) -> None:
        try:
            observations = self.memory.get_tensor_by_name("observations")
            actions = self.memory.get_tensor_by_name("actions")
            log_prob = self.memory.get_tensor_by_name("log_prob")
            terminated = self.memory.get_tensor_by_name("terminated")
            truncated = self.memory.get_tensor_by_name("truncated")
        except Exception:  # noqa: BLE001 — fall back to plain PPO
            super().update(timestep=timestep, timesteps=timesteps)
            return

        dones = (terminated | truncated).squeeze(-1)
        slices = self._group_slices(observations.shape[1])

        # 1. Refresh the Laplacian representation on the whole rollout.
        for _ in range(int(self.cfg.extractor_updates_per_rollout)):
            for k, v in self.extractor.learn(observations, dones).items():
                self.track_data(k, v)

        # 2. Intrinsic rewards. next_observations are the rollout shifted by one.
        next_observations = torch.cat([observations[1:], observations[-1:]], dim=0)
        T, N = observations.shape[0], observations.shape[1]
        r_int = self.intrinsic_rewards(
            observations.reshape(-1, observations.shape[-1]),
            next_observations.reshape(-1, next_observations.shape[-1]),
        ).reshape(T, N, -1)

        # 3. Train each option policy on ITS OWN slice and intrinsic reward.
        #    Kept separate on purpose — summing them cancels (module docstring).
        losses = []
        for n in range(self.num_options):
            lo, hi = slices[n + 1]
            adv = self._intrinsic_returns(r_int[:, lo:hi, n], dones[:, lo:hi])
            losses.append(self._update_option(
                n, observations[:, lo:hi], actions[:, lo:hi], log_prob[:, lo:hi], adv
            ))

        self.track_data("IRPO / Intrinsic reward (abs mean)", float(r_int.abs().mean().item()))
        self.track_data("IRPO / Option loss (mean)", float(sum(losses) / max(1, len(losses))))
        # Divergence between option policies = how differently they explore.
        with torch.no_grad():
            probe = self._observation_preprocessor(observations[0])
            means = []
            for n in range(self.num_options):
                _, out = self.option_policies[n].act({"observations": probe, "states": None},
                                                     role="policy")
                means.append(out["mean_actions"])
            spread = torch.stack(means).std(dim=0).mean() if len(means) > 1 else torch.tensor(0.0)
        self.track_data("IRPO / Option policy spread", float(spread.item()))

        # 4. Base policy: ordinary PPO on the environment reward over ALL slices;
        #    the stored behaviour log-probs make the ratio an importance weight.
        super().update(timestep=timestep, timesteps=timesteps)

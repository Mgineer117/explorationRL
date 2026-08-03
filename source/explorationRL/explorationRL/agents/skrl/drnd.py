"""DRND — Distributional Random Network Distillation, as a skrl agent.

Port of ``policy/drnd.py`` (the legacy pipeline's ``DRND_Learner``) onto skrl's
PPO. DRND is an exploration bonus: a *predictor* network is distilled toward an
ensemble of ``num_targets`` frozen random *target* networks, and the resulting
prediction error is added to the environment reward as intrinsic novelty.

Novelty (matching the original ``intrinsic_reward``), for predictor output
``p`` and target ensemble statistics ``mu = E[t]``, ``B2 = E[t^2]``:

    b1 = alpha * sum_i (p_i - mu_i)^2
    b2 = (1 - alpha) * sum_i sqrt( clip( |p_i^2 - mu_i^2| / (B2_i - mu_i^2), 1e-6, 1 ) )
    r_int = (b1 + b2) / running_std(b1 + b2)

and the predictor loss (``drnd_loss``) regresses ``p`` onto a *randomly chosen*
member of the target ensemble, over a random ``update_proportion`` subset of the
batch.

Difference from the original, stated plainly: the legacy learner carried a
*second* value head for the intrinsic stream with its own returns/advantages.
This port instead folds the bonus into the reward
(``extrinsic_reward_scale * r_ext + intrinsic_reward_scale * r_int``) and keeps
skrl's single PPO critic. That is the standard RND-as-reward-bonus formulation
and needs no fork of skrl's ``_update``; a two-critic variant would require
overriding PPO's whole update path.
"""

from __future__ import annotations

import dataclasses

import gymnasium
import torch

from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG

from explorationRL.agents.skrl.intrinsic import DRNDNovelty


@dataclasses.dataclass(kw_only=True)
class DRND_CFG(PPO_CFG):
    """PPO config + the DRND-specific knobs (names mirror ``policy/drnd.py``)."""

    num_targets: int = 5
    """Size of the frozen random target ensemble (the original's ``K``/``num``)."""

    embedding_dim: int = 64
    """Output width of the predictor and each target network."""

    predictor_hidden: list = dataclasses.field(default_factory=lambda: [512, 512, 512])
    """Hidden widths of the predictor MLP (ReLU)."""

    target_hidden: list = dataclasses.field(default_factory=lambda: [128, 128])
    """Hidden widths of each frozen target MLP (Tanh)."""

    alpha: float = 0.9
    """Mix between the two novelty terms b1 and b2. Range ``[0, 1]``."""

    intrinsic_reward_scale: float = 1.0
    """Weight on the normalized intrinsic reward."""

    extrinsic_reward_scale: float = 2.0
    """Weight on the environment reward."""

    drnd_loss_scale: float = 1.0
    """Scaling of the predictor distillation loss."""

    update_proportion: float = 0.25
    """Fraction of each batch used to update the predictor. Range ``(0, 1]``."""

    predictor_learning_rate: float = 3e-4
    """Adam learning rate for the predictor network."""


class DRND(PPO):
    """PPO driven by an environment reward augmented with a DRND novelty bonus."""

    def __init__(self, *, models, memory=None, observation_space=None, state_space=None,
                 action_space=None, device=None, cfg=None):
        super().__init__(
            models=models, memory=memory, observation_space=observation_space,
            state_space=state_space, action_space=action_space, device=device, cfg=cfg,
        )
        c: DRND_CFG = self.cfg
        obs_dim = int(gymnasium.spaces.flatdim(observation_space))

        self._novelty = DRNDNovelty(
            obs_dim,
            num_targets=int(c.num_targets),
            embedding_dim=int(c.embedding_dim),
            predictor_hidden=list(c.predictor_hidden),
            target_hidden=list(c.target_hidden),
            alpha=c.alpha,
            update_proportion=c.update_proportion,
            learning_rate=c.predictor_learning_rate,
            device=self.device,
        )
        # Persist the predictor with the agent checkpoint; the targets are frozen
        # random features and are reproduced from the seed.
        self.checkpoint_modules["predictor"] = self._novelty.predictor

        # next_observations seen during the current rollout, consumed by update().
        self._rollout_next_obs: list[torch.Tensor] = []

    # ── novelty ───────────────────────────────────────────────────────────── #
    def _preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        """Feed the DRND nets the same normalized observations the policy sees."""
        try:
            return self._observation_preprocessor(obs)
        except Exception:  # noqa: BLE001 — preprocessor is optional
            return obs

    @torch.no_grad()
    def _intrinsic_reward(self, next_observations: torch.Tensor) -> torch.Tensor:
        return self._novelty(self._preprocess(next_observations))

    # ── skrl hooks ────────────────────────────────────────────────────────── #
    def record_transition(self, *, observations, states, actions, rewards, next_observations,
                          next_states, terminated, truncated, infos, timestep, timesteps) -> None:
        if self.training:
            r_int = self._intrinsic_reward(next_observations)
            self._rollout_next_obs.append(next_observations.detach())
            rewards = (
                self.cfg.extrinsic_reward_scale * rewards
                + self.cfg.intrinsic_reward_scale * r_int
            )
            self.track_data("DRND / Intrinsic reward (mean)", r_int.mean().item())
        super().record_transition(
            observations=observations, states=states, actions=actions, rewards=rewards,
            next_observations=next_observations, next_states=next_states,
            terminated=terminated, truncated=truncated, infos=infos,
            timestep=timestep, timesteps=timesteps,
        )

    def update(self, *, timestep: int, timesteps: int) -> None:
        # Distil the predictor on this rollout's observations before PPO's update,
        # so the bonus for the next rollout reflects what has now been visited.
        if self._rollout_next_obs:
            obs = self._preprocess(torch.cat(self._rollout_next_obs, dim=0))
            self._rollout_next_obs.clear()
            loss = self.cfg.drnd_loss_scale * self._novelty.loss(obs)
            self._novelty.optimizer.zero_grad()
            loss.backward()
            self._novelty.optimizer.step()
            self.track_data("DRND / Predictor loss", loss.item())
        super().update(timestep=timestep, timesteps=timesteps)

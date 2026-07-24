"""PSNE — Parameter Space Noise for Exploration, as a skrl agent.

Port of ``policy/psne.py`` (the legacy ``PSNE_Learner``) onto skrl's TRPO.
PSNE is TRPO plus the adaptive parameter-space noise of Plappert et al. (2017),
`arXiv:1706.01905 <https://arxiv.org/abs/1706.01905>`_:

* a *perturbed* copy of the policy is kept, ``theta~ = theta + sigma * N(0, I)``;
* rollouts are collected by acting with that perturbed copy, which explores by
  perturbing the policy's parameters rather than its output distribution;
* ``sigma`` adapts to hold the induced policy divergence near a target:
  ``sigma *= alpha`` when ``KL(pi_clean || pi_perturbed) < delta``, else
  ``sigma /= alpha``;
* the *clean* policy is then updated by the usual TRPO step.

Faithfulness note — which log-prob is recorded. The legacy learner collected
data with the perturbed actor but built its surrogate from the **clean** actor's
log-probs (``-(logprobs * advantages).mean()``), i.e. an uncorrected policy
gradient. skrl's TRPO forms an importance ratio ``exp(log_prob - old_log_prob)``
instead, so this port records the log-prob of the taken action **under the clean
policy**. At the start of the update the ratio is then 1 and the surrogate
gradient is exactly ``grad log pi_clean * A`` — reproducing the legacy gradient
while keeping skrl's trust-region machinery intact.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import torch

from skrl.agents.torch.trpo import TRPO
from skrl.agents.torch.trpo.trpo_cfg import TRPO_CFG

from explorationRL.agents.skrl.common import flat_params, gaussian_kl, set_flat_params


@dataclasses.dataclass(kw_only=True)
class PSNE_CFG(TRPO_CFG):
    """TRPO config + the parameter-space-noise knobs (names mirror ``policy/psne.py``)."""

    initial_noise_std: float = 0.1
    """Initial parameter-noise standard deviation ``sigma``."""

    noise_adaptation_coefficient: float = 1.01
    """Multiplicative factor ``alpha`` used to grow/shrink ``sigma``."""

    noise_kl_threshold: float = 0.03
    """Target divergence ``delta``: ``sigma`` grows below it and shrinks above it."""

    noise_kl_sample_size: int = 512
    """Observations retained from the rollout to estimate the perturbation KL."""


class PSNE(TRPO):
    """TRPO whose rollouts are collected by a parameter-perturbed policy copy."""

    def __init__(self, *, models, memory=None, observation_space=None, state_space=None,
                 action_space=None, device=None, cfg=None):
        super().__init__(
            models=models, memory=memory, observation_space=observation_space,
            state_space=state_space, action_space=action_space, device=device, cfg=cfg,
        )
        # The perturbed copy only ever acts — it is never optimized.
        self._perturbed_policy = copy.deepcopy(self.policy)
        for p in self._perturbed_policy.parameters():
            p.requires_grad_(False)

        self._sigma = float(self.cfg.initial_noise_std)
        # Observations kept from the last rollout, used to measure the KL that
        # sigma is adapted against.
        self._kl_observations: torch.Tensor | None = None
        self._perturb_policy()

    # ── parameter-space noise ─────────────────────────────────────────────── #
    def _perturb_policy(self) -> None:
        """Resample ``theta~ = theta + sigma * eps`` and adapt ``sigma``."""
        with torch.no_grad():
            theta = flat_params(self.policy)
            set_flat_params(self._perturbed_policy, theta + self._sigma * torch.randn_like(theta))

            if self._kl_observations is not None:
                kl = self._perturbation_kl(self._kl_observations)
                if kl < self.cfg.noise_kl_threshold:
                    self._sigma *= self.cfg.noise_adaptation_coefficient
                else:
                    self._sigma /= self.cfg.noise_adaptation_coefficient
                self.track_data("PSNE / Perturbation KL", kl.item())
            self.track_data("PSNE / Noise sigma", self._sigma)

    def _perturbation_kl(self, observations: torch.Tensor) -> torch.Tensor:
        _, clean = self.policy.act({"observations": observations}, role="policy")
        _, noisy = self._perturbed_policy.act({"observations": observations}, role="policy")
        return gaussian_kl(
            clean["mean_actions"], clean["log_std"],
            noisy["mean_actions"], noisy["log_std"],
        )

    # ── skrl hooks ────────────────────────────────────────────────────────── #
    def act(self, observations: torch.Tensor, states: torch.Tensor | None, *,
            timestep: int, timesteps: int) -> tuple[torch.Tensor, dict[str, Any]]:
        # Evaluation (and the initial random phase) uses the clean policy.
        if not self.training or timestep < self.cfg.random_timesteps:
            return super().act(observations, states, timestep=timestep, timesteps=timesteps)

        inputs = {
            "observations": self._observation_preprocessor(observations),
            "states": self._state_preprocessor(states),
        }
        # Explore by acting with the PERTURBED policy ...
        actions, outputs = self._perturbed_policy.act(inputs, role="policy")
        # ... but record the log-prob of that action under the CLEAN policy, so
        # TRPO's ratio starts at 1 (see the module docstring).
        _, clean_outputs = self.policy.act({**inputs, "taken_actions": actions}, role="policy")
        self._current_log_prob = clean_outputs["log_prob"]
        outputs["log_prob"] = clean_outputs["log_prob"]

        values, _ = self.value.act(inputs, role="value")
        self._current_values = self._value_preprocessor(values, inverse=True)
        return actions, outputs

    def record_transition(self, *, observations, states, actions, rewards, next_observations,
                          next_states, terminated, truncated, infos, timestep, timesteps) -> None:
        if self.training:
            # Keep a bounded slice of observations for the sigma-adaptation KL.
            n = int(self.cfg.noise_kl_sample_size)
            obs = self._observation_preprocessor(observations).detach()
            self._kl_observations = obs[:n] if obs.shape[0] > n else obs
        super().record_transition(
            observations=observations, states=states, actions=actions, rewards=rewards,
            next_observations=next_observations, next_states=next_states,
            terminated=terminated, truncated=truncated, infos=infos,
            timestep=timestep, timesteps=timesteps,
        )

    def update(self, *, timestep: int, timesteps: int) -> None:
        super().update(timestep=timestep, timesteps=timesteps)
        # The clean policy just moved — draw a fresh perturbation around it.
        self._perturb_policy()

"""MAML — model-agnostic meta-learning policy optimization, as a skrl agent.

Port of ``policy/maml.py`` (the legacy ``MAML_Learner``) onto skrl's PPO.

The legacy learner adapts a clone of the actor over ``num_exp_updates`` inner
rounds, each collecting a fresh rollout (``adapt_actor``), then meta-updates the
base actor. This port expresses the same inner/outer structure inside a single
skrl rollout by splitting the parallel envs into a **support** slice and a
**query** slice:

    inner (support)   theta' = theta - alpha * grad_theta L(theta; D_support)
    outer (query)     theta  <- theta - beta  * grad     L(theta'; D_query)

so the base policy is optimised for *post-adaptation* performance — the MAML
objective — rather than for immediate return. The support/query split is what
makes the meta-gradient meaningful: adapting and evaluating on the same
transitions would just be two ordinary gradient steps.

First-order by default. ``first_order: true`` uses FOMAML — the outer gradient
is taken at ``theta'`` and applied to ``theta``, dropping the second-derivative
term. This is the standard practical choice and avoids differentiating through
the inner step (which, with skrl models, needs a functional re-parameterisation
of every module). ``first_order: false`` is rejected rather than silently
approximated, so a config can never quietly get something it did not ask for.

Exploration. Unlike IRPO/HRL this agent carries no option policies: what MAML
contributes is the meta-objective, and mixing in the option machinery would
confound which part is responsible for a result. Use ``--algorithm irpo`` for
the option-based exploration.
"""

from __future__ import annotations

import copy
import dataclasses

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG

from explorationRL.agents.skrl.common import compute_gae


@dataclasses.dataclass(kw_only=True)
class MAML_CFG(PPO_CFG):
    """PPO config + the meta-learning knobs (names mirror ``policy/maml.py``)."""

    inner_learning_rate: float = 0.1
    """Inner-loop step size ``alpha`` (the adaptation step)."""

    inner_steps: int = 1
    """Gradient steps taken on the support slice per update."""

    meta_learning_rate: float = 3e-4
    """Outer-loop step size ``beta`` applied to the base policy."""

    support_fraction: float = 0.5
    """Fraction of envs forming the support (adaptation) slice; the rest are query."""

    inner_ratio_clip: float = 0.2
    """Clip used by both the inner and outer surrogates."""

    first_order: bool = True
    """FOMAML. ``false`` is rejected — see the module docstring."""


class MAML(PPO):
    """PPO whose base policy is trained for post-adaptation (meta) performance."""

    def __init__(self, *, models, memory=None, observation_space=None, state_space=None,
                 action_space=None, device=None, cfg=None):
        super().__init__(
            models=models, memory=memory, observation_space=observation_space,
            state_space=state_space, action_space=action_space, device=device, cfg=cfg,
        )
        c: MAML_CFG = self.cfg
        if not bool(c.first_order):
            raise ValueError(
                "first_order: false (full second-order MAML) is not implemented — it "
                "requires differentiating through the inner step via a functional "
                "re-parameterisation of the skrl models. Set first_order: true."
            )
        self.meta_optimizer = torch.optim.Adam(self.policy.parameters(), lr=c.meta_learning_rate)
        # The adapted policy is exposed for acting so rollouts reflect the
        # post-adaptation behaviour the meta-objective optimises.
        self.adapted_policy = copy.deepcopy(self.policy).to(self.device)
        self.checkpoint_modules["adapted_policy"] = self.adapted_policy
        self._split: tuple[int, int] | None = None

    def _support_query(self, num_envs: int) -> tuple[int, int]:
        if self._split is None:
            n_support = max(1, min(num_envs - 1, int(num_envs * float(self.cfg.support_fraction))))
            self._split = (n_support, num_envs)
        return self._split

    # ── acting ────────────────────────────────────────────────────────────── #
    def act(self, observations, states, *, timestep: int, timesteps: int):
        if not self.training or timestep < self.cfg.random_timesteps:
            return super().act(observations, states, timestep=timestep, timesteps=timesteps)

        obs_p = self._observation_preprocessor(observations)
        states_p = self._state_preprocessor(states)
        n_support, n = self._support_query(observations.shape[0])

        values, _ = self.value.act({"observations": obs_p, "states": states_p}, role="value")
        self._current_values = self._value_preprocessor(values, inverse=True)

        actions = torch.zeros(n, self.action_space.shape[0], device=observations.device)
        log_probs = torch.zeros(n, 1, device=observations.device)
        # Support envs run the base policy (they produce the adaptation data);
        # query envs run the adapted policy (they measure post-adaptation return).
        for policy, lo, hi in ((self.policy, 0, n_support), (self.adapted_policy, n_support, n)):
            a, out = policy.act({"observations": obs_p[lo:hi], "states": None}, role="policy")
            actions[lo:hi] = a
            log_probs[lo:hi] = out["log_prob"]

        self._current_log_prob = log_probs
        return actions, {"log_prob": log_probs}

    # ── surrogate ─────────────────────────────────────────────────────────── #
    def _surrogate(self, policy, obs, act, old_lp, adv) -> torch.Tensor:
        _, out = policy.act({"observations": obs, "states": None, "taken_actions": act},
                            role="policy")
        ratio = torch.exp(out["log_prob"].reshape(-1) - old_lp)
        clip = self.cfg.inner_ratio_clip
        return -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()

    def _advantages(self, rewards, values, dones) -> torch.Tensor:
        return compute_gae(rewards, values, dones,
                           self.cfg.discount_factor, self.cfg.gae_lambda)

    # ── skrl hook ─────────────────────────────────────────────────────────── #
    def update(self, *, timestep: int, timesteps: int) -> None:
        try:
            observations = self.memory.get_tensor_by_name("observations")
            actions = self.memory.get_tensor_by_name("actions")
            log_prob = self.memory.get_tensor_by_name("log_prob")
            rewards = self.memory.get_tensor_by_name("rewards")
            terminated = self.memory.get_tensor_by_name("terminated")
            truncated = self.memory.get_tensor_by_name("truncated")
        except Exception:  # noqa: BLE001
            super().update(timestep=timestep, timesteps=timesteps)
            return

        dones = (terminated | truncated).squeeze(-1)
        values = self.memory.get_tensor_by_name("values").squeeze(-1)
        adv = self._advantages(rewards.squeeze(-1), values, dones)
        n_support, n = self._support_query(observations.shape[1])

        def flat(x, lo, hi):
            return x[:, lo:hi].reshape(-1, x.shape[-1]) if x.dim() == 3 else x[:, lo:hi].reshape(-1)

        obs_s = self._observation_preprocessor(flat(observations, 0, n_support))
        obs_q = self._observation_preprocessor(flat(observations, n_support, n))

        # ── inner loop: adapt a clone of theta on the SUPPORT slice ────────── #
        adapted = copy.deepcopy(self.policy)
        inner_opt = torch.optim.SGD(adapted.parameters(), lr=self.cfg.inner_learning_rate)
        inner_loss = torch.tensor(0.0)
        for _ in range(int(self.cfg.inner_steps)):
            inner_loss = self._surrogate(adapted, obs_s, flat(actions, 0, n_support),
                                         flat(log_prob, 0, n_support), flat(adv, 0, n_support))
            inner_opt.zero_grad()
            inner_loss.backward()
            nn.utils.clip_grad_norm_(adapted.parameters(), 1.0)
            inner_opt.step()

        # ── outer loop: evaluate theta' on the QUERY slice, apply to theta ─── #
        outer_loss = self._surrogate(adapted, obs_q, flat(actions, n_support, n),
                                     flat(log_prob, n_support, n), flat(adv, n_support, n))
        adapted.zero_grad(set_to_none=True)
        outer_loss.backward()
        # FOMAML: transplant the gradient taken at theta' onto theta.
        self.meta_optimizer.zero_grad()
        for p_base, p_adapted in zip(self.policy.parameters(), adapted.parameters()):
            p_base.grad = (p_adapted.grad.clone() if p_adapted.grad is not None
                           else torch.zeros_like(p_base))
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.meta_optimizer.step()

        # The adapted policy drives the query envs next rollout.
        self.adapted_policy.load_state_dict(adapted.state_dict())

        with torch.no_grad():
            drift = torch.norm(
                torch.cat([(a - b).reshape(-1) for a, b in
                           zip(adapted.parameters(), self.policy.parameters())])
            )
        self.track_data("MAML / Inner loss", float(inner_loss.item()))
        self.track_data("MAML / Outer (meta) loss", float(outer_loss.item()))
        self.track_data("MAML / Adaptation drift", float(drift.item()))

        # Critic still trains by ordinary PPO on the environment reward.
        super().update(timestep=timestep, timesteps=timesteps)

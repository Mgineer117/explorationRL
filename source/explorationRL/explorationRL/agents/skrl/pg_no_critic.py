"""PGNoCritic — multi-epoch clipped-surrogate policy gradient with NO learned
critic: advantages come from a per-rollout normalized discounted Monte-Carlo
return-to-go, not skrl's own GAE-with-value-bootstrap.

Ablation for a specific hypothesis about why every exploration variant tried
on PointMaze-v1 (PPO, PPO+DRND, PPO+ALLOGoal, AGA+ALLOGoal, AGA+best)
collapses its exploration to a small region near the start within the first
~10-30k steps in most seeds, independent of which exploration bonus is used
(see the visitation-heatmap sweep): this repo's configs use
``models.separate: false``, a SHARED actor/critic trunk. Before the critic has
learned anything true (every reward is 0 until the goal is found at least
once), its value-loss gradient is essentially noise -- and because the trunk
is shared, that noise directly shapes the *policy's own features*, not just
an isolated value head. The policy can then "confidently" lock onto whatever
narrow behavior the critic's noise happened to favor first, regardless of
what exploration bonus is layered on top.

This agent removes that noise source entirely: the value network is built
(skrl's Runner expects one) but never trained -- no value loss is ever
computed, so its output carries no information and, with a genuinely separate
network (``models.separate: true``), no gradient from it ever reaches the
policy. In its place, the advantage is a plain per-rollout normalized
discounted return-to-go (the same construction as ``aga.py``'s ``v_hat``
target), which is a *higher-variance* per-sample estimate than a well-fit GAE
baseline, but carries no *spurious directional bias* from a critic that
hasn't learned anything true yet.

Everything else -- multi-epoch reuse of the batch, PPO's clipped-ratio
surrogate, entropy bonus, KL-adaptive LR/early-stopping -- is kept identical
to plain PPO (this is a near-verbatim copy of ``skrl.agents.torch.ppo.PPO``'s
own ``update()``, with the value-network forward/loss and GAE call removed),
so this isolates exactly one variable: critic vs. no critic.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG
from skrl.resources.schedulers.torch import KLAdaptiveLR

from explorationRL.agents.skrl.common import return_to_go


@dataclasses.dataclass(kw_only=True)
class PGNC_CFG(PPO_CFG):
    """PPO config, minus the critic. ``value_loss_scale``/``value_clip`` are
    inherited from ``PPO_CFG`` but unused -- the value network is built (the
    framework expects one) and never trained."""


class PGNoCritic(PPO):
    """Clipped-surrogate policy gradient with a normalized Monte-Carlo
    return-to-go advantage instead of a learned critic/GAE baseline."""

    def update(self, *, timestep: int, timesteps: int) -> None:
        c: PGNC_CFG = self.cfg
        try:
            rewards = self.memory.get_tensor_by_name("rewards")
            terminated = self.memory.get_tensor_by_name("terminated")
            truncated = self.memory.get_tensor_by_name("truncated")
        except Exception:  # noqa: BLE001 — memory tensors not available yet
            return

        dones = (terminated.bool() | truncated.bool()).squeeze(-1)
        G = return_to_go(rewards.squeeze(-1), dones, c.discount_factor)
        advantages = ((G - G.mean()) / (G.std() + 1e-8)).unsqueeze(-1)

        self.memory.set_tensor_by_name("advantages", advantages)
        self.memory.set_tensor_by_name("returns", G.unsqueeze(-1))
        # `values` must exist for memory.sample()'s fixed tensor names; the
        # critic is never trained or read, so this is a harmless placeholder.
        self.memory.set_tensor_by_name("values", torch.zeros_like(advantages))

        cumulative_policy_loss = 0.0
        cumulative_entropy_loss = 0.0
        kl_divergences: list[torch.Tensor] = []

        for epoch in range(c.learning_epochs):
            kl_divergences = []
            for (sampled_observations, sampled_states, sampled_actions, sampled_log_prob,
                 _sampled_values, _sampled_returns, sampled_advantages) in self.memory.sample(
                    names=self._tensors_names, batch_size=len(self.memory), mini_batches=c.mini_batches):

                inputs = {
                    "observations": self._observation_preprocessor(sampled_observations, train=not epoch),
                    "states": self._state_preprocessor(sampled_states, train=not epoch),
                }
                _, outputs = self.policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                next_log_prob = outputs["log_prob"]

                with torch.no_grad():
                    ratio = next_log_prob - sampled_log_prob
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)
                if c.kl_threshold and kl_divergence > c.kl_threshold:
                    break

                entropy_loss = (-c.entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                                if c.entropy_loss_scale else 0.0)

                ratio = torch.exp(next_log_prob - sampled_log_prob)
                surrogate = sampled_advantages * ratio
                surrogate_clipped = sampled_advantages * torch.clip(
                    ratio, 1.0 - c.ratio_clip, 1.0 + c.ratio_clip)
                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                self.optimizer.zero_grad()
                (policy_loss + entropy_loss).backward()
                if c.grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), c.grad_norm_clip)
                self.optimizer.step()

                cumulative_policy_loss += policy_loss.item()
                if c.entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            if self.scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        n = c.learning_epochs * c.mini_batches
        self.track_data("Loss / Policy loss", cumulative_policy_loss / n)
        if c.entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / n)
        self.track_data("Policy / Standard deviation",
                        self.policy.distribution(role="policy").stddev.mean().item())
        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

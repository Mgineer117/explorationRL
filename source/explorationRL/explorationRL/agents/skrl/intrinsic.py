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

import torch
import torch.nn as nn


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
    def forward(self, observations: torch.Tensor, next_observations: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, num_options)`` intrinsic rewards."""
        if self.use_difference:
            delta = self.extractor(next_observations) - self.extractor(observations)
        else:
            delta = self.extractor(observations)
        rewards = delta[:, self._indices] * self._signs
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

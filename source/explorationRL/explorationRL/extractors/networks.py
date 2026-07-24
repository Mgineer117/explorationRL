"""Feature networks for the representation learners.

Port of ``extractor/base/mlp.py``'s ``NeuralNet``, trimmed to what the skrl-side
agents need: a plain MLP mapping observations to a ``feature_dim``-dimensional
embedding. (The legacy CNN / ImageEncoder variants existed for the Atari
observations, which the Isaac Lab pipeline does not carry.)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureMLP(nn.Module):
    """MLP encoder ``obs -> R^feature_dim``."""

    def __init__(self, input_dim: int, hidden_dim: list[int], feature_dim: int,
                 activation: nn.Module | None = None, device: str | torch.device = "cpu"):
        super().__init__()
        activation = activation if activation is not None else nn.ReLU()
        self.input_dim = int(input_dim)
        self.feature_dim = int(feature_dim)

        layers: list[nn.Module] = []
        d = self.input_dim
        for h in hidden_dim:
            layers += [nn.Linear(d, int(h)), activation.__class__()]
            d = int(h)
        layers.append(nn.Linear(d, self.feature_dim))
        self.net = nn.Sequential(*layers)
        self.device = device
        self.to(device)

    def forward(self, state: torch.Tensor, deterministic: bool = True):
        """Return ``(features, info)`` — the tuple shape the legacy code expects."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.net(state), {}

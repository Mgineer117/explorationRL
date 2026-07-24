"""Helpers shared by the ported research agents.

Small, dependency-free utilities that several of the ported algorithms need:
flat parameter get/set (parameter-space perturbation, TRPO-style line search)
and an analytic diagonal-Gaussian KL between two skrl Gaussian policies.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def flat_params(model: nn.Module) -> torch.Tensor:
    """Concatenate every parameter of ``model`` into one 1-D tensor."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    """Write a 1-D tensor produced by :func:`flat_params` back into ``model``."""
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx: idx + n].view_as(p.data))
        idx += n


def gaussian_kl(mean_p: torch.Tensor, log_std_p: torch.Tensor,
                mean_q: torch.Tensor, log_std_q: torch.Tensor) -> torch.Tensor:
    """Mean KL(p || q) for diagonal Gaussians, summed over action dimensions.

        KL = sum_i [ log(sd_q/sd_p) + (sd_p^2 + (mu_p - mu_q)^2) / (2 sd_q^2) - 1/2 ]
    """
    var_p = torch.exp(2.0 * log_std_p)
    var_q = torch.exp(2.0 * log_std_q)
    kl = log_std_q - log_std_p + (var_p + (mean_p - mean_q) ** 2) / (2.0 * var_q) - 0.5
    return kl.sum(dim=-1).mean()


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                discount: float, lam: float, normalize: bool = True) -> torch.Tensor:
    """GAE(lambda) advantages for a ``(T, num_envs)`` rollout.

    The value baseline matters more than it looks under a sparse reward: with
    every reward zero (a policy that has not yet reached the goal), a plain
    discounted-return advantage is identically zero once normalised, so the
    surrogate has zero gradient and nothing learns. Bootstrapping through
    ``values`` keeps the advantage informative because successive state values
    still differ.
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[1], device=rewards.device)
    for t in reversed(range(T)):
        not_done = (~dones[t]).float()
        next_value = values[t + 1] if t < T - 1 else values[t]
        delta = rewards[t] + discount * next_value * not_done - values[t]
        running = delta + discount * lam * not_done * running
        advantages[t] = running
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages


def policy_gaussian_kl(policy_p, policy_q, observations: torch.Tensor) -> torch.Tensor:
    """KL between two skrl Gaussian policies evaluated on ``observations``."""
    with torch.no_grad():
        _, out_p = policy_p.act({"observations": observations}, role="policy")
        _, out_q = policy_q.act({"observations": observations}, role="policy")
        return gaussian_kl(
            out_p["mean_actions"], out_p["log_std"],
            out_q["mean_actions"], out_q["log_std"],
        )

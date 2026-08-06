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


def return_to_go(b_t: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """Discounted return-to-go over a ``(T, N)`` batch, reset at episode ends
    (``dones[t]`` marks the terminal step of an episode, matching the convention
    of ``estimate_advantages``/``compute_gae`` below)."""
    T = b_t.shape[0]
    G = torch.zeros_like(b_t)
    running = torch.zeros(b_t.shape[1], device=b_t.device)
    for t in reversed(range(T)):
        running = b_t[t] + gamma * running * (~dones[t]).float()
        G[t] = running
    return G


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


def estimate_advantages(rewards: torch.Tensor, terminated: torch.Tensor,
                        truncated: torch.Tensor, values: torch.Tensor,
                        gamma: float, gae: float) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE(lambda) advantages **and** value targets for a ``(T, num_envs)`` rollout.

    Port of ``utils/rl.py``'s ``estimate_advantages``. Terminations and time-limit
    truncations are handled separately, as the original does: the value bootstraps
    across a *truncation* (the episode was cut by the clock, its value still
    matters) but not across a true *termination*, while the GAE chain is severed at
    either kind of boundary. Returns ``(advantages, returns)`` — the returns are the
    critic's regression targets, so this is used to fit the per-option critics.
    """
    T = rewards.shape[0]
    deltas = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    dones = (terminated.bool() | truncated.bool()).float()
    term = terminated.float()

    prev_value = torch.zeros(rewards.shape[1], device=rewards.device)
    prev_adv = torch.zeros(rewards.shape[1], device=rewards.device)
    for t in reversed(range(T)):
        bootstrap = 1.0 - term[t]           # value survives a truncation
        chain = 1.0 - dones[t]              # advantage chain severed at any boundary
        deltas[t] = rewards[t] + gamma * prev_value * bootstrap - values[t]
        advantages[t] = deltas[t] + gamma * gae * prev_adv * chain
        prev_value = values[t]
        prev_adv = advantages[t]
    returns = values + advantages
    return advantages, returns


def compute_policy_kl(old_actor, actor, observations: torch.Tensor) -> torch.Tensor:
    """Differentiable ``KL(old_actor || actor)`` on ``observations``.

    ``old_actor``'s statistics are detached, so the result differentiates only
    through ``actor`` — exactly what the TRPO Hessian-vector product needs.
    """
    with torch.no_grad():
        _, out_old = old_actor.act({"observations": observations}, role="policy")
    _, out_new = actor.act({"observations": observations}, role="policy")
    return gaussian_kl(out_old["mean_actions"].detach(), out_old["log_std"].detach(),
                       out_new["mean_actions"], out_new["log_std"])


def hessian_vector_product(kl_fn, model: nn.Module, damping: float,
                           v: torch.Tensor) -> torch.Tensor:
    """``(H + damping*I) v`` where ``H`` is the Hessian of ``kl_fn`` w.r.t. ``model``.

    Port of ``utils/rl.py``'s ``hessian_vector_product`` (Fisher-vector product for
    TRPO). ``allow_unused`` guards against parameters the KL has no path to.
    """
    kl = kl_fn()
    grads = torch.autograd.grad(kl, list(model.parameters()), create_graph=True,
                                allow_unused=True)
    grads = tuple(g if g is not None else torch.zeros_like(p)
                  for p, g in zip(model.parameters(), grads))
    flat_grads = torch.cat([g.view(-1) for g in grads])
    gv = (flat_grads * v).sum()
    hv = torch.autograd.grad(gv, list(model.parameters()), allow_unused=True)
    hv = tuple(h if h is not None else torch.zeros_like(p)
               for p, h in zip(model.parameters(), hv))
    flat_hv = torch.cat([h.contiguous().view(-1) for h in hv])
    return flat_hv + damping * v


def conjugate_gradients(Av_func, b: torch.Tensor, nsteps: int = 10,
                        tol: float = 1e-10) -> torch.Tensor:
    """Solve ``A x = b`` for ``x`` with conjugate gradients (``A`` via ``Av_func``)."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        Avp = Av_func(p)
        alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        if new_rdotr < tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    return x

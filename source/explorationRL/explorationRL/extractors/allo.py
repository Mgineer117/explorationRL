"""ALLO — Augmented-Lagrangian Laplacian Objective representation learner.

Port of ``extractor/extractor.py``'s ``ALLO``. It learns an embedding
``phi: S -> R^d`` whose coordinates approximate the eigenfunctions of the MDP's
graph Laplacian, by minimising

    graph loss        sum_j E[(phi_j(s) - phi_j(s'))^2]      (temporal smoothness)
    + dual loss       <Lambda, tril(<phi, phi> - I)>          (orthonormality)
    + barrier loss    b * sum (tril errors)^2

with ``s'`` drawn from a discounted future of ``s``, and the dual variables
``Lambda`` / barrier coefficient ``b`` updated by ascent after each step. The
diagonal of ``Lambda`` converges to the Laplacian eigenvalues, and the learned
coordinates are the eigenvector directions the option-based agents (IRPO, HRL)
turn into intrinsic rewards.

Differences from the legacy version, both mechanical:

* Rollouts arrive in skrl's ``(T, num_envs)`` layout rather than one flat
  concatenated stream, so episode bookkeeping is a suffix-min over the done
  flags and both samplers are fully vectorised (the legacy code looped over
  episodes in Python).
* Plot/​logging side effects were dropped; ``learn`` returns a scalar dict the
  caller can hand to ``track_data``.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn

from explorationRL.extractors.networks import FeatureMLP


@dataclasses.dataclass
class ALLO_CFG:
    """Hyperparameters for :class:`ALLO` (names mirror the legacy extractor)."""

    feature_dim: int = 8
    """Embedding width ``d`` — the number of eigenvector directions learned."""

    hidden_dim: list = dataclasses.field(default_factory=lambda: [256, 256])
    """Hidden widths of the encoder MLP."""

    positional_indices: list | None = None
    """Observation slice fed to the encoder (e.g. ``[0, 1]`` for xy). ``None`` uses
    the whole observation."""

    learning_rate: float = 1e-3
    batch_size: int = 512
    discount: float = 0.9
    """Discount of the geometric distribution the future state ``s'`` is drawn from."""

    lr_barrier_coeff: float = 0.1
    lr_duals: float = 1e-3
    lr_dual_velocities: float = 0.1
    min_duals: float = 0.0
    max_duals: float = 100.0
    min_barrier_coeff: float = 0.0
    max_barrier_coeff: float = 100.0
    use_barrier_for_duals: float = 0.0


class ALLO(nn.Module):
    """Laplacian representation learner (augmented-Lagrangian formulation)."""

    def __init__(self, observation_dim: int, cfg: ALLO_CFG | None = None,
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.cfg = cfg or ALLO_CFG()
        self.device = device
        self.d = int(self.cfg.feature_dim)

        in_dim = (len(self.cfg.positional_indices)
                  if self.cfg.positional_indices is not None else int(observation_dim))
        self.network = FeatureMLP(in_dim, list(self.cfg.hidden_dim), self.d, device=device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.cfg.learning_rate)

        # Augmented-Lagrangian state (lower-triangular; see the module docstring).
        z = lambda: torch.zeros((self.d, self.d), device=device)
        self.dual_variables = z()
        self.dual_velocities = z()
        self.barrier_coeffs = torch.tril(2.0 * torch.ones((self.d, self.d), device=device))
        self.nupdates = 0
        self.to(device)

    # ── encoding ──────────────────────────────────────────────────────────── #
    def _slice(self, observations: torch.Tensor) -> torch.Tensor:
        if self.cfg.positional_indices is not None:
            return observations[..., self.cfg.positional_indices]
        return observations

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features, _ = self.network(self._slice(observations))
        return features

    # ── sampling ──────────────────────────────────────────────────────────── #
    @staticmethod
    def _episode_end(dones: torch.Tensor) -> torch.Tensor:
        """``end[t, e]`` = first index >= t where env ``e`` is done (else T-1)."""
        T, N = dones.shape
        idx = torch.arange(T, device=dones.device).view(T, 1).expand(T, N)
        sentinel = torch.full_like(idx, T - 1)
        pos = torch.where(dones, idx, sentinel)
        # suffix-min: earliest done at or after each t
        end = torch.flip(torch.cummin(torch.flip(pos, [0]), dim=0).values, [0])
        return torch.clamp(end, max=T - 1)

    def _discounted_delta(self, ranges: torch.Tensor) -> torch.Tensor:
        """Inverse-transform sample from a truncated geometric over ``[0, ranges]``."""
        discount = float(self.cfg.discount)
        seeds = torch.rand(ranges.shape, device=ranges.device)
        if discount == 0.0:
            return torch.zeros_like(ranges)
        if discount == 1.0:
            return (seeds * ranges).floor().long()
        r = ranges.to(torch.float32)
        discount_pow = discount ** r
        samples = torch.log1p(-(1 - discount_pow) * seeds) / torch.log(
            torch.tensor(discount, device=ranges.device)
        )
        return samples.floor().long()

    def sample_discounted_pairs(self, observations: torch.Tensor, dones: torch.Tensor):
        """Sample ``(s, s')`` with ``s'`` a discounted-future state in the same episode."""
        T, N = observations.shape[0], observations.shape[1]
        B = int(self.cfg.batch_size)
        end = self._episode_end(dones)

        env = torch.randint(N, (B,), device=observations.device)
        t1 = torch.randint(max(1, T - 1), (B,), device=observations.device)
        max_delta = torch.clamp(end[t1, env] - t1, min=0)
        delta = torch.clamp(self._discounted_delta(max_delta) + 1, min=1)
        t2 = torch.clamp(t1 + delta, max=T - 1)
        return observations[t1, env], observations[t2, env]

    def sample_steps(self, observations: torch.Tensor) -> torch.Tensor:
        """Uniformly sample states (used for the orthogonality term)."""
        T, N = observations.shape[0], observations.shape[1]
        B = int(self.cfg.batch_size)
        env = torch.randint(N, (B,), device=observations.device)
        t = torch.randint(T, (B,), device=observations.device)
        return observations[t, env]

    # ── learning ──────────────────────────────────────────────────────────── #
    def learn(self, observations: torch.Tensor, dones: torch.Tensor) -> dict:
        """One ALLO update on a ``(T, num_envs, obs_dim)`` rollout."""
        s1, s2 = self.sample_discounted_pairs(observations, dones)
        phi1, phi2 = self(s1), self(s2)
        n, d = phi1.shape

        # Temporal smoothness: consecutive-in-time states embed close together.
        graph_loss = ((phi1 - phi2) ** 2).mean(dim=0).sum()

        # Orthonormality: <phi, phi>/n should be the identity. Two independent
        # draws give an unbiased product for the barrier term.
        u1, u2 = self(self.sample_steps(observations)), self(self.sample_steps(observations))
        identity = torch.eye(d, device=phi1.device)
        err1 = torch.tril((u1.T @ u1.detach()) / n - identity)
        err2 = torch.tril((u2.T @ u2.detach()) / n - identity)
        error_matrix = 0.5 * (err1 + err2)
        quadratic_error = err1 * err2

        dual_loss = (self.dual_variables.detach() * error_matrix).sum()
        barrier_loss = quadratic_error.sum()
        loss = graph_loss + dual_loss + self.barrier_coeffs[0, 0].detach() * barrier_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # ── dual ascent (with velocity smoothing) ─────────────────────────── #
        with torch.no_grad():
            scaled = 1 + self.cfg.use_barrier_for_duals * (self.barrier_coeffs[0, 0].item() - 1)
            updated = torch.clamp(
                self.dual_variables + self.cfg.lr_duals * scaled * torch.tril(error_matrix),
                min=self.cfg.min_duals, max=self.cfg.max_duals,
            )
            delta = updated - self.dual_variables
            norm_vel = torch.norm(self.dual_velocities)
            init = float(torch.isclose(norm_vel, torch.zeros_like(norm_vel), rtol=1e-10, atol=1e-13))
            rate = init + (1 - init) * self.cfg.lr_dual_velocities
            self.dual_velocities += rate * (delta - self.dual_velocities)
            self.dual_variables.copy_(torch.tril(updated))

            # ── barrier coefficient ascent ────────────────────────────────── #
            bump = torch.clamp(quadratic_error, min=0.0).mean()
            self.barrier_coeffs.copy_(torch.clamp(
                self.barrier_coeffs + self.cfg.lr_barrier_coeff * bump,
                min=self.cfg.min_barrier_coeff, max=self.cfg.max_barrier_coeff,
            ))

        self.nupdates += 1
        return {
            "ALLO / loss": loss.item(),
            "ALLO / graph loss": graph_loss.item(),
            "ALLO / dual loss": dual_loss.item(),
            "ALLO / barrier loss": barrier_loss.item(),
            "ALLO / dual norm": torch.linalg.norm(self.dual_variables).item(),
            "ALLO / barrier coeff": self.barrier_coeffs[0, 0].item(),
        }

    @torch.no_grad()
    def eigenvalues(self) -> torch.Tensor:
        """Current Laplacian eigenvalue estimates (the dual diagonal)."""
        return self.dual_variables.diag()

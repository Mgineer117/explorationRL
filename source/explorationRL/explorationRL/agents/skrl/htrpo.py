"""HTRPO — Hindsight TRPO, as a skrl agent.

Port of ``policy/htrpo.py`` (the legacy ``HTRPO_Learner``) onto skrl's TRPO.
After each rollout the batch is augmented with goal-relabelled copies: a set of
hindsight goals ``g'`` is drawn from the achieved goals, every transition's
*desired goal* slice of the observation is overwritten with ``g'``, and its
reward/termination are recomputed against ``g'``. The policy is then updated by
TRPO on the union of the original and relabelled data.

Layout assumption. The observation carries an achieved-goal slice and a
desired-goal slice at fixed indices (``achieved_goal_index`` /
``desired_goal_index``). For ``PointMaze-v1`` the observation is
``[x, y, vx, vy, ax, ay, gx, gy]``, so these are ``[4, 5]`` and ``[6, 7]`` —
the same slices the legacy ``ACHIEVED_GOAL_IDX``/``DESIRED_GOAL_IDX`` tables
selected with ``[-4, -3]`` / ``[-2, -1]``.

Hindsight goal filtering (HGF) picks a *diverse* goal set: candidates within
``goal_distance_threshold`` of some original goal are kept, then greedily
farthest-point sampled so the relabelled goals spread over the visited region
instead of clustering.

Weighted importance sampling (eq. 79 of the paper). A relabelled transition was
generated under the *original* goal, so it needs correcting. The per-step ratio
is ``pi(a|s,g') / pi(a|s,g)``; these are accumulated along the trajectory
(``cumprod``) and normalised **across trajectories at the same timestep**. This
port keeps rollouts in skrl's ``(T, num_envs)`` layout, so a trajectory is a
column and the WIS normalisation is a plain reduction over the env axis — the
variable-length episode bookkeeping the legacy flat-array version needed
disappears. Steps after a relabelled episode's first goal-reach get weight 0,
which is this layout's equivalent of the legacy ``valid_mask`` truncation.

The weights are applied by scaling TRPO's advantages (its surrogate is
``ratio * advantages``, so scaling the advantage scales each sample's
contribution), via a small memory proxy — which lets the whole update reuse
skrl's TRPO implementation unchanged.
"""

from __future__ import annotations

import dataclasses

import torch

from skrl.agents.torch.trpo import TRPO
from skrl.agents.torch.trpo.trpo_cfg import TRPO_CFG
from skrl.memories.torch import RandomMemory


@dataclasses.dataclass(kw_only=True)
class HTRPO_CFG(TRPO_CFG):
    """TRPO config + the hindsight knobs (names mirror ``policy/htrpo.py``)."""

    achieved_goal_index: list = dataclasses.field(default_factory=lambda: [4, 5])
    """Observation indices holding the achieved goal."""

    desired_goal_index: list = dataclasses.field(default_factory=lambda: [6, 7])
    """Observation indices holding the desired goal (overwritten when relabelling)."""

    goal_distance_threshold: float = 0.45
    """Success radius used to recompute the relabelled reward/termination."""

    num_hindsight_goals: int = 4
    """Number of relabelled copies of the rollout. Each one multiplies the update
    batch by 1x, so keep it small under massively parallel envs."""

    use_hindsight_goal_filtering: bool = True
    """Use HGF (diverse farthest-point goal selection) instead of uniform sampling."""

    use_weighted_importance_sampling: bool = True
    """Apply the eq.-79 WIS correction to the relabelled samples."""


class _WISMemoryProxy:
    """Memory wrapper that scales the ``advantages`` TRPO writes back.

    TRPO computes GAE and then stores it with ``set_tensor_by_name("advantages",
    ...)``. Intercepting exactly that write is the least invasive way to weight
    each sample's surrogate contribution without forking TRPO's update.
    """

    def __init__(self, memory, weights: torch.Tensor):
        self._memory = memory
        self._weights = weights

    def set_tensor_by_name(self, name: str, tensor: torch.Tensor):
        if name == "advantages":
            tensor = tensor * self._weights
        return self._memory.set_tensor_by_name(name, tensor)

    def __getattr__(self, name):
        return getattr(self._memory, name)

    def __len__(self):
        return len(self._memory)


class HTRPO(TRPO):
    """TRPO updated on a hindsight-relabelled, importance-weighted batch."""

    # ── goal selection ────────────────────────────────────────────────────── #
    def _select_goals(self, achieved: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
        """Pick ``num_hindsight_goals`` goals from the achieved-goal cloud."""
        candidates = achieved.reshape(-1, achieved.shape[-1])
        n_goals = int(self.cfg.num_hindsight_goals)
        if candidates.shape[0] == 0:
            return candidates[:0]

        if not self.cfg.use_hindsight_goal_filtering:
            idx = torch.randint(candidates.shape[0], (n_goals,), device=candidates.device)
            return candidates[idx]

        # HGF: keep candidates close to some ORIGINAL goal (i.e. plausibly
        # goal-like), then greedily spread the selection out.
        originals = desired.reshape(-1, desired.shape[-1])
        # Subsample to bound the pairwise distance matrix.
        if candidates.shape[0] > 4096:
            candidates = candidates[torch.randperm(candidates.shape[0], device=candidates.device)[:4096]]
        if originals.shape[0] > 1024:
            originals = originals[torch.randperm(originals.shape[0], device=originals.device)[:1024]]

        d_to_orig = torch.cdist(candidates, originals).min(dim=1).values
        valid = candidates[d_to_orig < self.cfg.goal_distance_threshold]
        if valid.shape[0] == 0:
            # Nothing reached a real goal yet — fall back to the closest ones.
            order = torch.argsort(d_to_orig)
            return candidates[order[:n_goals]]

        # Greedy farthest-point sampling for diversity.
        selected = [valid[torch.randint(valid.shape[0], (1,), device=valid.device).item()]]
        while len(selected) < n_goals and valid.shape[0] > 0:
            sel = torch.stack(selected)
            d = torch.cdist(valid, sel).min(dim=1).values
            selected.append(valid[int(torch.argmax(d).item())])
        while len(selected) < n_goals:  # pad if the pool was tiny
            selected.append(valid[torch.randint(valid.shape[0], (1,), device=valid.device).item()])
        return torch.stack(selected[:n_goals])

    # ── hindsight batch ───────────────────────────────────────────────────── #
    def _build_hindsight_memory(self):
        """Return (augmented memory, WIS weights, augmented next-observations).

        Returns ``None`` to fall back to a plain TRPO update.
        """
        mem = self.memory
        try:
            observations = mem.get_tensor_by_name("observations")
            actions = mem.get_tensor_by_name("actions")
            log_prob = mem.get_tensor_by_name("log_prob")
            rewards = mem.get_tensor_by_name("rewards")
            terminated = mem.get_tensor_by_name("terminated")
            truncated = mem.get_tensor_by_name("truncated")
        except Exception:  # noqa: BLE001 — never break training over this
            return None
        # TRPO bootstraps GAE from the final next-observation, so that tensor has
        # to be widened to match the augmented batch as well.
        next_observations = getattr(self, "_current_next_observations", None)
        if next_observations is None:
            return None

        ag = [i % observations.shape[-1] for i in self.cfg.achieved_goal_index]
        dg = [i % observations.shape[-1] for i in self.cfg.desired_goal_index]
        T, N = observations.shape[0], observations.shape[1]

        goals = self._select_goals(observations[..., ag], observations[..., dg])
        if goals.shape[0] == 0:
            return None

        obs_groups = [observations]
        rew_groups = [rewards]
        term_groups = [terminated]
        trunc_groups = [truncated]
        next_obs_groups = [next_observations]
        # Original data needs no IS correction (legacy: goal_id == 0 -> weight 1).
        wis_groups = [torch.ones(T, N, 1, device=observations.device)]

        for g in goals:
            obs_g = observations.clone()
            obs_g[..., dg] = g
            next_obs_g = next_observations.clone()
            next_obs_g[..., dg] = g
            next_obs_groups.append(next_obs_g)
            dist = torch.linalg.norm(obs_g[..., ag] - g, dim=-1)          # (T, N)
            reached = dist < self.cfg.goal_distance_threshold

            rew_g = reached.float().unsqueeze(-1)
            # Terminate at the FIRST reach of the relabelled goal in each column.
            first = reached & (reached.cumsum(dim=0) == 1)
            term_g = first.unsqueeze(-1)

            obs_groups.append(obs_g)
            rew_groups.append(rew_g)
            term_groups.append(term_g)
            trunc_groups.append(torch.zeros_like(truncated))
            wis_groups.append(self._wis_weights(obs_g, actions, log_prob, reached))

        aug_obs = torch.cat(obs_groups, dim=1)
        aug_rew = torch.cat(rew_groups, dim=1)
        aug_term = torch.cat(term_groups, dim=1)
        aug_trunc = torch.cat(trunc_groups, dim=1)
        aug_wis = torch.cat(wis_groups, dim=1)
        aug_actions = actions.repeat(1, len(obs_groups), 1)
        aug_next_obs = torch.cat(next_obs_groups, dim=0)

        # Values and old log-probs must be recomputed: both are functions of the
        # observation, which relabelling changed. Using the current policy's
        # log-prob makes TRPO's ratio start at 1, so the WIS weights above carry
        # the whole off-goal correction.
        with torch.no_grad():
            flat_obs = aug_obs.reshape(-1, aug_obs.shape[-1])
            flat_act = aug_actions.reshape(-1, aug_actions.shape[-1])
            inputs = {"observations": self._observation_preprocessor(flat_obs), "states": None}
            values, _ = self.value.act(inputs, role="value")
            _, out = self.policy.act({**inputs, "taken_actions": flat_act}, role="policy")
            aug_values = values.reshape(aug_obs.shape[0], aug_obs.shape[1], 1)
            aug_logp = out["log_prob"].reshape(aug_obs.shape[0], aug_obs.shape[1], 1)

        total_envs = aug_obs.shape[1]
        aug_mem = RandomMemory(memory_size=T, num_envs=total_envs, device=self.device)
        aug_mem.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
        aug_mem.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
        aug_mem.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
        for name, dtype in (("rewards", torch.float32), ("terminated", torch.bool),
                            ("truncated", torch.bool), ("log_prob", torch.float32),
                            ("values", torch.float32), ("returns", torch.float32),
                            ("advantages", torch.float32)):
            aug_mem.create_tensor(name=name, size=1, dtype=dtype)

        aug_mem.set_tensor_by_name("observations", aug_obs)
        aug_mem.set_tensor_by_name("actions", aug_actions)
        aug_mem.set_tensor_by_name("rewards", aug_rew)
        aug_mem.set_tensor_by_name("terminated", aug_term.bool())
        aug_mem.set_tensor_by_name("truncated", aug_trunc.bool())
        aug_mem.set_tensor_by_name("log_prob", aug_logp)
        aug_mem.set_tensor_by_name("values", aug_values)
        aug_mem.filled = True  # so len(memory) == T * total_envs

        self.track_data("HTRPO / Hindsight goals", float(goals.shape[0]))
        self.track_data("HTRPO / Augmented batch envs", float(total_envs))
        self.track_data("HTRPO / Relabelled reward (mean)", float(aug_rew[:, N:].mean().item()))
        return aug_mem, aug_wis, aug_next_obs

    def _wis_weights(self, obs_g: torch.Tensor, actions: torch.Tensor,
                     log_prob_orig: torch.Tensor, reached: torch.Tensor) -> torch.Tensor:
        """Eq.-79 WIS weights for one relabelled group, shape ``(T, N, 1)``."""
        T, N = obs_g.shape[0], obs_g.shape[1]
        with torch.no_grad():
            flat_obs = self._observation_preprocessor(obs_g.reshape(-1, obs_g.shape[-1]))
            flat_act = actions.reshape(-1, actions.shape[-1])
            _, out = self.policy.act({"observations": flat_obs, "states": None,
                                      "taken_actions": flat_act}, role="policy")
            logp_new = out["log_prob"].reshape(T, N, 1)

            if not self.cfg.use_weighted_importance_sampling:
                w = torch.ones(T, N, 1, device=obs_g.device)
            else:
                # per-step ratio pi(a|s,g') / pi(a|s,g), accumulated along the trajectory
                ratio = torch.exp(torch.clamp(logp_new - log_prob_orig, -10.0, 10.0))
                cum = torch.cumprod(ratio, dim=0)
                # normalise ACROSS trajectories at each timestep (the env axis)
                denom = cum.sum(dim=1, keepdim=True) + 1e-8
                w = cum * N / denom

            # Drop everything after the relabelled episode's first success — the
            # legacy code physically truncated those steps out of the batch.
            after = (reached.cumsum(dim=0) > 1).unsqueeze(-1)
            w = w.masked_fill(after, 0.0)
        return w

    # ── skrl hook ─────────────────────────────────────────────────────────── #
    def update(self, *, timestep: int, timesteps: int) -> None:
        built = self._build_hindsight_memory()
        if built is None:
            super().update(timestep=timestep, timesteps=timesteps)
            return
        aug_mem, wis, aug_next_obs = built
        original_memory = self.memory
        original_next_obs = self._current_next_observations
        self.memory = _WISMemoryProxy(aug_mem, wis) if self.cfg.use_weighted_importance_sampling else aug_mem
        self._current_next_observations = aug_next_obs
        try:
            super().update(timestep=timestep, timesteps=timesteps)
        finally:
            self.memory = original_memory
            self._current_next_observations = original_next_obs

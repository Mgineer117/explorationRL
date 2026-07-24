"""HRL — hierarchical RL over intrinsic-reward options, as a skrl agent.

Port of ``policy/hrl.py`` + ``trainer/hrl_trainer.py`` (the legacy
``HRL_Learner`` and its two-phase trainer) onto skrl.

Structure, mirroring the legacy design:

* **Options.** ``num_options`` low-level policies, each trained on one signed
  eigenvector direction of the ALLO representation — the same option machinery
  IRPO uses (``explorationRL.agents.skrl.intrinsic``).
* **Controller.** A high-level categorical policy over the options. It picks an
  option, that option drives the env for up to ``option_horizon`` steps, then
  the controller picks again — a semi-MDP, which is what the legacy trainer's
  ``option_termination`` loop implemented.
* **Credit assignment.** The controller is trained on the *discounted return
  accumulated over each option segment*, attributed to the decision step, rather
  than per-step reward. That is the SMDP policy gradient the hierarchy requires.

Why the controller is built here instead of coming from the yaml. skrl's model
instantiator sizes ``policy`` from the env's action space — continuous ``Box(2)``
for PointMaze. The controller's action is an *option index*, so it needs a
categorical head of width ``num_options``. The yaml's ``policy`` block is
therefore used as the template each option policy is cloned from, and the
controller is a small MLP constructed in ``__init__``.

Legacy Phase 1 (pretrain options, then freeze and train the controller) is
expressed here as concurrent training: options keep learning from their
intrinsic rewards while the controller learns to sequence them. Set
``option_learning_rate: 0.0`` to recover the frozen-option behaviour.
"""

from __future__ import annotations

import copy
import dataclasses

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG

from explorationRL.agents.skrl.common import compute_gae

from explorationRL.agents.skrl.intrinsic import ALLOIntrinsicRewards, RandomIntrinsicRewards
from explorationRL.extractors import ALLO, ALLO_CFG


@dataclasses.dataclass(kw_only=True)
class HRL_CFG(PPO_CFG):
    """PPO config + the hierarchy knobs (names mirror ``policy/hrl.py``)."""

    num_options: int = 4
    """Number of low-level options the controller chooses between."""

    option_horizon: int = 8
    """Max steps an option runs before the controller re-selects (semi-MDP ``K``)."""

    controller_learning_rate: float = 3e-4
    controller_epochs: int = 4
    controller_ratio_clip: float = 0.2
    controller_hidden_dim: list = dataclasses.field(default_factory=lambda: [256, 256])
    controller_entropy_scale: float = 0.01

    option_learning_rate: float = 3e-4
    """Adam LR for the option policies. 0.0 freezes them (legacy Phase-2 behaviour)."""

    option_epochs: int = 4
    option_ratio_clip: float = 0.2
    intrinsic_discount: float = 0.99
    intrinsic_reward_type: str = "allo"

    extractor_feature_dim: int = 8
    extractor_hidden_dim: list = dataclasses.field(default_factory=lambda: [256, 256])
    extractor_learning_rate: float = 1e-3
    extractor_batch_size: int = 512
    extractor_discount: float = 0.9
    extractor_updates_per_rollout: int = 1
    extractor_positional_indices: list | None = dataclasses.field(
        default_factory=lambda: [0, 1]
    )


class _Controller(nn.Module):
    """Categorical policy over option indices."""

    def __init__(self, obs_dim: int, num_options: int, hidden: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, int(h)), nn.ELU()]
            d = int(h)
        layers.append(nn.Linear(d, num_options))
        self.net = nn.Sequential(*layers)

    def distribution(self, observations: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.net(observations))


class HRL(PPO):
    """Controller over intrinsic-reward options, with semi-MDP execution."""

    def __init__(self, *, models, memory=None, observation_space=None, state_space=None,
                 action_space=None, device=None, cfg=None):
        super().__init__(
            models=models, memory=memory, observation_space=observation_space,
            state_space=state_space, action_space=action_space, device=device, cfg=cfg,
        )
        c: HRL_CFG = self.cfg
        self.num_options = int(c.num_options)
        obs_dim = int(observation_space.shape[0])

        self.option_policies = nn.ModuleList(
            [copy.deepcopy(self.policy) for _ in range(self.num_options)]
        ).to(self.device)
        self.option_optimizers = [
            torch.optim.Adam(p.parameters(), lr=max(c.option_learning_rate, 1e-12))
            for p in self.option_policies
        ]

        self.controller = _Controller(obs_dim, self.num_options,
                                      list(c.controller_hidden_dim)).to(self.device)
        self.controller_optimizer = torch.optim.Adam(
            self.controller.parameters(), lr=c.controller_learning_rate
        )

        needed = self.num_options // 2 + 2
        self.extractor = ALLO(
            observation_dim=obs_dim,
            cfg=ALLO_CFG(
                feature_dim=max(int(c.extractor_feature_dim), needed),
                hidden_dim=list(c.extractor_hidden_dim),
                positional_indices=c.extractor_positional_indices,
                learning_rate=c.extractor_learning_rate,
                batch_size=c.extractor_batch_size,
                discount=c.extractor_discount,
            ),
            device=self.device,
        )
        reward_cls = RandomIntrinsicRewards if c.intrinsic_reward_type == "random" else ALLOIntrinsicRewards
        self.intrinsic_rewards = reward_cls(self.extractor, self.num_options, device=self.device)

        self.checkpoint_modules["controller"] = self.controller
        self.checkpoint_modules["option_policies"] = self.option_policies
        self.checkpoint_modules["extractor"] = self.extractor

        # Semi-MDP execution state (lazily sized to num_envs on the first act).
        self._active_option: torch.Tensor | None = None
        self._steps_in_option: torch.Tensor | None = None
        # Per-step trace consumed by update(): which option ran, whether this step
        # was a controller decision, and that decision's log-prob.
        self._trace: list[dict] = []

    # ── semi-MDP acting ───────────────────────────────────────────────────── #
    def act(self, observations: torch.Tensor, states: torch.Tensor | None, *,
            timestep: int, timesteps: int):
        if not self.training or timestep < self.cfg.random_timesteps:
            return super().act(observations, states, timestep=timestep, timesteps=timesteps)

        n = observations.shape[0]
        obs_p = self._observation_preprocessor(observations)
        states_p = self._state_preprocessor(states)

        if self._active_option is None or self._active_option.shape[0] != n:
            self._active_option = torch.zeros(n, dtype=torch.long, device=observations.device)
            self._steps_in_option = torch.full((n,), int(self.cfg.option_horizon),
                                               dtype=torch.long, device=observations.device)

        # Value first: a shared policy/value trunk caches its forward between role
        # lookups, so evaluating it after the per-option calls would alias a slice.
        values, _ = self.value.act({"observations": obs_p, "states": states_p}, role="value")
        self._current_values = self._value_preprocessor(values, inverse=True)

        # Controller re-selects wherever the previous option has run its horizon.
        decide = self._steps_in_option >= int(self.cfg.option_horizon)
        ctrl_log_prob = torch.zeros(n, 1, device=observations.device)
        dist = self.controller.distribution(obs_p)
        if bool(decide.any()):
            sampled = dist.sample()
            self._active_option = torch.where(decide, sampled, self._active_option)
            self._steps_in_option = torch.where(
                decide, torch.zeros_like(self._steps_in_option), self._steps_in_option
            )
        ctrl_log_prob[:, 0] = dist.log_prob(self._active_option)

        # The active option drives the env; envs are grouped by option so each
        # option policy runs once per step rather than per-env.
        actions = torch.zeros(n, self.action_space.shape[0], device=observations.device)
        log_probs = torch.zeros(n, 1, device=observations.device)
        for o in range(self.num_options):
            mask = self._active_option == o
            if not bool(mask.any()):
                continue
            a, out = self.option_policies[o].act({"observations": obs_p[mask], "states": None},
                                                 role="policy")
            actions[mask] = a
            log_probs[mask] = out["log_prob"]

        self._steps_in_option = self._steps_in_option + 1
        self._current_log_prob = log_probs
        self._trace.append({
            "option": self._active_option.clone(),
            "decision": decide.clone(),
            "ctrl_log_prob": ctrl_log_prob.detach(),
        })
        return actions, {"log_prob": log_probs}

    # ── updates ───────────────────────────────────────────────────────────── #
    def _discounted(self, reward: torch.Tensor, dones: torch.Tensor, discount: float) -> torch.Tensor:
        out = torch.zeros_like(reward)
        running = torch.zeros(reward.shape[1], device=reward.device)
        for t in reversed(range(reward.shape[0])):
            running = reward[t] + discount * running * (~dones[t]).float()
            out[t] = running
        return out

    def _update_option(self, o: int, obs, act, old_lp, adv) -> float:
        if float(self.cfg.option_learning_rate) <= 0.0 or obs.numel() == 0:
            return 0.0
        policy, optimizer = self.option_policies[o], self.option_optimizers[o]
        last = 0.0
        for _ in range(int(self.cfg.option_epochs)):
            _, out = policy.act({"observations": obs, "states": None, "taken_actions": act},
                                role="policy")
            ratio = torch.exp(out["log_prob"].reshape(-1) - old_lp)
            clip = self.cfg.option_ratio_clip
            loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            last = float(loss.item())
        return last

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
            self._trace.clear()
            return

        T, N = observations.shape[0], observations.shape[1]
        dones = (terminated | truncated).squeeze(-1)
        trace = self._trace[-T:] if len(self._trace) >= T else self._trace
        self._trace.clear()
        if len(trace) != T:  # trace/rollout desync — fall back rather than mis-attribute
            super().update(timestep=timestep, timesteps=timesteps)
            return

        options = torch.stack([s["option"] for s in trace])            # (T, N)
        decisions = torch.stack([s["decision"] for s in trace])        # (T, N)
        ctrl_lp_old = torch.stack([s["ctrl_log_prob"] for s in trace]).squeeze(-1)  # (T, N)

        # 1. Representation.
        for _ in range(int(self.cfg.extractor_updates_per_rollout)):
            for k, v in self.extractor.learn(observations, dones).items():
                self.track_data(k, v)

        # 2. Option policies, each on the steps IT actually drove.
        next_observations = torch.cat([observations[1:], observations[-1:]], dim=0)
        r_int = self.intrinsic_rewards(
            observations.reshape(-1, observations.shape[-1]),
            next_observations.reshape(-1, next_observations.shape[-1]),
        ).reshape(T, N, -1)

        losses = []
        for o in range(self.num_options):
            mask = options == o
            if not bool(mask.any()):
                continue
            adv_full = self._discounted(r_int[..., o], dones, self.cfg.intrinsic_discount)
            adv_full = (adv_full - adv_full.mean()) / (adv_full.std() + 1e-8)
            obs_o = self._observation_preprocessor(observations[mask])
            losses.append(self._update_option(
                o, obs_o, actions[mask], log_prob[mask].reshape(-1), adv_full[mask].reshape(-1)
            ))

        # 3. Controller: SMDP return over each option segment, credited to the
        #    decision step. GAE over the ENV reward (a plain discounted return is
        #    identically zero under a sparse reward -- see common.compute_gae).
        values = self.memory.get_tensor_by_name("values").squeeze(-1)
        adv_c = compute_gae(rewards.squeeze(-1), values, dones,
                            self.cfg.discount_factor, self.cfg.gae_lambda)
        obs_flat = self._observation_preprocessor(observations.reshape(-1, observations.shape[-1]))
        opt_flat = options.reshape(-1)
        adv_flat = adv_c.reshape(-1)
        old_flat = ctrl_lp_old.reshape(-1)
        dec_flat = decisions.reshape(-1)
        if bool(dec_flat.any()):
            obs_d, opt_d = obs_flat[dec_flat], opt_flat[dec_flat]
            adv_d, old_d = adv_flat[dec_flat], old_flat[dec_flat]
            for _ in range(int(self.cfg.controller_epochs)):
                dist = self.controller.distribution(obs_d)
                ratio = torch.exp(dist.log_prob(opt_d) - old_d)
                clip = self.cfg.controller_ratio_clip
                loss = -torch.min(ratio * adv_d, torch.clamp(ratio, 1 - clip, 1 + clip) * adv_d).mean()
                loss = loss - self.cfg.controller_entropy_scale * dist.entropy().mean()
                self.controller_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.controller.parameters(), 1.0)
                self.controller_optimizer.step()
            self.track_data("HRL / Controller loss", float(loss.item()))
            with torch.no_grad():
                self.track_data("HRL / Controller entropy",
                                float(self.controller.distribution(obs_d).entropy().mean().item()))

        self.track_data("HRL / Option loss (mean)", float(sum(losses) / max(1, len(losses))))
        self.track_data("HRL / Decisions per rollout", float(decisions.float().sum().item()))
        # How unevenly the controller uses its options (0 = uniform).
        usage = torch.bincount(options.reshape(-1), minlength=self.num_options).float()
        self.track_data("HRL / Option usage imbalance",
                        float((usage / usage.sum()).std().item()))

        # 4. Value function: reuse PPO's update for the critic on the env reward.
        super().update(timestep=timestep, timesteps=timesteps)

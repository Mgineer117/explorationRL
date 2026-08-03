"""IRPO — Intrinsic Reward Policy Optimization, as a skrl agent.

Port of ``policy/irpo.py`` (the legacy ``IRPO_Learner``) onto skrl, faithful to
the surrogate-gradient formulation of the paper *"Intrinsic Reward Policy
Optimization for Sparse-Reward Environments"* (arXiv:2601.21391): the **base
policy is optimised directly for the extrinsic reward** using a surrogate
gradient assembled from several intrinsic-reward-driven adaptations of that same
policy — no pretrained subpolicies are deployed.

Algorithm (per update, mirroring ``IRPO_Learner.learn``)
--------------------------------------------------------
The ALLO Laplacian representation is learned **once, offline, on a near-uniform
state buffer** (``pretrain_extractor``) and then frozen; its signed eigenvector
directions define ``num_options`` intrinsic rewards. Then, on each skrl rollout:

1. For every option ``i``, clone the base policy and take ``num_exp_updates``
   differentiable inner gradient steps — the first ``num_exp_updates - 1`` on
   option ``i``'s *intrinsic* advantage, the last on the *extrinsic* advantage.
   Per-option intrinsic/extrinsic critics supply GAE advantages.
2. Each option's post-adaptation extrinsic return updates a per-option EMA
   ``perf_gains[i]``.
3. Assemble each option's surrogate gradient of the extrinsic objective w.r.t.
   the *base* parameters by differentiating back through the inner chain
   (``_meta_gradient``; the MAML back-substitution ``g <- g - alpha * H g``).
4. Aggregate the per-option gradients with ``perf_gains``-derived weights
   (softmax / argmax / uniform) and apply them to the base policy by **TRPO**
   with a conjugate-gradient step and a KL line search.

Why this is not plain PPO over pooled option data
-------------------------------------------------
The signed ``+/-`` intrinsic rewards make the *intrinsic* option gradients exact
negatives that cancel on summation — but IRPO never sums those. It sums the
gradients of each adapted policy's *extrinsic* return, which do not cancel, and
the surrogate is what carries a learning signal when the true extrinsic gradient
is near-zero under a sparse reward. That is the whole point of the method.

Single-rollout adaptation
-------------------------
The legacy learner collects a *fresh* rollout for every inner step. skrl's
trainer owns the environment loop and hands ``update()`` one rollout, so the
inner steps here reuse that single batch (batch-MAML). This is the one
mechanical departure from the legacy code; the meta-gradient structure,
aggregation, and TRPO base update are otherwise faithful.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
import os

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG

from explorationRL.agents.skrl.common import (
    compute_policy_kl, conjugate_gradients, estimate_advantages,
    flat_params, hessian_vector_product, set_flat_params,
)
from explorationRL.agents.skrl.intrinsic import ALLOIntrinsicRewards, RandomIntrinsicRewards
from explorationRL.extractors import ALLO, ALLO_CFG

_GAUSSIAN_ENTROPY_CONST = 0.5 * math.log(2.0 * math.pi * math.e)


@dataclasses.dataclass(kw_only=True)
class IRPO_CFG(PPO_CFG):
    """PPO config + the IRPO knobs (names mirror ``policy/irpo.py``)."""

    num_options: int = 4
    """Number of signed eigenvector directions explored (``+/-`` pairs)."""

    num_exp_updates: int = 2
    """Inner adaptation steps per option (``>= 2``): the first ``n-1`` on the
    intrinsic advantage, the last on the extrinsic advantage."""

    inner_learning_rate: float = 0.01
    """Inner-loop step size ``alpha`` used for each exploratory adaptation."""

    entropy_scaler: float = 0.0
    """Entropy bonus weight in the exploratory actor loss."""

    critic_learning_rate: float = 1e-3
    critic_epochs: int = 5
    """Adam LR / epochs for the per-option intrinsic & extrinsic critics."""

    meta_batch_size: int = 4096
    """Transitions subsampled for the second-order (``create_graph``) adaptation
    and the TRPO KL. Caps meta-gradient memory the way the legacy
    ``grad_batch_size`` did; critics still fit on the full rollout."""

    perf_gain_beta: float = 0.9
    """EMA smoothing of each option's post-adaptation extrinsic return."""

    aggregation_method: str = "softmax"
    """How per-option surrogate gradients are combined: ``softmax`` | ``argmax``
    | ``uniform``."""

    aggregation_temperature: float = 1.0
    """Softmax temperature anneals from 1 to 0 over this fraction of training,
    then behaves like ``argmax`` (``<= 0`` disables aggregation → pure argmax)."""

    base_target_kl: float = 0.01
    """TRPO trust-region size for the base-policy update."""

    trpo_damping: float = 0.1
    trpo_cg_iters: int = 10
    trpo_backtrack_iters: int = 10
    trpo_backtrack_coeff: float = 0.7

    intrinsic_reward_type: str = "allo"
    """``allo`` (learned Laplacian) or ``random`` (frozen-encoder ablation)."""

    # ── ALLO representation (learned offline, then frozen) ────────────────────
    extractor_feature_dim: int = 8
    extractor_hidden_dim: list = dataclasses.field(default_factory=lambda: [256, 256])
    extractor_learning_rate: float = 1e-3
    extractor_batch_size: int = 512
    extractor_discount: float = 0.9
    extractor_positional_indices: list | None = dataclasses.field(
        default_factory=lambda: [0, 1]
    )
    """Observation slice the representation is learned over (xy for PointMaze)."""

    extractor_pretrain_epochs: int = 5000
    """Offline ALLO gradient steps on the uniform-coverage buffer. ``0`` skips the
    pretrain and falls back to updating the representation online each rollout."""

    extractor_collect_steps: int = 256
    """Rollout length (per env) of the uniform-random buffer ALLO pretrains on."""

    extractor_uniform_reset: bool = True
    """Collect the pretrain buffer with the env's uniform-cell reset, so ALLO sees
    a near-uniform state distribution (the measure its orthonormality assumes)."""

    extractor_cache_dir: str = "model"
    """Where a pretrained+frozen extractor is cached/reused across runs
    (``""`` disables caching)."""


class IRPO(PPO):
    """PPO rollout + surrogate-gradient (MAML-style) base-policy optimisation."""

    def __init__(self, *, models, memory=None, observation_space=None, state_space=None,
                 action_space=None, device=None, cfg=None):
        super().__init__(
            models=models, memory=memory, observation_space=observation_space,
            state_space=state_space, action_space=action_space, device=device, cfg=cfg,
        )
        c: IRPO_CFG = self.cfg
        self.num_options = int(c.num_options)
        assert int(c.num_exp_updates) >= 2, "num_exp_updates must be at least 2"
        obs_dim = int(observation_space.shape[0])

        # Per-option intrinsic & extrinsic critics (clones of the base value net).
        self.ext_critics = nn.ModuleList(
            [copy.deepcopy(self.value) for _ in range(self.num_options)]
        ).to(self.device)
        self.int_critics = nn.ModuleList(
            [copy.deepcopy(self.value) for _ in range(self.num_options)]
        ).to(self.device)
        self.ext_critic_optim = [torch.optim.Adam(m.parameters(), lr=c.critic_learning_rate)
                                 for m in self.ext_critics]
        self.int_critic_optim = [torch.optim.Adam(m.parameters(), lr=c.critic_learning_rate)
                                 for m in self.int_critics]

        # Snapshots of each option's fully-adapted policy — deployed at eval time
        # (mirrors IRPO_Learner.forward, which returns the best exploratory policy).
        self.final_exp_policies = nn.ModuleList(
            [copy.deepcopy(self.policy) for _ in range(self.num_options)]
        ).to(self.device)
        # Per-option EMA of post-adaptation extrinsic return (the aggregation logits).
        # skrl agents are not nn.Modules, so this is a plain tensor attribute.
        self.perf_gains = torch.zeros(self.num_options, device=self.device)

        needed = self.num_options // 2 + 2  # highest eigenvector index used + 1
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
        self._extractor_frozen = False

        self.checkpoint_modules["extractor"] = self.extractor
        self.checkpoint_modules["ext_critics"] = self.ext_critics
        self.checkpoint_modules["int_critics"] = self.int_critics
        self.checkpoint_modules["final_exp_policies"] = self.final_exp_policies

    # ── offline ALLO pretraining ──────────────────────────────────────────── #
    def _cache_path(self, tag: str | None) -> str | None:
        c: IRPO_CFG = self.cfg
        if not c.extractor_cache_dir:
            return None
        key = "|".join(str(x) for x in (
            tag, c.intrinsic_reward_type, self.extractor.d, list(c.extractor_hidden_dim),
            c.extractor_discount, c.extractor_positional_indices,
            c.extractor_collect_steps, c.extractor_pretrain_epochs, c.extractor_uniform_reset,
        ))
        digest = hashlib.md5(key.encode()).hexdigest()[:12]
        return os.path.join(c.extractor_cache_dir, "irpo_allo", f"ALLO_{digest}.pth")

    def pretrain_extractor(self, env, tag: str | None = None) -> None:
        """Learn the ALLO representation offline on a uniform-coverage buffer, then
        freeze it. Called once by the training script before ``trainer.train()``.

        Data is collected under a uniform-random policy (uniform actions) and, if
        ``extractor_uniform_reset`` and the env supports it, uniform-cell resets —
        so the representation is fit to a near-uniform state distribution rather
        than the learning policy's ever-shifting occupancy. A cached extractor with
        matching hyperparameters is reused instead of recollecting.
        """
        c: IRPO_CFG = self.cfg
        if self._extractor_frozen or int(c.extractor_pretrain_epochs) <= 0:
            return

        path = self._cache_path(tag)
        if path is not None and os.path.exists(path):
            self.extractor.load_state_dict(torch.load(path, map_location=self.device))
            self._freeze_extractor()
            print(f"[INFO] IRPO: loaded pretrained ALLO from {path}.")
            return

        observations, dones = self._collect_uniform_buffer(env)
        print(f"[INFO] IRPO: pretraining ALLO for {int(c.extractor_pretrain_epochs)} "
              f"epochs on {observations.shape[0]}x{observations.shape[1]} transitions.")
        log_every = max(1, int(c.extractor_pretrain_epochs) // 20)
        metrics: dict = {}
        for epoch in range(int(c.extractor_pretrain_epochs)):
            metrics = self.extractor.learn(observations, dones)
            if epoch % log_every == 0 or epoch == int(c.extractor_pretrain_epochs) - 1:
                print(f"[INFO]   ALLO epoch {epoch:5d}  "
                      f"loss={metrics['ALLO / loss']:.4f}  "
                      f"graph={metrics['ALLO / graph loss']:.4f}  "
                      f"dual_norm={metrics['ALLO / dual norm']:.4f}")

        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.extractor.state_dict(), path)
            print(f"[INFO] IRPO: cached pretrained ALLO to {path}.")
        self._freeze_extractor()

    @torch.no_grad()
    def _collect_uniform_buffer(self, env):
        """Roll out a uniform-random policy to a ``(T, num_envs)`` obs/done buffer."""
        c: IRPO_CFG = self.cfg
        base = getattr(env, "unwrapped", env)
        prev_uniform = getattr(base, "_uniform_reset", None)
        if c.extractor_uniform_reset and hasattr(base, "_uniform_reset"):
            base._uniform_reset = True

        T = int(c.extractor_collect_steps)
        act_dim = int(self.action_space.shape[0])
        observations, _ = env.reset()
        n = observations.shape[0]
        obs_buf = torch.empty(T, n, observations.shape[-1], device=self.device)
        done_buf = torch.empty(T, n, dtype=torch.bool, device=self.device)
        for t in range(T):
            actions = torch.rand(n, act_dim, device=self.device) * 2.0 - 1.0
            next_obs, _, terminated, truncated, _ = env.step(actions)
            obs_buf[t] = observations
            done_buf[t] = (terminated.bool() | truncated.bool()).reshape(n)
            observations = next_obs

        # Restore the env for the RL task: fixed start, clean episode buffers.
        if prev_uniform is not None:
            base._uniform_reset = prev_uniform
        env.reset()
        return obs_buf, done_buf

    def _freeze_extractor(self) -> None:
        for p in self.extractor.parameters():
            p.requires_grad_(False)
        self.extractor.eval()
        self._extractor_frozen = True

    # ── eval-time acting ──────────────────────────────────────────────────── #
    def act(self, observations: torch.Tensor, states: torch.Tensor | None, *,
            timestep: int, timesteps: int):
        # Training: the base policy drives every env (standard PPO rollout).
        if self.training:
            return super().act(observations, states, timestep=timestep, timesteps=timesteps)
        # Eval: deploy the best adapted exploratory policy (as IRPO_Learner.forward).
        idx = int(torch.argmax(self.perf_gains).item())
        obs_p = self._observation_preprocessor(observations)
        actions, outputs = self.final_exp_policies[idx].act(
            {"observations": obs_p, "states": None}, role="policy")
        return actions, outputs

    # ── advantage helpers ─────────────────────────────────────────────────── #
    def _critic_values(self, critic: nn.Module, obs_p: torch.Tensor, shape) -> torch.Tensor:
        values, _ = critic.act({"observations": obs_p, "states": None}, role="value")
        return values.reshape(shape)

    @torch.no_grad()
    def _advantages(self, critic, obs_p, rewards, terminated, truncated, shape):
        """GAE advantages + returns for a reward tensor, using ``critic`` as
        baseline. Detached (no_grad): advantages are constants in the surrogate and
        returns are fixed regression targets for the critic."""
        values = self._critic_values(critic, obs_p, shape)
        adv, ret = estimate_advantages(rewards, terminated, truncated, values,
                                       self.cfg.discount_factor, self.cfg.gae_lambda)
        return adv, ret

    def _actor_loss(self, actor, obs_p, actions, advantages) -> torch.Tensor:
        """Entropy-regularised policy-gradient surrogate (legacy ``actor_loss``)."""
        _, out = actor.act({"observations": obs_p, "states": None, "taken_actions": actions},
                           role="policy")
        logprob = out["log_prob"].reshape(-1)
        entropy = (out["log_std"] + _GAUSSIAN_ENTROPY_CONST).sum(-1).mean()
        return -(logprob * advantages).mean() - self.cfg.entropy_scaler * entropy

    def _fit_critic(self, critic, optimizer, obs_p, returns) -> None:
        target = returns.reshape(-1, 1)
        for _ in range(int(self.cfg.critic_epochs)):
            values, _ = critic.act({"observations": obs_p, "states": None}, role="value")
            loss = ((values - target) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ── one option's adaptation + surrogate gradient ──────────────────────── #
    @staticmethod
    def _clean_clone(module: nn.Module) -> nn.Module:
        """Deep-copy a skrl model safely. A forwarded GaussianMixin caches a
        non-leaf ``_g_distribution`` that ``deepcopy`` cannot copy; clear it first."""
        if hasattr(module, "_g_distribution"):
            module._g_distribution = None
        return copy.deepcopy(module)

    def _adapt_option(self, i, obs_p, actions_flat, int_adv, ext_adv):
        """Clone the base policy, adapt it for option ``i``, and return the
        surrogate extrinsic gradient w.r.t. the base parameters plus the adapted
        policy and its post-adaptation extrinsic return proxy."""
        alpha = float(self.cfg.inner_learning_rate)
        J = int(self.cfg.num_exp_updates)
        # Pre-build clean clones up front (each round needs independent leaves so
        # _meta_gradient can back-substitute round by round). Forwarding a clone
        # caches a non-leaf distribution on it, so they must be copied *before*
        # any forward pass — not lazily inside the loop.
        policies = [self._clean_clone(self.policy) for _ in range(J + 1)]
        grad_rounds = []
        for j in range(J):
            actor = policies[j]
            adv = ext_adv if j == J - 1 else int_adv
            loss = self._actor_loss(actor, obs_p, actions_flat, adv)
            grads = torch.autograd.grad(loss, list(actor.parameters()),
                                        create_graph=True, allow_unused=True)
            grads = tuple(g if g is not None else torch.zeros_like(p)
                          for p, g in zip(actor.parameters(), grads))
            grad_rounds.append(grads)
            # Data-only inner step onto the next (already clean) clone.
            with torch.no_grad():
                for p_next, p_cur, g in zip(policies[j + 1].parameters(),
                                            actor.parameters(), grads):
                    p_next.data.copy_(p_cur.data - alpha * g.detach())

        surrogate = self._meta_gradient(policies, grad_rounds, alpha)
        return surrogate, policies[-1]

    @staticmethod
    def _meta_gradient(policies, grad_rounds, alpha):
        """MAML back-substitution: fold each inner step's Hessian into the final
        (extrinsic) gradient, ``g <- g - alpha * H_j g`` for j = J-2 .. 0."""
        grads = grad_rounds[-1]
        for j in reversed(range(len(grad_rounds) - 1)):
            params = list(policies[j].parameters())
            outputs = grad_rounds[j]
            active = [(k, o) for k, o in enumerate(outputs) if o.grad_fn is not None]
            if active:
                idxs, outs = zip(*active)
                Hv = torch.autograd.grad(outs, [params[k] for k in idxs],
                                         grad_outputs=[grads[k] for k in idxs],
                                         allow_unused=True, retain_graph=True)
                hv_map = {k: (h if h is not None else torch.zeros_like(params[k]))
                          for k, h in zip(idxs, Hv)}
            else:
                hv_map = {}
            Hv_full = tuple(hv_map.get(k, torch.zeros_like(p)) for k, p in enumerate(params))
            grads = tuple(g - alpha * h for g, h in zip(grads, Hv_full))
        return grads

    # ── TRPO base-policy update ───────────────────────────────────────────── #
    def _trpo_step(self, grads, obs_p):
        c: IRPO_CFG = self.cfg
        old_actor = self._clean_clone(self.policy)
        grad_flat = torch.cat([g.reshape(-1) for g in grads]).detach()

        def kl_fn():
            return compute_policy_kl(old_actor, self.policy, obs_p)

        Hv = lambda v: hessian_vector_product(kl_fn, self.policy, c.trpo_damping, v)
        step_dir = conjugate_gradients(Hv, grad_flat, nsteps=int(c.trpo_cg_iters))
        sAs = 0.5 * torch.dot(step_dir, Hv(step_dir))
        if sAs < 1e-8:
            return 0, False
        lm = torch.sqrt(sAs / c.base_target_kl)
        full_step = step_dir / lm
        if not torch.isfinite(full_step).all():
            return 0, False

        with torch.no_grad():
            old_params = flat_params(self.policy)
            for k in range(int(c.trpo_backtrack_iters)):
                set_flat_params(self.policy, old_params - (c.trpo_backtrack_coeff ** k) * full_step)
                if compute_policy_kl(old_actor, self.policy, obs_p) <= c.base_target_kl:
                    return k, True
            set_flat_params(self.policy, old_params)
        return int(c.trpo_backtrack_iters), False

    # ── aggregation weights ───────────────────────────────────────────────── #
    def _aggregation_weights(self, progress: float) -> torch.Tensor:
        c: IRPO_CFG = self.cfg
        logits = self.perf_gains
        if c.aggregation_temperature <= 0:
            temperature = 1e-8
        else:
            temperature = max(1e-8, 1.0 - progress / c.aggregation_temperature)
        if self.num_options == 1 or temperature <= 1e-8 or c.aggregation_method == "argmax":
            w = torch.zeros_like(logits)
            w[int(torch.argmax(logits).item())] = 1.0
            return w, temperature
        if c.aggregation_method == "uniform":
            return torch.ones_like(logits) / logits.shape[0], temperature
        return torch.softmax(logits / temperature, dim=0), temperature

    # ── skrl hook ─────────────────────────────────────────────────────────── #
    def update(self, *, timestep: int, timesteps: int) -> None:
        try:
            observations = self.memory.get_tensor_by_name("observations")
            actions = self.memory.get_tensor_by_name("actions")
            rewards = self.memory.get_tensor_by_name("rewards")
            terminated = self.memory.get_tensor_by_name("terminated")
            truncated = self.memory.get_tensor_by_name("truncated")
        except Exception:  # noqa: BLE001 — fall back to plain PPO
            super().update(timestep=timestep, timesteps=timesteps)
            return

        T, N = observations.shape[0], observations.shape[1]
        obs_dim = observations.shape[-1]
        dones = (terminated.bool() | truncated.bool()).squeeze(-1)

        # 1. Representation. Frozen after offline pretrain; only trained online if
        #    pretraining was disabled (extractor_pretrain_epochs = 0).
        if not self._extractor_frozen:
            for k, v in self.extractor.learn(observations, dones).items():
                self.track_data(k, v)

        # 2. Intrinsic rewards on the RAW rollout (ALLO reads xy directly, unscaled).
        next_observations = torch.cat([observations[1:], observations[-1:]], dim=0)
        r_int = self.intrinsic_rewards(
            observations.reshape(-1, obs_dim), next_observations.reshape(-1, obs_dim),
        ).reshape(T, N, -1)

        # Preprocessed observations for the policy/critic networks.
        obs_p_flat = self._observation_preprocessor(observations.reshape(-1, obs_dim), train=True)
        actions_flat = actions.reshape(-1, actions.shape[-1])
        rew2d = rewards.squeeze(-1)

        # Subsample once for the second-order adaptation + TRPO KL (memory cap).
        B = obs_p_flat.shape[0]
        meta_B = min(B, int(self.cfg.meta_batch_size))
        meta_idx = torch.randperm(B, device=self.device)[:meta_B] if meta_B < B else None
        obs_p_meta = obs_p_flat if meta_idx is None else obs_p_flat[meta_idx]
        actions_meta = actions_flat if meta_idx is None else actions_flat[meta_idx]

        # 3. Adapt every option and collect its surrogate extrinsic gradient.
        surrogates, ext_return_means = [], []
        for i in range(self.num_options):
            ext_adv, ext_ret = self._advantages(self.ext_critics[i], obs_p_flat,
                                                 rew2d, terminated.squeeze(-1),
                                                 truncated.squeeze(-1), (T, N))
            int_adv, int_ret = self._advantages(self.int_critics[i], obs_p_flat,
                                                r_int[..., i], terminated.squeeze(-1),
                                                truncated.squeeze(-1), (T, N))
            ext_adv_n = ((ext_adv - ext_adv.mean()) / (ext_adv.std() + 1e-8)).reshape(-1)
            int_adv_n = ((int_adv - int_adv.mean()) / (int_adv.std() + 1e-8)).reshape(-1)
            if meta_idx is not None:
                ext_adv_n, int_adv_n = ext_adv_n[meta_idx], int_adv_n[meta_idx]

            surrogate, adapted = self._adapt_option(i, obs_p_meta, actions_meta,
                                                    int_adv_n, ext_adv_n)
            surrogates.append(surrogate)
            self.final_exp_policies[i].load_state_dict(adapted.state_dict())
            ext_return_means.append(float(ext_ret.mean().item()))

            # Fit this option's critics to their GAE targets.
            self._fit_critic(self.ext_critics[i], self.ext_critic_optim[i], obs_p_flat, ext_ret)
            self._fit_critic(self.int_critics[i], self.int_critic_optim[i], obs_p_flat, int_ret)

        # 4. EMA performance gains → aggregation weights.
        beta = float(self.cfg.perf_gain_beta)
        with torch.no_grad():
            for i, r in enumerate(ext_return_means):
                self.perf_gains[i] = beta * self.perf_gains[i] + (1 - beta) * r
        progress = timestep / max(1, timesteps)
        weights, temperature = self._aggregation_weights(progress)

        aggregated = tuple(
            sum(weights[i] * surrogates[i][p] for i in range(self.num_options))
            for p in range(len(surrogates[0]))
        )

        # 5. TRPO base-policy update along the aggregated surrogate gradient.
        backtrack_iter, backtrack_success = self._trpo_step(aggregated, obs_p_meta.detach())

        # ── metrics (mirror IRPO_Learner.learn) ───────────────────────────── #
        greedy_idx = int(torch.argmax(self.perf_gains).item())
        self.track_data("IRPO / avg_ext_returns", float(self.perf_gains.mean().item()))
        self.track_data("IRPO / max_ext_returns", float(self.perf_gains.max().item()))
        self.track_data("IRPO / Contributing Option", float(greedy_idx))
        self.track_data("IRPO / Backtrack_iter", float(backtrack_iter))
        self.track_data("IRPO / Backtrack_success", float(backtrack_success))
        self.track_data("IRPO / target_kl", float(self.cfg.base_target_kl))
        self.track_data("IRPO / temperature", float(temperature))
        self.track_data("IRPO / kl_divergence", float(self._exploratory_kl(obs_p_flat)))
        self.track_data("IRPO / Intrinsic reward (abs mean)", float(r_int.abs().mean().item()))

    @torch.no_grad()
    def _exploratory_kl(self, obs_p) -> float:
        """Mean pairwise KL among the adapted exploratory policies (legacy
        ``measure_kl_among_exp_policies``) — how differently the options explore."""
        if self.num_options <= 1:
            return 0.0
        probe = obs_p[: min(obs_p.shape[0], 4096)]
        dists = []
        for policy in self.final_exp_policies:
            _, out = policy.act({"observations": probe, "states": None}, role="policy")
            dists.append((out["mean_actions"], out["log_std"]))
        total, count = 0.0, 0
        for i in range(self.num_options):
            mi, li = dists[i]
            for j in range(self.num_options):
                if i == j:
                    continue
                mj, lj = dists[j]
                vi, vj = torch.exp(2 * li), torch.exp(2 * lj)
                kl = (lj - li + (vi + (mi - mj) ** 2) / (2 * vj) - 0.5).sum(-1).mean()
                total += float(kl.item())
                count += 1
        return total / max(1, count)

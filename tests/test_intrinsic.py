"""Unit checks for the intrinsic reward wrapper and DRND novelty module
(explorationRL/agents/skrl/intrinsic.py).

No Isaac Sim needed: the wrapper only needs a vectorised env exposing
``observation_space`` / ``step`` / ``reset``, which is faked here.

Run:
    python tests/test_intrinsic.py
Exits non-zero if any check fails.
"""

from __future__ import annotations

import gymnasium
import torch

from explorationRL.agents.skrl.intrinsic import (
    DRNDNovelty, IntrinsicRewardWrapper, wrap_intrinsic,
)

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


N, OBS_DIM = 4, 6


class FakeEnv:
    """Vec env whose observations drift, so novelty has something to chase."""

    device = torch.device("cpu")
    num_envs = N
    observation_space = gymnasium.spaces.Box(-1e3, 1e3, (OBS_DIM,))
    action_space = gymnasium.spaces.Box(-1.0, 1.0, (2,))
    state_space = None

    def __init__(self):
        self.t = 0

    def reset(self):
        self.t = 0
        return torch.zeros(N, OBS_DIM), {}

    def step(self, actions):
        self.t += 1
        obs = torch.full((N, OBS_DIM), float(self.t) * 0.1) + torch.randn(N, OBS_DIM) * 0.01
        done = torch.zeros(N, 1, dtype=torch.bool)
        done[:] = self.t % 7 == 0
        return obs, torch.zeros(N, 1), done, torch.zeros(N, 1, dtype=torch.bool), {}

    def close(self):
        pass


# ── 1. int_reward=None is a true passthrough (no wrapper at all) ──────────── #
env = FakeEnv()
check("int_reward=None returns the env untouched", wrap_intrinsic(env, None) is env)
check("int_reward='none' returns the env untouched", wrap_intrinsic(env, "none") is env)

# ── 2. DRND: per-step bonus in infos, predictor actually trains ───────────── #
torch.manual_seed(0)
wrapped = wrap_intrinsic(FakeEnv(), "drnd", train_every=8)
check("drnd wrapping produces a wrapper", isinstance(wrapped, IntrinsicRewardWrapper))
obs, infos = wrapped.reset()
check("reset exposes int_reward", infos["int_reward"].shape == (N, 1))
# The wrapper must be transparent to the runner: skrl's base Wrapper resolves
# spaces against env.unwrapped, which for Isaac Lab is the *batched* space.
check("spaces/num_envs come from the wrapped env, not unwrapped",
      wrapped.observation_space is FakeEnv.observation_space
      and wrapped.action_space is FakeEnv.action_space
      and wrapped.num_envs == N)

rewards = []
# skrl's trainer runs the whole interaction loop under no_grad, and the wrapper
# trains its model from inside step() — so exercise it in that context.
with torch.no_grad():
    for _ in range(32):
        obs, r, terminated, truncated, infos = wrapped.step(torch.zeros(N, 2))
        check_shape = infos["int_reward"].shape == (N, 1)
        if not check_shape:
            break
        rewards.append(infos["int_reward"].mean().item())
check("drnd int_reward is (num_envs, 1) every step", check_shape)
check("drnd int_reward is finite and non-negative",
      all(r == r and r >= 0.0 for r in rewards), f"got {rewards[:3]}")
check("env reward is left untouched", float(r.abs().max()) == 0.0)

# reward_scale=0 (default) must leave reward untouched; nonzero must add it in
# -- this is how a PLAIN agent (e.g. ppo) actually consumes an intrinsic bonus.
torch.manual_seed(0)
scaled = wrap_intrinsic(FakeEnv(), "drnd", train_every=8, reward_scale=2.0)
scaled.reset()
with torch.no_grad():
    _, r_scaled, _, _, infos_scaled = scaled.step(torch.zeros(N, 2))
expected_r = 2.0 * infos_scaled["int_reward"]
check("reward_scale adds reward_scale*int_reward onto the env reward",
      torch.allclose(r_scaled, expected_r), f"got {r_scaled.flatten().tolist()}")

# The predictor is distilled on a fixed batch: the loss must fall.
torch.manual_seed(0)
novelty = DRNDNovelty(OBS_DIM, predictor_hidden=[32], target_hidden=[32], embedding_dim=8)
batch = torch.randn(64, OBS_DIM)
unseen_batch = torch.randn(64, OBS_DIM) * 3.0
first = novelty.loss(batch).item()
for _ in range(500):
    novelty.learn(batch)
last = novelty.loss(batch).item()
check("drnd predictor distillation reduces the loss", last < first, f"{first:.4f} -> {last:.4f}")

# Novelty is lower for states the predictor was distilled on than for fresh ones.
# update_stats=False: both are scored on the same normalizer, so the comparison
# is about the predictor, not about which batch happened to be seen first.
seen = novelty(batch, update_stats=False).mean().item()
unseen = novelty(unseen_batch, update_stats=False).mean().item()
check("drnd novelty is lower on distilled states", seen < unseen, f"{seen:.3f} vs {unseen:.3f}")

# ── 3. ALLO: the requested option index is the column returned ────────────── #
torch.manual_seed(0)
wrapped = wrap_intrinsic(FakeEnv(), "allo", option=2, num_options=4,
                         train_every=8, buffer_steps=16, allo_epochs=2)
obs, _ = wrapped.reset()
prev = obs
with torch.no_grad():
    for _ in range(24):
        obs, _, _, _, infos = wrapped.step(torch.zeros(N, 2))
        r_int = infos["int_reward"]
check("allo int_reward is (num_envs, 1)", r_int.shape == (N, 1))
check("allo int_reward is finite", bool(torch.isfinite(r_int).all()))
check("allo extractor is trained online", wrapped.extractor.nupdates > 0,
      f"nupdates={wrapped.extractor.nupdates}")

# option must select the matching column of the full per-option reward
# `_intrinsic` is the training-time path (it advances the normalizer), so score
# the full per-option reward *after* it, on the same normalizer state.
selected = wrapped._intrinsic(prev, obs)
full = wrapped.model(prev, obs, update_stats=False)
check("allo returns the requested option column", torch.allclose(selected, full[:, 2:3]))

try:
    wrap_intrinsic(FakeEnv(), "allo", option=9, num_options=4)
    check("out-of-range option is rejected", False)
except ValueError:
    check("out-of-range option is rejected", True)

# ── 4. "best": geodesic guidance ──────────────────────────────────────────── #
from explorationRL.agents.skrl.intrinsic import GeodesicGuidance  # noqa: E402

MAZE = [
    [1, 1, 1, 1, 1, 1],
    [1, "r", 1, "g", 0, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1],
]
g = GeodesicGuidance(MAZE, 1.0, bins_per_cell=16)


def phi(x, y):
    return float(g.potential(torch.tensor([[x, y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])))


start, goal = (-1.5, 1.0), (0.5, 1.0)
check("goal cell has zero potential", phi(*goal) == 0.0, f"{phi(*goal)}")
# straight line start->goal is 2 m; the route around the corridor is ~4.9 m
check("potential is geodesic, not euclidean", phi(*start) > 4.0, f"d(start)={phi(*start):.2f}")
# moving *down* out of the start cell (the only way out) must make progress,
# moving toward the goal through the wall must not
down = g(torch.tensor([[*start, 0.0, 0, 0, 0, 0, 0]]),
         torch.tensor([[-1.5, 0.0, 0.0, 0, 0, 0, 0, 0]]), update_stats=False)
thru = g(torch.tensor([[*start, 0.0, 0, 0, 0, 0, 0]]),
         torch.tensor([[-0.6, 1.0, 0.0, 0, 0, 0, 0, 0]]), update_stats=False)
check("progress down the corridor is positive", float(down) > 0, f"{float(down):.3f}")
check("progress straight at the wall is not positive", float(thru) <= 0, f"{float(thru):.3f}")

wrapped = wrap_intrinsic(FakeEnv(), "best", best_kwargs={"maze_map": MAZE, "cell_size": 1.0})
with torch.no_grad():
    wrapped.reset()
    for _ in range(4):
        _, _, _, _, infos = wrapped.step(torch.zeros(N, 2))
check("best int_reward is (num_envs, 1) and finite",
      infos["int_reward"].shape == (N, 1) and bool(torch.isfinite(infos["int_reward"]).all()))

try:
    wrap_intrinsic(FakeEnv(), "bogus")
    check("unknown int_reward is rejected", False)
except ValueError:
    check("unknown int_reward is rejected", True)

# ── 5. potential-based shaping: F = γ·Φ(s') - Φ(s) (not_done-gated) ───────── #
# (Ng, Harada & Russell 1999) — the policy-invariance guarantee needs this
# exact γ coefficient, not the plain Φ(s')-Φ(s) used before.
from explorationRL.agents.skrl.intrinsic import ALLOIntrinsicRewards  # noqa: E402
from explorationRL.extractors import ALLO, ALLO_CFG  # noqa: E402

obs5 = torch.tensor([[-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [0.5, 0.5]])
next_obs5 = torch.tensor([[0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [-0.5, -0.5]])
not_done5 = torch.tensor([1.0, 1.0, 0.0, 0.0])

g_disc = GeodesicGuidance(MAZE, 1.0, bins_per_cell=16, discount=0.9)


def full_obs(xy: torch.Tensor) -> torch.Tensor:
    return torch.cat([xy, torch.zeros(xy.shape[0], 6)], dim=-1)


out5 = g_disc.forward(full_obs(obs5), full_obs(next_obs5), not_done5, update_stats=False)
expected5_raw = (g_disc.potential(full_obs(obs5))
                 - 0.9 * not_done5 * g_disc.potential(full_obs(next_obs5))).unsqueeze(-1)
expected5 = g_disc.reward_rms.normalize_var_only(expected5_raw)
check("GeodesicGuidance: F = γ·Φ(s') - Φ(s)",
      torch.allclose(out5, expected5), f"got {out5.flatten().tolist()}")

zeroed5 = (g_disc.potential(full_obs(obs5[2:]))
          - 0.9 * 0.0 * g_disc.potential(full_obs(next_obs5[2:]))).unsqueeze(-1)
check("GeodesicGuidance: not_done=0 drops the next-state potential",
      torch.allclose(expected5_raw[2:], zeroed5))

# defaults (discount=1, not_done=None) must reproduce the old undiscounted formula
out5_default = g.forward(full_obs(obs5), full_obs(next_obs5), update_stats=False)
expected5_default_raw = (g.potential(full_obs(obs5)) - g.potential(full_obs(next_obs5))).unsqueeze(-1)
expected5_default = g.reward_rms.normalize_var_only(expected5_default_raw)
check("GeodesicGuidance: defaults reproduce the old undiscounted formula",
      torch.allclose(out5_default, expected5_default))

torch.manual_seed(0)
extractor5 = ALLO(observation_dim=2, cfg=ALLO_CFG(feature_dim=3, positional_indices=None), device="cpu")
allo_model = ALLOIntrinsicRewards(extractor5, num_options=2, discount=0.8, device="cpu")
out6 = allo_model.forward(obs5, next_obs5, not_done5, update_stats=False)
nd = not_done5.reshape(-1, 1)
delta_expected = 0.8 * nd * extractor5(next_obs5) - extractor5(obs5)
rewards_expected_raw = delta_expected[:, allo_model._indices] * allo_model._signs
expected6 = allo_model.reward_rms.normalize_var_only(rewards_expected_raw)
check("ALLOIntrinsicRewards: F = γ·Φ(s') - Φ(s)",
      torch.allclose(out6, expected6), f"got {out6.tolist()}")

allo_default = ALLOIntrinsicRewards(extractor5, num_options=2, device="cpu")  # discount defaults to 1.0
out6_default = allo_default.forward(obs5, next_obs5, update_stats=False)  # no not_done passed
delta_default = extractor5(next_obs5) - extractor5(obs5)
rewards_default_raw = delta_default[:, allo_default._indices] * allo_default._signs
expected6_default = allo_default.reward_rms.normalize_var_only(rewards_default_raw)
check("ALLOIntrinsicRewards: defaults reproduce the old undiscounted formula",
      torch.allclose(out6_default, expected6_default))

# ── 6. ALLOGoalKernelRewards: goal-conditioned kernel over ALLO's directions ─ #
from explorationRL.agents.skrl.intrinsic import (  # noqa: E402
    KERNEL_MODES, ALLOGoalKernelRewards, _apply_kernel,
)

torch.manual_seed(0)
phi_a = torch.randn(5, 4)
phi_b = torch.randn(5, 4)
for mode in KERNEL_MODES:
    k = _apply_kernel(phi_a, phi_b, mode)
    lo = 0.0 if mode == "cosine" else 0.0
    check(f"_apply_kernel[{mode}] is in (0, 1] and shaped (batch, 1)",
          bool(((k > lo) & (k <= 1.0 + 1e-6)).all()) and tuple(k.shape) == (5, 1),
          f"got {k.flatten().tolist()}")
check("_apply_kernel[rbf] peaks at 1 when phi(s) == phi(g)",
      torch.allclose(_apply_kernel(phi_a, phi_a, "rbf"), torch.ones(5, 1)))
try:
    _apply_kernel(phi_a, phi_b, "bogus")
    check("_apply_kernel rejects an unknown mode", False)
except ValueError:
    check("_apply_kernel rejects an unknown mode", True)

# int_reward="allo_kernel" needs a goal-conditioned env: OBS_DIM=6, agent xy at
# [0,1], goal xy at [4,5] (mirrors pointmaze's [x,y,vx,vy,...,goal_x,goal_y]).
wrapped = wrap_intrinsic(FakeEnv(), "allo_kernel", goal_indices=[4, 5],
                         allo_cfg=ALLO_CFG(positional_indices=[0, 1]),
                         train_every=8, buffer_steps=16, allo_epochs=2)
check("allo_kernel wrapping builds an ALLOGoalKernelRewards model",
      isinstance(wrapped.model, ALLOGoalKernelRewards))
obs, _ = wrapped.reset()
with torch.no_grad():
    for _ in range(24):
        obs, _, _, _, infos = wrapped.step(torch.zeros(N, 2))
        r_kernel = infos["int_reward"]
check("allo_kernel int_reward is (num_envs, 1)", r_kernel.shape == (N, 1))
check("allo_kernel int_reward is finite", bool(torch.isfinite(r_kernel).all()))

try:
    wrap_intrinsic(FakeEnv(), "allo_kernel")  # no goal_indices
    check("allo_kernel without goal_indices is rejected", False)
except ValueError:
    check("allo_kernel without goal_indices is rejected", True)

# ALLOGoalKernelRewards potential shaping directly: F = discount*not_done*K(s',g) - K(s,g)
torch.manual_seed(0)
extractor7 = ALLO(observation_dim=6, cfg=ALLO_CFG(feature_dim=4, positional_indices=[0, 1]), device="cpu")
kernel_model = ALLOGoalKernelRewards(extractor7, mode="rbf", discount=0.8, device="cpu")
state7 = torch.randn(4, 6)
next_state7 = torch.randn(4, 6)
goal7 = torch.randn(4, 6)
not_done7 = torch.tensor([1.0, 1.0, 0.0, 0.0])
out7 = kernel_model.forward({"state": state7, "goal": goal7}, {"state": next_state7, "goal": goal7},
                            not_done7, update_stats=False)
phi_g7, phi_s7, phi_sn7 = extractor7(goal7)[:, 1:], extractor7(state7)[:, 1:], extractor7(next_state7)[:, 1:]
k_s7 = _apply_kernel(phi_s7, phi_g7, "rbf")
k_sn7 = _apply_kernel(phi_sn7, phi_g7, "rbf")
expected7_raw = 0.8 * not_done7.reshape(-1, 1) * k_sn7 - k_s7
expected7 = kernel_model.reward_rms.normalize_var_only(expected7_raw)
check("ALLOGoalKernelRewards: F = discount*not_done*K(s',g) - K(s,g), index 0 dropped",
      torch.allclose(out7, expected7), f"got {out7.flatten().tolist()}")

# True (unblended) reward tracking: the tensorboard tag must reflect the raw env
# reward, not env_reward + reward_scale*int_reward -- see plot_learning_curves.py.
import tempfile

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

with tempfile.TemporaryDirectory() as tmpdir:
    torch.manual_seed(0)
    wrapped8 = wrap_intrinsic(FakeEnv(), "drnd", train_every=8, reward_scale=0.5,
                              log_dir=tmpdir, log_every=5)
    wrapped8.reset()
    for _ in range(10):
        _, r8, _, _, _ = wrapped8.step(torch.zeros(N, 2))
    check("reward_scale=0.5 blends a nonzero bonus into the returned reward",
          bool((r8 != 0).any()), f"got {r8.flatten().tolist()}")
    wrapped8.close()

    ea8 = EventAccumulator(tmpdir)
    ea8.Reload()
    true_tag = "Reward / True total reward (mean)"
    check(f"{true_tag!r} scalar is logged", true_tag in ea8.Tags()["scalars"])
    true_vals8 = [e.value for e in ea8.Scalars(true_tag)]
    # FakeEnv's raw reward is always exactly 0 -- the true-reward tag must stay 0
    # even though the returned (blended) reward above was nonzero.
    check("true-reward tag stays 0 (FakeEnv's raw reward), unlike the blended reward",
          all(v == 0.0 for v in true_vals8), f"got {true_vals8}")

print("\n" + ("ALL PASSED" if not fails else f"FAILED: {fails}"))
raise SystemExit(1 if fails else 0)

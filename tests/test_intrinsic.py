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

print("\n" + ("ALL PASSED" if not fails else f"FAILED: {fails}"))
raise SystemExit(1 if fails else 0)

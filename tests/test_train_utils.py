"""Unit check for write_seed_result's reward_scale walk (scripts/skrl/train_utils.py).

Regression test for a bug where --record_visitation wraps IntrinsicRewardWrapper
in VisitationRecorder: `skrl_env.reward_scale = 0.0` on the outer wrapper just
created a shadow attribute there instead of reaching the real one, silently
leaving the intrinsic-reward blend on during "deterministic" eval and inflating
return/success numbers. Fixed by walking the `._env` chain to the actual owner.

Run:
    python tests/test_train_utils.py
Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "skrl"))
from train_utils import apply_wandb_sweep_overrides  # noqa: E402

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


class Inner:
    def __init__(self):
        self.reward_scale = 0.1


class Outer:
    """Mimics VisitationRecorder wrapping an IntrinsicRewardWrapper: no
    reward_scale of its own, just a ._env pointer (as skrl's Wrapper uses)."""

    def __init__(self, env):
        self._env = env


def find_scale_owner(skrl_env):
    owner = skrl_env
    while owner is not None and "reward_scale" not in vars(owner):
        owner = getattr(owner, "_env", None)
    return owner


inner = Inner()
outer = Outer(inner)

owner = find_scale_owner(outer)
check("walk finds the real owner through the wrapper chain", owner is inner)

prev = owner.reward_scale
owner.reward_scale = 0.0
check("zeroing via the found owner reaches the inner wrapper", inner.reward_scale == 0.0)
owner.reward_scale = prev
check("restoring via the found owner reaches the inner wrapper", inner.reward_scale == 0.1)

check("walk on the owner itself (no wrapping) finds itself", find_scale_owner(inner) is inner)
check("walk on an env with no reward_scale anywhere returns None", find_scale_owner(Outer(Outer(None))) is None)

# apply_wandb_sweep_overrides: dotted keys -> agent_cfg, "seed" -> agent_cfg,
# a bare key matching an args_cli attribute -> setattr(args_cli, ...) (needed
# for flags like --int_reward_coef that aren't part of the agent config at
# all, see search/configs/ppo_allogoal.yaml), anything else is ignored.
import wandb

wandb.init(mode="disabled", config={
    "agent.discount_factor": 0.99,
    "seed": 7,
    "int_reward_coef": 0.3,
    "not_a_real_arg": 123,
})
agent_cfg = {"agent": {"discount_factor": 0.95}, "seed": 42}
args_cli = argparse.Namespace(int_reward_coef=0.1, algorithm="ppo_allogoal")
apply_wandb_sweep_overrides(agent_cfg, args_cli)
check("dotted key overrides agent_cfg", agent_cfg["agent"]["discount_factor"] == 0.99)
check("'seed' overrides agent_cfg", agent_cfg["seed"] == 7)
check("bare key matching an args_cli attr overrides it", args_cli.int_reward_coef == 0.3)
check("bare key with no matching args_cli attr is ignored, no crash", args_cli.algorithm == "ppo_allogoal")
wandb.finish()

print("\n" + ("ALL PASSED" if not fails else f"FAILED: {fails}"))
raise SystemExit(1 if fails else 0)

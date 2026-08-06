"""Unit checks for the AGA agent's maze geometry, return-to-go, and the
asymmetric PD preconditioner (explorationRL/agents/skrl/aga.py).

No Isaac Sim needed: the only reason ``aga.py`` imports from
``pointmaze_env_cfg`` is two plain constants (``CELL_SIZE``,
``POINTMAZE_V1_MAP``), so that module is stubbed out here to dodge the
``isaaclab`` import chain.

Run:
    python tests/test_aga.py
Exits non-zero if any check fails.
"""

from __future__ import annotations

import sys
import types

import torch

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


# ── stub the isaaclab-dependent module so `aga.py` imports cleanly ────────── #
POINTMAZE_V1_MAP = [
    [1, 1, 1, 1, 1, 1],
    [1, "r", 1, "g", 0, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1],
]
stub = types.ModuleType("explorationRL.tasks.direct.pointmaze.pointmaze_env_cfg")
stub.POINTMAZE_V1_MAP = POINTMAZE_V1_MAP
stub.CELL_SIZE = 1.0
sys.modules["explorationRL.tasks.direct.pointmaze.pointmaze_env_cfg"] = stub

from explorationRL.agents.skrl.aga import _MazeCells  # noqa: E402
from explorationRL.agents.skrl.common import return_to_go  # noqa: E402

device = torch.device("cpu")

# ── 1. bottleneck detection ────────────────────────────────────────────────── #
# Hand-checked against POINTMAZE_V1_MAP: the vertical passages out of the 'r'
# room ((2,1) below 'r', (2,4) below 'g') and the two interior cells of the
# 1-wide horizontal corridor row ((3,2), (3,3)) are the only doorway cells.
maze = _MazeCells(POINTMAZE_V1_MAP, cell_size=1.0, device=device)
expected_doorways = {(2, 1), (2, 4), (3, 2), (3, 3)}
doorway_cells = {
    cell for cell, idx in zip(
        [(i, j) for i in range(5) for j in range(6) if POINTMAZE_V1_MAP[i][j] != 1],
        range(maze.n_cells),
    )
    if maze.bottleneck_reward[idx].item() == 1.0
}
check("bottleneck cells match hand-derived set", doorway_cells == expected_doorways,
      f"got {doorway_cells}")
check("bottleneck reward is 0/1 valued", set(maze.bottleneck_reward.tolist()) <= {0.0, 1.0})

# ── 2. return-to-go: discounted, resets at episode boundaries ─────────────── #
gamma = 0.5
b_t = torch.tensor([[1.0], [0.0], [1.0], [0.0]])  # (T=4, N=1)
dones = torch.tensor([[False], [True], [False], [False]])
G = return_to_go(b_t, dones, gamma)
# episode 1 = steps 0..1: G[1]=0, G[0]=1 + 0.5*0 = 1
# episode 2 = steps 2..3: G[3]=0, G[2]=1 + 0.5*0 = 1
expected_G = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
check("return-to-go resets at dones", torch.allclose(G, expected_G), f"got {G.flatten().tolist()}")

b_t2 = torch.tensor([[0.0], [0.0], [1.0]])
dones2 = torch.tensor([[False], [False], [False]])
G2 = return_to_go(b_t2, dones2, gamma)
# G2[2]=1, G2[1]=0+0.5*1=0.5, G2[0]=0+0.5*0.5=0.25 — credits *approaching* a
# target cell, not just occupying one (spec property #1).
expected_G2 = torch.tensor([[0.25], [0.5], [1.0]])
check("return-to-go credits approach, not just occupancy",
      torch.allclose(G2, expected_G2), f"got {G2.flatten().tolist()}")

# ── 3. asymmetric PD preconditioner: fixed points + directional bias ──────── #
d = 5
v_hat = torch.randn(d)
v_hat = v_hat / v_hat.norm()
beta, damp = 2.0, 0.9


def precond(delta: torch.Tensor) -> torch.Tensor:
    proj = torch.dot(v_hat, delta)
    scale = beta if proj.item() > 0 else -damp
    return delta + scale * proj * v_hat


check("delta_ppo=0 is an exact fixed point",
      torch.allclose(precond(torch.zeros(d)), torch.zeros(d)))

delta = torch.randn(d)
delta_pos = delta + max(0.0, -torch.dot(v_hat, delta).item() + 1.0) * v_hat  # force proj>0
g_pos = precond(delta_pos)
check("proj>0 branch amplifies along v_hat (PD, eigenvalue 1+beta)",
      torch.dot(v_hat, g_pos).item() > torch.dot(v_hat, delta_pos).item())

delta_neg = delta - max(0.0, torch.dot(v_hat, delta).item() + 1.0) * v_hat  # force proj<0
g_neg = precond(delta_neg)
proj_neg = torch.dot(v_hat, delta_neg).item()
check("proj<0 branch damps but keeps sign (PD, eigenvalue 1-damp in (0,1])",
      0.0 < torch.dot(v_hat, g_neg).item() / proj_neg <= 1.0)

# ── 4. smoothed branch (precond_tau>0) stays inside the PD bounds and is
#      continuous through proj=0, unlike the hard sign() switch ──────────── #
def precond_smooth(delta: torch.Tensor, tau: float) -> torch.Tensor:
    proj = torch.dot(v_hat, delta)
    t = torch.tanh(proj / tau)
    scale = 0.5 * (beta + damp) * t + 0.5 * (beta - damp)
    return delta + scale * proj * v_hat

check("smoothed branch: delta_ppo=0 is still an exact fixed point",
      torch.allclose(precond_smooth(torch.zeros(d), tau=1.0), torch.zeros(d)))

eps = torch.tensor(1e-6)
g_lo = precond_smooth(-eps * v_hat, tau=1.0)
g_hi = precond_smooth(eps * v_hat, tau=1.0)
check("smoothed branch has no jump across proj=0 (hard switch would)",
      torch.allclose(g_lo, -g_hi, atol=1e-5), f"got {g_lo.tolist()} vs {-g_hi}")

print(f"\n{len(fails)} failure(s)." if fails else "\nAll checks passed.")
sys.exit(1 if fails else 0)

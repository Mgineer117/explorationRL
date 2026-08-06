"""ExplorationRunner — skrl Runner extended with this package's custom agents.

The PPO / TRPO / SAC baselines are resolved entirely by stock skrl. This subclass
adds two things so the ported research algorithms (DRND, PSNE, HTRPO, IRPO, HRL,
MAML, AGA, PGNoCritic) can be selected from a yaml exactly like a built-in:

1. ``_COMPONENT_REGISTRY`` — a ``name -> factory`` table consulted before skrl's
   own ``_component`` lookup. Each algorithm registers its agent class under
   ``<name>`` and its config dataclass under ``<name>_cfg``.

2. ``_generate_agent`` — skrl's version dispatches on a *hardcoded* list of agent
   class names (``["a2c", ..., "ppo", "sac", "trpo"]``), so a custom class would
   fall through it with no ``agent_cfg``/``agent_kwargs`` bound. The override
   below reproduces skrl's single-agent construction path for registered custom
   agents and delegates everything else to ``super()``.

3. ``_generate_models`` — a top-level yaml ``zero_init_critic: true`` zeroes the
   value model's output head right after construction (works for ANY agent,
   including stock PPO, and both ``models.separate`` true/false), so ``V(s)=0``
   at init instead of arbitrary/noisy — see ``_zero_init_value_head``.

Note the one deliberate difference from skrl's path: the config dataclass is
passed as an *instance* rather than ``dataclasses.asdict(...)``. skrl's agents do
``PPO_CFG(**cfg) if isinstance(cfg, dict) else cfg``, so passing the instance is
what lets a *subclass* config (with extra fields such as DRND's ``alpha``)
survive the base agent's constructor instead of raising on unexpected keys.
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn
from skrl.utils.runner.torch import Runner


def _zero_init_value_head(model: nn.Module) -> None:
    """Zero the final linear layer that outputs the scalar value estimate, so
    ``V(s) = 0`` for every state at init instead of an arbitrary/noisy value --
    an early GAE advantage then reduces to the plain (undiscounted-by-critic)
    reward until the critic has actually learned something, rather than being
    driven by an untrained critic's noise (see ``pg_no_critic.py``'s docstring
    for the motivating hypothesis: this targets the same noise source without
    removing the critic entirely).

    Handles both a standalone value model (zeroes its last ``nn.Linear``) and
    skrl's shared actor-critic model (``models.separate: false``, which exposes
    a dedicated ``value_layer`` head after ``init_state_dict`` resolves its lazy
    shape -- zeroing it leaves the policy head and shared trunk untouched)."""
    head = getattr(model, "value_layer", None)
    if head is None:
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        head = linears[-1] if linears else None
    if head is not None:
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)


def _drnd_agent():
    from explorationRL.agents.skrl.drnd import DRND

    return DRND


def _drnd_cfg():
    from explorationRL.agents.skrl.drnd import DRND_CFG

    return DRND_CFG


def _psne_agent():
    from explorationRL.agents.skrl.psne import PSNE

    return PSNE


def _psne_cfg():
    from explorationRL.agents.skrl.psne import PSNE_CFG

    return PSNE_CFG


def _irpo_agent():
    from explorationRL.agents.skrl.irpo import IRPO

    return IRPO


def _irpo_cfg():
    from explorationRL.agents.skrl.irpo import IRPO_CFG

    return IRPO_CFG


def _aga_agent():
    from explorationRL.agents.skrl.aga import AGA

    return AGA


def _aga_cfg():
    from explorationRL.agents.skrl.aga import AGA_CFG

    return AGA_CFG


def _pgnc_agent():
    from explorationRL.agents.skrl.pg_no_critic import PGNoCritic

    return PGNoCritic


def _pgnc_cfg():
    from explorationRL.agents.skrl.pg_no_critic import PGNC_CFG

    return PGNC_CFG


def _hrl_agent():
    from explorationRL.agents.skrl.hrl import HRL

    return HRL


def _hrl_cfg():
    from explorationRL.agents.skrl.hrl import HRL_CFG

    return HRL_CFG


def _maml_agent():
    from explorationRL.agents.skrl.maml import MAML

    return MAML


def _maml_cfg():
    from explorationRL.agents.skrl.maml import MAML_CFG

    return MAML_CFG


def _htrpo_agent():
    from explorationRL.agents.skrl.htrpo import HTRPO

    return HTRPO


def _htrpo_cfg():
    from explorationRL.agents.skrl.htrpo import HTRPO_CFG

    return HTRPO_CFG


class ExplorationRunner(Runner):
    # Lowercased name -> zero-arg factory returning the component. Factories keep
    # the (potentially heavy) agent imports lazy until actually requested.
    _COMPONENT_REGISTRY: dict[str, Callable[[], object]] = {
        "drnd": _drnd_agent,
        "drnd_cfg": _drnd_cfg,
        "psne": _psne_agent,
        "psne_cfg": _psne_cfg,
        "htrpo": _htrpo_agent,
        "htrpo_cfg": _htrpo_cfg,
        "irpo": _irpo_agent,
        "irpo_cfg": _irpo_cfg,
        "aga": _aga_agent,
        "aga_cfg": _aga_cfg,
        "pgnocritic": _pgnc_agent,
        "pgnocritic_cfg": _pgnc_cfg,
        "hrl": _hrl_agent,
        "hrl_cfg": _hrl_cfg,
        "maml": _maml_agent,
        "maml_cfg": _maml_cfg,
    }

    @classmethod
    def register_component(cls, name: str, factory: Callable[[], object]) -> None:
        """Register a custom component under ``name`` (case-insensitive).

        An algorithm needs two entries: ``<name>`` (the agent class) and
        ``<name>_cfg`` (its config dataclass, a subclass of the base agent's).
        """
        cls._COMPONENT_REGISTRY[name.lower()] = factory

    def _component(self, name: str):
        factory = self._COMPONENT_REGISTRY.get(name.lower())
        if factory is not None:
            return factory()
        return super()._component(name)

    def _generate_models(self, env, cfg: dict):
        models = super()._generate_models(env, cfg)
        if bool(cfg.get("zero_init_critic", False)):
            for roles in models.values():
                value_model = roles.get("value")
                if value_model is not None:
                    _zero_init_value_head(value_model)
        return models

    def _generate_agent(self, env, cfg: dict, models: dict):
        agent_class = str(cfg.get("agent", {}).get("class", "")).lower()
        if agent_class not in self._COMPONENT_REGISTRY:
            return super()._generate_agent(env, cfg, models)

        device = env.device
        num_envs = env.num_envs
        observation_space = env.observation_space
        state_space = env.state_space
        action_space = env.action_space

        # memory (same contract as skrl: memory_size < 0 means "one rollout")
        if "memory" not in cfg or "class" not in cfg["memory"]:
            raise ValueError("The 'memory.class' field is not defined in the configuration")
        memory_class = self._component(cfg["memory"]["class"])
        if cfg["memory"]["memory_size"] < 0:
            cfg["memory"]["memory_size"] = cfg["agent"]["rollouts"]
        memory = memory_class(num_envs=num_envs, device=device, **self._process_cfg(cfg["memory"]))

        # config dataclass INSTANCE (see the module docstring for why)
        cfg_class = self._component(f"{agent_class}_cfg")
        agent_cfg = cfg_class(**self._process_cfg(cfg["agent"]))
        agent_cfg.observation_preprocessor_kwargs.update({"size": observation_space, "device": device})
        agent_cfg.state_preprocessor_kwargs.update({"size": state_space, "device": device})
        agent_cfg.value_preprocessor_kwargs.update({"size": 1, "device": device})

        return self._component(agent_class)(
            models=models["agent"],
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            cfg=agent_cfg,
            device=device,
        )

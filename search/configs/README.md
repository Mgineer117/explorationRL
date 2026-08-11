# `search/configs/` — one search space per algorithm

Each `<algorithm>.yaml` here declares the hyperparameters and ranges swept for
that algorithm, and applies to **every** env. `search.sh` prompts for the
algorithm + env, and `build_sweep.py` merges the chosen config with the env to
emit a W&B sweep yaml.

## File schema

```yaml
label:      ppo                          # shown in search.sh's preview
algorithm:  ppo                          # scripts/skrl/train.py --algorithm value
num_envs:   4096                         # parallel envs per trial

metric:            # verbatim W&B sweep `metric:` block
  name: "Reward / Total reward (mean)"   # a scalar skrl logs (must match exactly)
  goal: maximize | minimize

parameters:        # verbatim W&B sweep `parameters:` block; keys are DOTTED paths
  agent.discount_factor:                 #   into the agent yaml (agent.*, models.*)
    values: [0.95, 0.98, 0.99, 0.995]

# ── optional ──────────────────────────────────────────────────────────────
method: bayes      # bayes (default) | grid | random. grid needs every parameter
                   # discrete (a `values:` list) — build_sweep.py rejects a grid
                   # config that still declares a continuous `distribution:`.

extra_args:        # fixed (unswept) CLI flags spliced into every trial's command,
  - "--agent"      # right after --num_envs and before the sweep's own ${args}.
  - "skrl_ppo_cfg_entry_point"   # for flags train.py needs that aren't agent.*/
  - "--int_reward"               # models.* config keys (e.g. --agent, --int_reward,
  - "drnd"                       # --int_reward_coef) and aren't themselves searched.
```

The dotted parameter keys (`agent.discount_factor`, …) are written straight onto
`wandb.config`; `scripts/skrl/train.py` applies them to the loaded agent config
before any model is built (see `apply_wandb_sweep_overrides`). The metric name
must be a scalar skrl actually logs — `"Reward / Total reward (mean)"` is the
episodic-reward scalar the trainer emits.

## Which algorithms exist

| config | `--algorithm` | metric | notes |
|---|---|---|---|
| `ppo.yaml`   | `ppo`   | reward | on-policy baseline, many envs |
| `sac.yaml`   | `sac`   | reward | off-policy baseline, few envs + replay |
| `trpo.yaml`  | `trpo`  | reward | on-policy trust region |
| `drnd.yaml`  | `drnd`  | reward | PPO + Distributional RND novelty bonus |
| `psne.yaml`  | `psne`  | reward | TRPO + adaptive parameter-space noise |
| `htrpo.yaml` | `htrpo` | reward | TRPO + hindsight relabelling (HGF) + WIS |
| `irpo.yaml`  | `irpo`  | reward | ALLO option policies (flagship) |
| `hrl.yaml`   | `hrl`   | reward | controller over options, semi-MDP |
| `maml.yaml`  | `maml`  | reward | FOMAML support/query meta-update |

`irpo` and `hrl` partition the parallel envs across their option policies, so
their `num_envs` must exceed `num_options`. `htrpo` multiplies the update batch
by `num_hindsight_goals + 1`, which is why its `num_envs` is set lower.

## Adding an algorithm

Drop a new `<name>.yaml` following the schema. `search.sh` discovers configs by
globbing this directory — nothing else needs editing.

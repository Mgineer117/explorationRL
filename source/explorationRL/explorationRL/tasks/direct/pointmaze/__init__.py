"""PointMaze direct RL task registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="PointMaze-v1",
    entry_point=f"{__name__}.pointmaze_env:PointMazeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pointmaze_env_cfg:PointMazeEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        "skrl_trpo_cfg_entry_point": f"{agents.__name__}:skrl_trpo_cfg.yaml",
        "skrl_drnd_cfg_entry_point": f"{agents.__name__}:skrl_drnd_cfg.yaml",
        "skrl_psne_cfg_entry_point": f"{agents.__name__}:skrl_psne_cfg.yaml",
        "skrl_htrpo_cfg_entry_point": f"{agents.__name__}:skrl_htrpo_cfg.yaml",
        "skrl_irpo_cfg_entry_point": f"{agents.__name__}:skrl_irpo_cfg.yaml",
        "skrl_hrl_cfg_entry_point": f"{agents.__name__}:skrl_hrl_cfg.yaml",
        "skrl_maml_cfg_entry_point": f"{agents.__name__}:skrl_maml_cfg.yaml",
    },
)

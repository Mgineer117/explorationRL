Changelog
---------

0.1.0 (2026-07-23)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Initial ``explorationRL`` Isaac Lab extension (skrl backend).
* ``PointMaze-v1`` direct RL environment — a force-actuated point mass whose
  dynamics are matched to the gymnasium-robotics MuJoCo ``PointMaze``.
* skrl PPO / TRPO / SAC baselines wired through ``scripts/skrl/train.py``.
* Research algorithms ported onto skrl agents: ``DRND``, ``PSNE``, ``HTRPO``,
  ``IRPO``, ``HRL`` and ``MAML``.
* ``ALLO`` Laplacian representation learner and the eigenvector-direction
  intrinsic rewards used by ``IRPO``/``HRL``.
* ``ExplorationRunner`` — resolves the custom agents/config dataclasses that
  skrl's hardcoded agent dispatch cannot.

Removed
^^^^^^^

* The legacy non-skrl pipeline (``main.py``, ``algorithms/``, ``policy/``,
  ``trainer/``, ``utils/``, ``extractor/``, ``gridworld/`` and the vendored
  ``gymnasium_robotics/``). Pretrained ALLO checkpoints under ``model/`` are kept.

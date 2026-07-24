"""skrl agents and the ExplorationRunner.

The runner (:class:`~explorationRL.agents.skrl.runner.ExplorationRunner`) is a
thin subclass of skrl's ``Runner`` that resolves this package's custom agent
classes and model factories, falling back to skrl's stock components for the
PPO / TRPO / SAC baselines.
"""

"""explorationRL — Isaac Lab extension (skrl backend).

Importing this package registers every gym environment it defines. The import
is wrapped so the package can also be imported in a *classic* (no Isaac Sim)
context — where the Isaac-only task modules are simply skipped — for the
lightweight/discrete environments that do not need the simulator.
"""

# Register Gym environments (Isaac Sim only; skipped when Isaac is absent).
try:
    from .tasks import *  # noqa: F401,F403
except (ImportError, ModuleNotFoundError):
    pass

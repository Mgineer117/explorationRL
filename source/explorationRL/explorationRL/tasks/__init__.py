"""Task (environment) implementations for the explorationRL extension.

Importing this package registers every gym environment. Under Isaac Lab the
``import_packages`` helper walks the sub-packages; without it (e.g. a classic
context) we fall back to importing the direct-task package directly.
"""

##
# Register Gym environments.
##

try:
    from isaaclab_tasks.utils import import_packages

    # Prevent importing helper/config-only sub-packages as if they were tasks.
    _BLACKLIST_PKGS = ["utils", ".mdp"]
    import_packages(__name__, _BLACKLIST_PKGS)
except (ImportError, ModuleNotFoundError):
    # No isaaclab_tasks available — still register whatever direct tasks import
    # cleanly on their own.
    try:
        from . import direct  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        pass

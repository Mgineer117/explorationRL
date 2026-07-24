"""Representation learners used by the option-based agents (IRPO, HRL, MAML).

``ALLO`` learns a Laplacian representation of the state space; the eigenvector
directions of that representation define the intrinsic rewards each exploratory
option policy is trained on.
"""

from explorationRL.extractors.allo import ALLO, ALLO_CFG  # noqa: F401
from explorationRL.extractors.networks import FeatureMLP  # noqa: F401

__all__ = ["ALLO", "ALLO_CFG", "FeatureMLP"]

"""ga_torch - PyTorch-native Island Genetic Algorithm framework.

Usage:
    from ga_torch import IslandGA
    from ga_torch import operators, population, simulation
"""

from .island import IslandGA
from . import operators
from . import population
from . import simulation
from . import reproduction

__all__ = ['IslandGA', 'operators', 'population', 'simulation', 'reproduction']

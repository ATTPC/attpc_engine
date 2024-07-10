from typing import Protocol
from numpy.random import Generator
import numpy as np


class PolarDistribution(Protocol):
    """Protocol for defining a reaction polar angle distribution in the CM frame

    Note that for polar angle distributions, the coordinate to sample
    is cos(polar) not polar (to cover solid angle uniformly). This means
    that the valid domain of these distributions is [0.0, pi]

    Methods
    -------
    sample(rng)
        Sample the distribution. Returns angle in radians
    """

    def sample(self, rng: Generator) -> float:  # type: ignore
        """Sample the distribution, returning a polar angle in radians

        Parameters
        ----------
        rng: numpy.random.Generator
            The random number generator

        Returns
        -------
        float:
            The sampled angle in radians
        """
        pass


class PolarUniform:
    """A uniform polar angle distribution in the CM frame

    Note that for polar angle distributions, the coordinate to sample
    is cos(polar) not polar. This means that the valid domain of these
    distributions is [0.0, pi]

    Parameters
    ----------
    angle_min: float
        The minimum polar angle in radians
    angle_max: float
        The maximum polar angle in radians

    Attributes
    ----------
    cos_angle_min: float
        The minimum cos(polar)
    cos_angle_max: float
        The maximum cos(polar)

    Methods
    -------
    sample(rng)
        Sample the distribution, returning a polar angle in radians
    """

    def __init__(self, angle_min: float, angle_max: float):
        # cos(polar) flips the min/max order
        self.cos_angle_min = np.cos(angle_max)
        self.cos_angle_max = np.cos(angle_min)

    def sample(self, rng: Generator) -> float:
        """Sample the distribution, returning a polar angle in radians

        Parameters
        ----------
        rng: numpy.random.Generator
            The random number generator

        Returns
        -------
        float:
            The sampled angle in radians
        """
        return np.arccos(rng.uniform(self.cos_angle_min, self.cos_angle_max))

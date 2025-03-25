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
        float
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
        float
            The sampled angle in radians
        """
        return np.arccos(rng.uniform(self.cos_angle_min, self.cos_angle_max))


class PolarArbitrary:
    """An arbitrary, finite-precision polar angular distribution in the CM frame

    Define an arbitrary probability distribution for the polar angle using an
    array of uniformly-spaced angles and their probabilities, as well as the spacing
    of the angle values. We then sample from the distribution and smear the output angle
    within the spacing.

    Note: If your distribution is defined using *any* of the pre-defined distributions,
    you should favor using those over PolarArbitrary. Sampling of arbitrary functions
    comes with a runtime performance penalty. That is: do not use this to sample from a
    uniform distribution.

    Parameters
    ----------
    angles: numpy.ndarray
        The array of *lower* angle values. Each angle corresponds to a bin covering
        angle + bin_width. Angles should be in units of radians.
    probabilities: numpy.ndarray
        The probability of a given angle bin. Should sum to 1.0
    angle_bin_width: float
        The width of the angle bins in radians

    Attributes
    ----------
    angle_width: float
        The angle bin width in radians
    probs: numpy.ndarray
        The probability of a given angle bin
    angles: numpy.ndarray
        The array of *lower* angle values. Each angle corresponds to a bin covering
        angle + bin_width. Angles should be in units of radians.

    Methods
    -------
    sample(rng)
        Sample the distribution, returning a polar angle in radians
    """

    def __init__(
        self,
        angles: np.ndarray,
        probabilities: np.ndarray,
        angle_bin_width: float,
    ):
        if np.sum(probabilities) > 1.0:
            raise ValueError(
                f"The sum of the probabilities passed to PolarArbitrary should be 1.0. Yours sum to {np.sum(probabilities)}"
            )
        self.angle_width = angle_bin_width
        self.probs = probabilities
        self.angles = angles

    def sample(self, rng: Generator) -> float:
        """Sample the distribution, returning a polar angle in radians

        Parameters
        ----------
        rng: numpy.random.Generator
            The random number generator

        Returns
        -------
        float
            The sampled angle in radians
        """
        return (
            rng.choice(self.angles, p=self.probs)
            + rng.uniform(0.0, 1.0) * self.angle_width
        )

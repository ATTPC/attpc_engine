from typing import Protocol
from numpy.random import Generator


class ExciationDistribution(Protocol):
    """Definition of an exciation energy distribution

    Methods
    -------
    sample(rng)
        Sample the distribution
    """

    def sample(self, rng: Generator) -> float:
        """Sample the distribution

        Parameters
        ----------
        rng: numpy.random.Generator
            The random number generator

        Returns
        -------
        float
            The sampled value
        """
        ...


class ExcitationGaussian:
    """Represents the sampling parameters for an excitation as a gaussian peak

    Parameters
    ----------
    centroid: float
        The state mean/centroid value in MeV
    width: float
        The state FWHM value in MeV

    Attributes
    ----------
    centroid: float
        The state mean/centroid value in MeV
    width: float
        The state FWHM value in MeV
    sigma: float
        The state std. deviation in MeV

    Methods
    -------
    sample(rng)
        Sample the distribution
    """

    def __init__(self, centroid: float = 0.0, width: float = 0.0):

        self.centroid = centroid
        self.width = width  # FWHM
        self.sigma = self.width / 2.355

    def sample(self, rng: Generator) -> float:
        """Sample the distribution

        Parameters
        ----------
        rng: numpy.random.Generator
            The random number generator

        Returns
        -------
        float
            The sampled value
        """
        return rng.normal(self.centroid, self.sigma)


class ExcitationUniform:
    """Represents the sampling parameters for an excitation as a flat distribution

    Parameters
    ----------
    min_value: float
        The minimum value of the range
    max_value: float
        The maximum value of the range

    Attributes
    ----------
    min_value: float
        The minimum value of the range
    max_value: float
        The maximum value of the range

    Methods
    -------
    sample(rng)
        Sample the distribution
    """

    def __init__(self, min_value: float = 0.0, max_value: float = 0.0):
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, rng: Generator) -> float:
        return rng.uniform(self.min_value, self.max_value)

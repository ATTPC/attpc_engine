from typing import Protocol
from numpy.random import Generator
from scipy.stats import rel_breitwigner


class ExcitationDistribution(Protocol):
    """Definition of a nuclear excitation with a specified energy
    distribution

    Methods
    -------
    sample(rng)
        Sample the distribution
    """

    def sample(self, rng: Generator) -> float:  # type: ignore
        """Sample the energy distribution of the excitation.

        Parameters
        ----------
        rng: numpy.random.Generator
            The random number generator

        Returns
        -------
        float
            The sampled value
        """
        pass


class ExcitationGaussian:
    """Represents the sampling parameters for a nuclear excitation with the energy
    sampled as a gaussian distribution.

    Parameters
    ----------
    centroid: float
        The state mean/centroid value in MeV.
    width: float
        The state FWHM value in MeV.

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
        Sample the energy distribution of the excitation.
    """

    def __init__(
        self,
        centroid: float = 0.0,
        width: float = 0.0,
    ):
        self.centroid = centroid
        self.width = width  # FWHM
        self.sigma = self.width / 2.355

    def sample(self, rng: Generator) -> float:
        """Sample the energy distribution of the excitation.

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
    """Represents the sampling parameters for a nuclear excitation with the energy
    sampled as a flat distribution.

    Parameters
    ----------
    min_value: float
        The minimum value of the range.
    max_value: float
        The maximum value of the range.

    Attributes
    ----------
    min_value: float
        The minimum value of the range
    max_value: float
        The maximum value of the range

    Methods
    -------
    sample(rng)
        Sample the energy distribution of the excitation.
    """

    def __init__(
        self,
        min_value: float = 0.0,
        max_value: float = 0.0,
    ):
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, rng: Generator) -> float:
        """Sample the energy distribution of the excitation.

        Parameters
        ----------
        rng: numpy.random.Generator
            The random number generator

        Returns
        -------
        float
            The sampled value
        """
        return rng.uniform(self.min_value, self.max_value)


class ExcitationBreitWigner:
    """Represents the sampling parameters for a nuclear excitation with the energy
    sampled from the relativistic Breit-Wigner distribution.

    Beware that this excitation class is slower than the others because it uses
    scipy for its relativistic Breit Wigner distribution.

    Parameters
    ----------
    rest_mass: float
        The rest mass of the nucleus that is excited in MeV.
    centroid: float
        The state mean/centroid value in MeV.
    width: float
        Width of the state in MeV.

    Attributes
    ----------
    rest_mass: float
        The rest mass of the nucleus that is excited in MeV.
    centroid: float
        The state mean/centroid value in MeV.
    width: float
        Width of the state in MeV.

    Methods
    -------
    sample(rng)
        Sample the energy distribution of the excitation.
    """

    def __init__(
        self,
        rest_mass: float,
        centroid: float,
        width: float,
    ):
        self.rest_mass = rest_mass
        self.centroid = centroid
        self.width = width

    def sample(self, rng: Generator) -> float:
        """Sample the energy distribution of the excitation.

        Parameters
        ----------
        rng: numpy.random.Generator
            The random number generator

        Returns
        -------
        float
            The sampled value
        """
        rho = (self.rest_mass + self.centroid) / self.width
        total_energy = rel_breitwigner.rvs(rho, scale=self.width, random_state=rng)
        excitation_energy = total_energy - self.rest_mass
        return excitation_energy

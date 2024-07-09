from typing import Protocol
from numpy.random import Generator
from numpy import pi
from scipy.stats import rel_breitwigner


class ExcitationDistribution(Protocol):
    """Definition of a nuclear excitation with a specified energy and angle
    distribution.

    Methods
    -------
    sample(rng)
        Sample the distribution
    """

    def sample_energy(self, rng: Generator) -> float:
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

    def sample_theta(self, rng: Generator) -> float:
        """Sample the theta angle distribution of the excitation.

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
    ang_min: float
        The minimum allowed cm angle for emission in radians.
    ang_max: float
        The maximum allowed cm angle for emission in radians.

    Attributes
    ----------
    centroid: float
        The state mean/centroid value in MeV
    width: float
        The state FWHM value in MeV
    sigma: float
        The state std. deviation in MeV
    ang_min: float
        The minimum allowed cm angle for emission in radians.
    ang_max: float
        The maximum allowed cm angle for emission in radians.

    Methods
    -------
    sample_energy(rng)
        Sample the energy distribution of the excitation.
    sample_angle(rng)
        Sample the theta angle distribution of the excitation.
    """

    def __init__(
        self,
        centroid: float = 0.0,
        width: float = 0.0,
        ang_min: float = 0.0,
        ang_max: float = pi,
    ):

        self.centroid = centroid
        self.width = width  # FWHM
        self.sigma = self.width / 2.355
        self.ang_min = ang_min
        self.ang_max = ang_max

    def sample_energy(self, rng: Generator) -> float:
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

    def sample_theta(self, rng: Generator) -> float:
        """Sample the theta angle distribution of the excitation.

        Parameters
        ----------
        rng: numpy.random.Generator
            The random number generator

        Returns
        -------
        float
            The sampled value
        """
        return rng.uniform(self.ang_min, self.ang_max)


class ExcitationUniform:
    """Represents the sampling parameters for a nuclear excitation with the energy
    sampled as a flat distribution.

    Parameters
    ----------
    min_value: float
        The minimum value of the range.
    max_value: float
        The maximum value of the range.
    ang_min: float
        The minimum allowed cm angle for emission in radians.
    ang_max: float
        The maximum allowed cm angle for emission in radians.

    Attributes
    ----------
    min_value: float
        The minimum value of the range
    max_value: float
        The maximum value of the range
    ang_min: float
        The minimum allowed cm angle for emission in radians.
    ang_max: float
        The maximum allowed cm angle for emission in radians.

    Methods
    -------
    sample_energy(rng)
        Sample the energy distribution of the excitation.
    sample_angle(rng)
        Sample the theta angle distribution of the excitation.
    """

    def __init__(
        self,
        min_value: float = 0.0,
        max_value: float = 0.0,
        ang_min: float = 0.0,
        ang_max: float = pi,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.ang_min = ang_min
        self.ang_max = ang_max

    def sample_energy(self, rng: Generator) -> float:
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

    def sample_theta(self, rng: Generator) -> float:
        """Sample the theta angle distribution of the excitation.

        Parameters
        ----------
        rng: numpy.random.Generator
            The random number generator

        Returns
        -------
        float
            The sampled value
        """
        return rng.uniform(self.ang_min, self.ang_max)


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
    ang_min: float
        The minimum allowed cm angle for emission in radians.
    ang_max: float
        The maximum allowed cm angle for emission in radians.

    Attributes
    ----------
    rest_mass: float
        The rest mass of the nucleus that is excited in MeV.
    centroid: float
        The state mean/centroid value in MeV.
    width: float
        Width of the state in MeV.
    ang_min: float
        The minimum allowed cm angle for emission in radians.
    ang_max: float
        The maximum allowed cm angle for emission in radians.

    Methods
    -------
    sample_energy(rng)
        Sample the energy distribution of the excitation.
    sample_angle(rng)
        Sample the theta angle distribution of the excitation.
    """

    def __init__(
        self,
        rest_mass: float,
        centroid: float,
        width: float,
        ang_min: float = 0.0,
        ang_max: float = pi,
    ):
        self.rest_mass = rest_mass
        self.centroid = centroid
        self.width = width
        self.ang_min = ang_min
        self.ang_max = ang_max

    def sample_energy(self, rng: Generator) -> float:
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

    def sample_theta(self, rng: Generator) -> float:
        """Sample the theta angle distribution of the excitation.

        Parameters
        ----------
        rng: numpy.random.Generator
            The random number generator

        Returns
        -------
        float
            The sampled value
        """
        return rng.uniform(self.ang_min, self.ang_max)

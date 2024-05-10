"""
Constants defined for ATTPC_Engine

Some of these are just short aliases to scipy constants to avoid long keystrings

Attributes
----------
NUM_TB: int
    For the GET system, 512
MEV_2_JOULE: float
    scipy alias, J/MeV 
MEV_2_KG: float
    scipy alias, kg/MeV/c^2
C: float
    scipy alias, speed_of_light m/s
E_CHARGE: float
    scipy alias, elementary_charge in Coulombs
"""

from scipy.constants import physical_constants
from scipy.constants import speed_of_light, elementary_charge

NUM_TB: int = 512

MEV_2_JOULE: float = (
    physical_constants["electron volt-joule relationship"][0] * 1.0e6
)  # J/ev * ev/MeV = J/MeV

MEV_2_KG: float = (
    physical_constants["electron volt-kilogram relationship"][0] * 1.0e6
)  # kg/ev * ev/MeV = kg/MeV (per c^2)

C = speed_of_light  # m/s

E_CHARGE = elementary_charge  # Coulombs

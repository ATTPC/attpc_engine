import math
import numpy as np

from spyral_utils.nuclear.target import GasTarget
from spyral_utils.nuclear import NucleusData

from ..constants import MEV_2_JOULE, MEV_2_KG, C, E_CHARGE

KE_LIMIT = 1e-6  # 1 eV


# State = [x, y, z, px, py, pz]
# Derivative = [vx, vy, vz, ax, ay, az] (returns)
def equation_of_motion(
    t: float,
    state: np.ndarray,
    Bfield: float,
    Efield: float,
    target: GasTarget,
    ejectile: NucleusData,
) -> np.ndarray:
    """The equations of motion for a charged particle in a static electromagnetic field which experiences energy loss through some material.

    Field directions are chosen based on standard AT-TPC configuration

    Parameters
    ----------
    t: float
        time step
    state: ndarray
        the state of the particle (x,y,z,vx,vy,vz)
    Bfield: float
        the magnitude of the magnetic field
    Efield: float
        the magnitude of the electric field
    target: Target
        the material through which the particle travels
    ejectile: NucleusData
        ejectile (from the reaction) nucleus data

    Returns
    -------
    ndarray
        the derivatives of the state
    """

    momentum = math.sqrt(state[3] ** 2.0 + state[4] ** 2.0 + state[5] ** 2.0)
    mass_kg = ejectile.mass * MEV_2_KG
    p_m = momentum / mass_kg
    speed = p_m / math.sqrt((1.0 + (p_m * 1.0 / C) ** 2.0))

    unit_vector = state[3:] / momentum  # direction
    velo = unit_vector * speed
    kinetic_energy = ejectile.mass * (
        1.0 / math.sqrt(1.0 - (speed / C) ** 2.0) - 1.0
    )  # MeV

    charge_c = ejectile.Z * E_CHARGE

    deceleration = (
        target.get_dedx(ejectile, kinetic_energy) * MEV_2_JOULE * target.density * 100.0
    )
    results = np.zeros(6)
    results[0] = velo[0]
    results[1] = velo[1]
    results[2] = velo[2]
    results[3] = charge_c * velo[1] * Bfield - deceleration * unit_vector[0]
    results[4] = charge_c * (-1.0 * velo[0] * Bfield) - deceleration * unit_vector[1]
    results[5] = charge_c * Efield - deceleration * unit_vector[2]

    return results


# These function sigs must match the ODE function
def stop_condition(
    t: float,
    state: np.ndarray,
    Bfield: float,
    Efield: float,
    target: GasTarget,
    ejectile: NucleusData,
) -> float:
    """Detect if a solution has reached the low-energy stopping condition

    Terminate the integration of the ivp if the condition specified has been reached.
    Scipy finds the roots of this function with the ivp. The parameters must match
    those given to the equation to be solved

    Parameters
    ----------
    t: float
        time step, unused
    state: ndarray
        the state of the particle (x,y,z,vx,vy,vz)
    Bfield: float
        the magnitude of the magnetic field, unused
    Efield: float
        the magnitude of the electric field, unused
    target: Target
        the material through which the particle travels, unused
    ejectile: NucleusData
        ejectile (from the reaction) nucleus data

    Returns
    -------
    float
        The difference between the kinetic energy and the lower limit. When
        this function returns zero the termination condition has been reached.

    """
    momentum = math.sqrt(state[3] ** 2.0 + state[4] ** 2.0 + state[5] ** 2.0)
    mass_kg = ejectile.mass * MEV_2_KG
    p_m = momentum / mass_kg
    speed = p_m / math.sqrt((1.0 + (p_m * 1.0 / C) ** 2.0))
    kinetic_energy = ejectile.mass * (
        1.0 / math.sqrt(1.0 - (speed / C) ** 2.0) - 1.0
    )  # MeV
    return kinetic_energy - KE_LIMIT


def forward_z_bound_condition(
    t: float,
    state: np.ndarray,
    Bfield: float,
    Efield: float,
    target: GasTarget,
    ejectile: NucleusData,
) -> float:
    """Detect if a solution has reached the end-of-detector-in-z condition

    Terminate the integration of the ivp if the condition specified has been reached.
    Scipy finds the roots of this function with the ivp. The parameters must match
    those given to the equation to be solved

    Parameters
    ----------
    t: float
        time step, unused
    state: ndarray
        the state of the particle (x,y,z,vx,vy,vz)
    Bfield: float
        the magnitude of the magnetic field, unused
    Efield: float
        the magnitude of the electric field, unused
    target: Target
        the material through which the particle travels, unused
    ejectile: NucleusData
        ejectile (from the reaction) nucleus data, unused

    Returns
    -------
    float
        The difference between the current z position and the length of the detector (1m). When
        this function returns zero the termination condition has been reached.

    """
    return state[2] - 1.0


# These function sigs must match the ODE function
def backward_z_bound_condition(
    t: float,
    state: np.ndarray,
    Bfield: float,
    Efield: float,
    target: GasTarget,
    ejectile: NucleusData,
) -> float:
    """Detect if a solution has reached the end-of-detector-in-z condition

    Terminate the integration of the ivp if the condition specified has been reached.
    Scipy finds the roots of this function with the ivp. The parameters must match
    those given to the equation to be solved

    Parameters
    ----------
    t: float
        time step, unused
    state: ndarray
        the state of the particle (x,y,z,vx,vy,vz)
    Bfield: float
        the magnitude of the magnetic field, unused
    Efield: float
        the magnitude of the electric field, unused
    target: Target
        the material through which the particle travels, unused
    ejectile: NucleusData
        ejectile (from the reaction) nucleus data, unused

    Returns
    -------
    float
        The difference between the current z position and the beginning of the detector (0.0 m). When
        this function returns zero the termination condition has been reached.

    """
    return state[2]


# These function sigs must match the ODE function
def rho_bound_condition(
    t: float,
    state: np.ndarray,
    Bfield: float,
    Efield: float,
    target: GasTarget,
    ejectile: NucleusData,
) -> float:
    """Detect if a solution has reached the end-of-detector-in-rho condition

    Terminate the integration of the ivp if the condition specified has been reached.
    Scipy finds the roots of this function with the ivp. The parameters must match
    those given to the equation to be solved

    Note here that the edge in rho (292 mm) is padded by 30 mm to account for conditions where the vertex is off axis

    Parameters
    ----------
    t: float
        time step, unused
    state: ndarray
        the state of the particle (x,y,z,vx,vy,vz)
    Bfield: float
        the magnitude of the magnetic field, unused
    Efield: float
        the magnitude of the electric field, unused
    target: Target
        the material through which the particle travels, unused
    ejectile: NucleusData
        ejectile (from the reaction) nucleus data, unused

    Returns
    -------
    float
        The difference between the current z position and the maximum rho of the detector (332 mm). When
        this function returns zero the termination condition has been reached.

    """
    return float(np.linalg.norm(state[:2])) - 0.332

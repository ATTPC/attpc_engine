import math
import numpy as np
from numpy.random import Generator
from scipy.integrate import solve_ivp

from spyral_utils.nuclear.target import GasTarget
from spyral_utils.nuclear import NucleusData

from .parameters import Config, DetectorParams
from .typed_dict import NumbaTypedDict
from .transporter import transport_track
from .constants import MEV_2_JOULE, MEV_2_KG, C, E_CHARGE

KE_LIMIT = 1e-6  # 1 eV
# Time steps to solve ODE at. Each step is 1e-10 s
TIME_STEPS = np.linspace(0, 10e-7, 10001)


def equation_of_motion(
    t: float,
    state: np.ndarray,
    bfield: float,
    efield: float,
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
        the state of the particle (x,y,z,gvx,gvy,gvz)
    bfield: float
        the magnitude of the magnetic field
    efield: float
        the magnitude of the electric field
    target: spyral_utils.nuclear.GasTarget
        the material through which the particle travels
    ejectile: spyral_utils.nuclear.NucleusData
        ejectile (from the reaction) nucleus data

    Returns
    -------
    ndarray
        the derivatives of the state
    """

    gv = math.sqrt(state[3] ** 2.0 + state[4] ** 2.0 + state[5] ** 2.0)
    beta = math.sqrt(gv**2.0 / (1.0 + gv**2.0))
    gamma = gv / beta

    unit_vector = state[3:] / gv  # direction
    velo = unit_vector * beta * C  # convert to m/s
    kinetic_energy = ejectile.mass * (gamma - 1.0)  # MeV

    charge_c = ejectile.Z * E_CHARGE
    mass_kg = ejectile.mass * MEV_2_KG
    q_m = charge_c / mass_kg

    deceleration = (
        target.get_dedx(ejectile, kinetic_energy) * MEV_2_JOULE * target.density * 100.0
    ) / mass_kg

    results = np.zeros(6)
    results[0] = velo[0]
    results[1] = velo[1]
    results[2] = velo[2]
    results[3] = (q_m * velo[1] * bfield - deceleration * unit_vector[0]) / C
    results[4] = (q_m * (-1.0 * velo[0] * bfield) - deceleration * unit_vector[1]) / C
    results[5] = (q_m * efield - deceleration * unit_vector[2]) / C

    return results


# These function sigs must match the ODE function
def stop_condition(
    t: float,
    state: np.ndarray,
    bfield: float,
    efield: float,
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
    bfield: float
        the magnitude of the magnetic field, unused
    efield: float
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
    gv = math.sqrt(state[3] ** 2.0 + state[4] ** 2.0 + state[5] ** 2.0)
    beta = math.sqrt(gv**2.0 / (1.0 + gv**2.0))
    gamma = gv / beta
    kinetic_energy = ejectile.mass * (gamma - 1.0)  # MeV

    return kinetic_energy - KE_LIMIT


def forward_z_bound_condition(
    t: float,
    state: np.ndarray,
    bfield: float,
    efield: float,
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
    bfield: float
        the magnitude of the magnetic field, unused
    efield: float
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
    bfield: float,
    efield: float,
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
    bfield: float
        the magnitude of the magnetic field, unused
    efield: float
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
    bfield: float,
    efield: float,
    target: GasTarget,
    ejectile: NucleusData,
) -> float:
    """Detect if a solution has reached the end-of-detector-in-rho condition

    Terminate the integration of the ivp if the condition specified has been reached.
    Scipy finds the roots of this function with the ivp. The parameters must match
    those given to the equation to be solved.

    Parameters
    ----------
    t: float
        time step, unused
    state: ndarray
        the state of the particle (x,y,z,vx,vy,vz)
    bfield: float
        the magnitude of the magnetic field, unused
    efield: float
        the magnitude of the electric field, unused
    target: spyral_utils.nuclear.GasTarget
        the material through which the particle travels, unused
    ejectile: NucleusData
        ejectile (from the reaction) nucleus data, unused

    Returns
    -------
    float
        The difference between the current z position and the maximum rho of the detector (332 mm). When
        this function returns zero the termination condition has been reached.

    """
    return float(np.linalg.norm(state[:2])) - 0.292


def generate_trajectory(
    vertex: np.ndarray,
    momentum: np.ndarray,
    nucleus: NucleusData,
    params: DetectorParams,
) -> np.ndarray:
    """Solves EoM for a nucleus in the AT-TPC

    Solution is evaluated over a fixed time interval.

    Parameters
    ----------
    vertex: numpy.ndarray
        The reaction vertex
    momentum: numpy.ndarray
        The inital 4-momentum of the nucleus
    nucleus: spyral_utils.nuclear.NucleusData
        The particle species charge, mass, etc.
    params: DetectorParams
        Parameters defining detector shape, fields, material, etc.

    Returns
    -------
    numpy.ndarray
        Nx6 array where each row is a solution to one of the ODEs evaluated at
        the Nth time step.
    """
    # Find initial state of nucleus. (x, y, z, px, py, pz)
    initial_state: np.ndarray = np.zeros(6)
    initial_state[:3] = vertex  # m
    initial_state[3:] = momentum[:3] / nucleus.mass  # unitless (gamma * beta)

    # Set ODE stop conditions. See SciPy solve_ivp docs
    stop_condition.terminal = True
    stop_condition.direction = -1.0
    forward_z_bound_condition.terminal = True
    forward_z_bound_condition.direction = 1.0
    backward_z_bound_condition.terminal = True
    backward_z_bound_condition.direction = -1.0
    rho_bound_condition.terminal = True
    rho_bound_condition.direction = 1.0

    track = solve_ivp(
        equation_of_motion,
        (0.0, 1.0),  # Set upper time limit to 1 sec. This should never be reached
        initial_state,
        method="Radau",
        events=[
            stop_condition,
            forward_z_bound_condition,
            backward_z_bound_condition,
            rho_bound_condition,
        ],
        t_eval=TIME_STEPS,
        args=(
            params.bfield * -1.0,
            params.efield * -1.0,
            params.gas_target,
            nucleus,
        ),
    )

    return track.y.T  # Return transpose to easily index by row


def generate_electrons(
    track: np.ndarray, nucleus: NucleusData, params: DetectorParams, rng: Generator
):
    """Find the number of electrons made at each point of the nucleus' track.

    Parameters
    ----------
    track: numpy.ndarray
        Nx6 array where each row is a solution to one of the ODEs evaluated at
        the Nth time step.
    nucleus: spyral_utils.nuclear.NucleusData
        The particle species charge, mass, etc.
    params: DetectorParams
        Parameters defining detector geometry, fields, material, etc.
    rng: numpy.random.Generator
        numpy random number generator

    Returns
    -------
    numpy.ndarray
        Returns an array of the number of electrons made at each
        point in the trajectory
    """
    # Find energy of nucleus at each point of its track
    gv = np.linalg.norm(track[:, 3:], axis=1)
    beta = np.sqrt(gv**2.0 / (1.0 + gv**2.0))
    gamma = gv / beta
    energy = nucleus.mass * (gamma - 1.0)  # MeV

    # Find number of electrons created at each point of its track
    electrons = np.zeros_like(energy)
    electrons[1:] = abs(np.diff(energy))  # Magnitude of energy lost
    electrons *= 1.0e6 / params.w_value  # Convert to eV to match units of W-value

    # Adjust number of electrons by Fano factor, can only have integer amount of electrons
    electrons = np.array(
        [rng.normal(point, np.sqrt(params.fano_factor * point)) for point in electrons],
        dtype=np.int64,
    )
    return electrons


def generate_point_cloud(
    momentum: np.ndarray,
    vertex: np.ndarray,
    nucleus: NucleusData,
    config: Config,
    rng: Generator,
    points: NumbaTypedDict[int, tuple[int, int]],
    label: int,
) -> None:
    """Create a point cloud from a given particle

    Generates the trajectory of a particle, propagates electron transport,
    and evaluates which pad on the AT-TPC detector plane the signal is recorded.

    Parameters
    ----------
    momentum: numpy.ndarray
        The particle 4-momentum
    vertex: numpy.ndarray
        The reaction vertex
    nucleus: spyral_utils.nuclear.NucleusData
        The particle species charge, mass, etc.
    config: Config
        The simulation configuration
    rng: numpy.random.Generator
        numpy random number generator
    points: numba.typed.Dict[int, tuple[int,int]]
        A dictionary mapping a unique pad,tb key to the number of electrons and a label.
        The points created by this nucleus are stored here.
    label: int
        The label for points generated by this nucleus.
    """
    # Generate nucleus' track and calculate the electrons made at each point
    track = generate_trajectory(vertex, momentum, nucleus, config.det_params)
    electrons = generate_electrons(track, nucleus, config.det_params, rng)

    # Remove points in trajectory that create less than 1 electron
    mask = electrons >= 1
    track = track[mask]
    electrons = electrons[mask]

    # Apply gain factor from micropattern gas detectors
    electrons *= config.det_params.mpgd_gain

    # Convert z position of trajectory to exact time buckets
    dv = config.drift_velocity
    track[:, 2] = (
        config.det_params.length - track[:, 2]
    ) / dv + config.elec_params.micromegas_edge

    if config.pad_grid_edges is None or config.pad_grid is None:
        raise ValueError("Pad grid is not loaded at generate_point_cloud!")

    transport_track(
        config.pad_grid,
        config.pad_grid_edges,
        config.det_params.diffusion,
        config.det_params.efield,
        config.drift_velocity,
        track,
        electrons,
        points,
        label,
    )

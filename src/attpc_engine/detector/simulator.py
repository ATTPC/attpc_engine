from .parameters import Config
from .solver import (
    equation_of_motion,
    stop_condition,
    forward_z_bound_condition,
    backward_z_bound_condition,
    rho_bound_condition,
)
from .transporter import transport_track
from .writer import SimulationWriter
from .constants import NUM_TB
from .. import nuclear_map
from .pairing import unpair

import numpy as np
import h5py as h5
from tqdm import trange
from scipy.integrate import solve_ivp
from numpy.random import default_rng, Generator
from pathlib import Path
from numba.typed import Dict
from numba.core import types
from numba import njit


# Time steps to solve ODE at. Each step is 1e-10 s
TIME_STEPS = np.linspace(0, 10e-7, 10001)


class SimEvent:
    """A simulated event from the kinematics pipeline.

    Parameters
    ----------
    kinematics: numpy.ndarray
        Nx4 array of four vectors of all N nuclei in the simulated event. From first
        to last column are px, py, pz, E.
    vertex: numpy.ndarray
        The reaction vertex position in meters. Given as a 3-array formated
        [x,y,z].
    proton_numbers: numpy.ndarray
        Nx1 array of proton numbers of all nuclei from reaction and decays in the pipeline.
    mass_numbers: numpy.ndarray
        Nx1 array of mass numbers of all nuclei from reaction and decasy in the pipeline.

    Attributes
    ----------
    nuclei: list[SimParticle]
        Instance of SimParticle for each nuclei in the exit channel of the pipeline.

    Methods
    -------
    digitize(config)
        Transform the kinematics into point clouds
    """

    def __init__(
        self,
        kinematics: np.ndarray,
        vertex: np.ndarray,
        proton_numbers: np.ndarray,
        mass_numbers: np.ndarray,
    ):
        self.nuclei: list[SimParticle] = []

        # Only simulate the nuclei in the exit channel
        # Pattern is:
        # skip target, projectile, take ejectile
        # then skip every other. Append last nucleus as well
        total_nuclei = len(kinematics) - 1
        for idx in range(2, total_nuclei, 2):
            if proton_numbers[idx] == 0:
                continue
            self.nuclei.append(
                SimParticle(
                    kinematics[idx], vertex, proton_numbers[idx], mass_numbers[idx]
                )
            )
        if proton_numbers[-1] != 0:
            self.nuclei.append(
                SimParticle(
                    kinematics[-1], vertex, proton_numbers[-1], mass_numbers[-1]
                )
            )

    def digitize(self, config: Config, rng: Generator) -> np.ndarray:
        """Transform the kinematics into point clouds

        Digitizes the simulated event, converting the number of
        electrons detected on each pad to a signal in ADC units.
        The hardware ID of each pad is included to make the complete
        AT-TPC trace.

        Parameters
        ----------
        config: Config
            The simulation configuration

        Returns
        -------
        point_array: numpy.ndarray
            An Nx3 array representing the point cloud. Each row is a point, with elements
            [pad id, time bucket, electrons]
        """
        # Sum points from all particles
        points = Dict.empty(key_type=types.int64, value_type=types.int64)
        for nuc in self.nuclei:
            nuc.generate_point_cloud(config, rng, points)

        # Convert to numpy array of [pad, tb, e], now combined over pad/tb combos
        point_array = dict_to_points(points)

        # Wiggle point TBs over interval [0.0, 1.0). This simulates effect of converting
        # the (in principle) int TBs to floats.
        point_array[:, 1] += rng.uniform(low=0.0, high=1.0, size=len(point_array))

        # Remove points outside legal bounds in time. TODO check if this is needed
        point_array = point_array[point_array[:, 1] < NUM_TB]

        return point_array


@njit
def dict_to_points(points: Dict):
    """
    Converts dictionary of N pad,tb keys with corresponding number of electrons
    to Nx3 array where each row is [pad, tb, e], now combined over pad/tb combos.

    Parameters
    ----------
    points: numba.typed.Dict[int, int]
        A dictionary mapping a unique pad,tb key to the number of electrons.

    Returns
    -------
    point_array: numpy.ndarray
        Array of points.
    """
    point_array = np.empty((len(points), 3), dtype=float)
    for idx, point in enumerate(points.items()):
        tb, pad = unpair(point[0])
        point_array[idx, 0] = pad
        point_array[idx, 1] = tb
        point_array[idx, 2] = point[1]

    return point_array


class SimParticle:
    """
    A nucleus in a simulated event

    Parameters
    ----------
    data: numpy.ndarray
        Simulated four vector of nucleus from kinematics.
    vertex: numpy.ndarray
        The reaction vertex position in meters. Given as a 3-array formated
        [x,y,z].
    proton_number: int
        Number of protons in nucleus
    mass_number: int
        Number of nucleons in nucleus

    Attributes
    ----------
    data: numpy.ndarray
        Simulated four vector of nucleus from kinematics.
    nucleus: spyral_utils.nuclear.NucleusData
        Data of the simulated nucleus
    vertex: numpy.ndarray
        The reaction vertex position in meters. Given as a 3-array formated
        [x,y,z].

    Methods
    ----------
    generate_track()
        Solves EoM for this nucleus in the AT-TPC
    generate_electrons()
        Finds the number of electrons produced at each point of the
        simulated track
    generate_point_cloud()
        Makes the AT-TPC point cloud of this nucleus' track
    """

    def __init__(
        self,
        data: np.ndarray,
        vertex: np.ndarray,
        proton_number: int,
        mass_number: int,
    ):
        self.data = data
        self.nucleus = nuclear_map.get_data(proton_number, mass_number)
        self.vertex = vertex

    def generate_track(self, config: Config) -> np.ndarray:
        """Solves EoM for this nucleus in the AT-TPC

        Solution is evaluated over a fixed time interval.

        Parameters
        ----------
        config: Config
            The simulation configuration

        Returns
        -------
        numpy.ndarray
            Nx6 array where each row is a solution to one of the ODEs evaluated at
            the Nth time step.
        """
        # Find initial state of nucleus. (x, y, z, px, py, pz)
        initial_state: np.ndarray = np.zeros(6)
        initial_state[:3] = self.vertex  # m
        initial_state[3:] = self.data[:3] / self.nucleus.mass  # unitless (gamma * beta)

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
                config.det_params.bfield * -1.0,
                config.det_params.efield * -1.0,
                config.det_params.gas_target,
                self.nucleus,
            ),
        )

        return track.y.T  # Return transpose to easily index by row

    def generate_electrons(
        self, config: Config, track: np.ndarray, rng: Generator
    ) -> np.ndarray:
        """Find the number of electrons made at each point of the nucleus' track.

        Parameters
        ----------
        config: Config
            The simulation configuration
        track: numpy.ndarray
            Nx6 array where each row is a solution to one of the ODEs evaluated at
            the Nth time step.
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
        energy = self.nucleus.mass * (gamma - 1.0)  # MeV

        # Find number of electrons created at each point of its track
        electrons = np.zeros_like(energy)
        electrons[1:] = abs(np.diff(energy))  # Magnitude of energy lost
        electrons *= (
            1.0e6 / config.det_params.w_value
        )  # Convert to eV to match units of W-value

        # Adjust number of electrons by Fano factor, can only have integer amount of electrons
        electrons = np.array(
            [
                rng.normal(point, np.sqrt(config.det_params.fano_factor * point))
                for point in electrons
            ],
            dtype=np.int64,
        )
        return electrons

    def generate_point_cloud(
        self, config: Config, rng: Generator, points
    ) -> np.ndarray:
        """Create the point cloud

        Finds the pads hit by the electrons transported from each point
        of the nucleus' trajectory to the pad plane.

        Parameters
        ----------
        config: Config
            The simulation configuration
        rng: numpy.random.Generator
            numpy random number generator
        points: numba.typed.Dict[int, int]
            A dictionary mapping a unique pad,tb key to the number of electrons.

        Returns
        -------
        points: numba.typed.Dict[int, int]
            A dictionary mapping a unique pad,tb key to the number of electrons.
        """
        # Generate nucleus' track and calculate the electrons made at each point
        track = self.generate_track(config)
        electrons = self.generate_electrons(config, track, rng)

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
            raise ValueError("Pad grid is not loaded at SimParticle.generate_hits!")

        points = transport_track(
            config.pad_grid,
            config.pad_grid_edges,
            config.det_params.diffusion,
            config.det_params.efield,
            config.drift_velocity,
            track,
            electrons,
            points,
        )

        return points


def run_simulation(
    config: Config,
    input_path: Path,
    writer: SimulationWriter,
):
    """Run the simulation

    Runs the AT-TPC simulation with the input parameters on the specified
    kinematic data hdf5 file generated by the kinematic pipeline.

    Parameters
     ----------
    config: Config
        The simulation configuration
    input_path: pathlib.Path
        Path to HDF5 file containing kinematics
    writer: SimulationWriter
        An object which implements the SimulationWriter Protocol
    """
    print("------- AT-TPC Simulation Engine -------")
    print(f"Applying detector effects to kinematics from file: {input_path}")
    input = h5.File(input_path, "r")
    input_data_group = input["data"]
    proton_numbers = input_data_group.attrs["proton_numbers"]
    mass_numbers = input_data_group.attrs["mass_numbers"]

    n_events: int = input_data_group.attrs["n_events"]  # type: ignore
    print(f"Found {n_events} kinematics events.")
    print(f"Output will be written to {writer.get_directory_name()}.")

    rng = default_rng()

    print("Go!")
    # ASCII art edited from https://emojicombos.com/f1-car-text-art
    print(
        r"""
            ⠀⢀⣀⣀⣀⠀⠀⠀⠀⢀⣀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
            ⠀⢸⣿⣿⡿⢀⣠⣴⣾⣿⣿⣿⣿⣇⡀⠀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
            ⠀⢸⣿⣿⠟⢋⡙⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣿⡿⠓⡐⠒⢶⣤⣄⡀⠀⠀
            ⠀⠸⠿⠇⢰⣿⣿⡆⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⡷⠈⣿⣿⣉⠁⠀
            ⠀⠀⠀⠀⠀⠈⠉⠀⠈⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠀⠈⠉⠁⠀⠈⠉⠉⠀⠀
        """
    )

    for event_number in trange(n_events):  # type: ignore
        dataset: h5.Dataset = input_data_group[f"event_{event_number}"]  # type: ignore
        sim = SimEvent(
            dataset[:].copy(),  # type: ignore
            np.array(
                [
                    dataset.attrs["vertex_x"],
                    dataset.attrs["vertex_y"],
                    dataset.attrs["vertex_z"],
                ]
            ),
            proton_numbers,  # type: ignore
            mass_numbers,  # type: ignore
        )

        cloud = sim.digitize(config, rng)
        if len(cloud) == 0:
            continue

        writer.write(cloud, config, event_number)
    writer.close()
    print("Done.")
    print("----------------------------------------")

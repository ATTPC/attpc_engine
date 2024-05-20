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

import numpy as np
import h5py as h5
from tqdm import trange
from scipy.integrate import solve_ivp
from random import normalvariate

# Time steps to solve ODE at. Each step is 1e-10 s
TIME_STEPS = np.linspace(0, 10e-7, 10001)


class SimEvent:
    """A simulated event from the kinematics pipeline.

    Parameters
    ----------
    kine: np.ndarray
        Nx4 array of four vectors of all N nuclei in the simulated event. From first
        to last column are px, py, pz, E.
    distance: float
        Linear distance the incoming nucleus travelled before it underwent a reaction.
        The angle between the incoming particle and the detector is 0 degrees.
    proton_numbers: np.ndarray
        Nx1 array of proton numbers of all nuclei from reaction and decays in the pipeline.
    mass_numbers: np.ndarray
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
        distance: float,
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
                    kinematics[idx], distance, proton_numbers[idx], mass_numbers[idx]
                )
            )
        if proton_numbers[-1] != 0:
            self.nuclei.append(
                SimParticle(
                    kinematics[-1], distance, proton_numbers[-1], mass_numbers[-1]
                )
            )

    def digitize(self, config: Config) -> np.ndarray:
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
        np.ndarray
            An Nx3 array representing the point cloud. Each row is a point, with elements
            [pad id, time bucket, electrons]
        """
        # Sum traces from all particles
        points: np.ndarray = np.empty((0, 3))
        for nuc in self.nuclei:
            new_points = nuc.generate_point_cloud(config)
            if len(new_points) != 0:
                points = np.vstack((points, new_points))

        # Remove dead points (no pad or electrons)
        points = points[np.logical_and(points[:, 0] != -1.0, points[:, 1] != -1.0)]

        # Remove points outside legal bounds in time
        points = points[points[:, 1] < NUM_TB]

        return points


class SimParticle:
    """
    A nucleus in a simulated event

    Parameters
    ----------
    data: np.ndarray
        Simulated four vector of nucleus from kinematics.
    distance: float
        The on-beam-axis position of the reaction in the detector
    proton_number: int
        Number of protons in nucleus
    mass_number: int
        Number of nucleons in nucleus

    Attributes
    ----------
    data: np.ndarray
        Simulated four vector of nucleus from kinematics.
    nucleus: spyral_utils.nuclear.NucleusData
        Data of the simulated nucleus
    distance: float
        The on-beam-axis position of the reaction in the detector

    Methods
    ----------
    generate_track()
        Solves EoM for this nucleus in the AT-TPC

    """

    def __init__(
        self,
        data: np.ndarray,
        distance: float,
        proton_number: int,
        mass_number: int,
    ):
        self.data = data
        self.nucleus = nuclear_map.get_data(proton_number, mass_number)
        self.distance = distance

    def generate_track(self, config: Config, distance: float) -> np.ndarray:
        """Solves EoM for this nucleus in the AT-TPC

        Solution is evaluated over a fixed time interval.

        Parameters
        ----------
        config: Config
            The simulation configuration
        distance: float
            The on-beam-axis position of the reaction in the detector

        Returns
        -------
        np.ndarray
            Nx6 array where each row is a solution to one of the ODEs evaluated at
            the Nth time step.
        """
        # Find initial state of nucleus. (x, y, z, px, py, pz)
        initial_state: np.ndarray = np.zeros(6)
        initial_state[2] = distance  # m
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

    def generate_electrons(self, config: Config, track: np.ndarray) -> np.ndarray:
        """Find the number of electrons made at each point of the nucleus' track.

        Parameters
        ----------
        config: Config
            The simulation configuration
        track: np.ndarray
            Nx6 array where each row is a solution to one of the ODEs evaluated at
            the Nth time step.

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
                normalvariate(point, np.sqrt(config.det_params.fano_factor * point))
                for point in electrons
            ],
            dtype=np.int64,
        )
        return electrons

    def generate_point_cloud(self, config: Config) -> np.ndarray:
        """Create the point cloud

        Finds the pads hit by the electrons transported from each point
        of the nucleus' trajectory to the pad plane.

        Parameters
        ----------
        config: Config
            The simulation configuration

        Returns
        -------
        numpy.ndarray
            Array of points (point cloud)
        """
        # Generate nucleus' track and calculate the electrons made at each point
        track = self.generate_track(config, self.distance)
        electrons = self.generate_electrons(config, track)

        # Remove points in trajectory that create less than 1 electron
        mask = electrons >= 1
        track = track[mask]
        electrons = electrons[mask]

        # Apply gain factor from micropattern gas detectors
        electrons *= config.det_params.mpgd_gain

        # Convert z position of trajectory to time buckets
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
        )
        return points


def run_simulation(
    config: Config,
    input_path: str,
    writer: SimulationWriter,
):
    """Run the simulation

    Runs the AT-TPC simulation with the input parameters on the specified
    kinematic data hdf5 file generated by the kinematic pipeline.

    Parameters
     ----------
    config: Config
        The simulation configuration
    input_path: str
        Path to HDF5 file containing kinematics
    writer: SimulationWriter
        An object which implements the SimulationWriter Protocol
    """

    input = h5.File(input_path, "r")
    input_data_group = input["data"]
    proton_numbers = input_data_group.attrs["proton_numbers"]
    mass_numbers = input_data_group.attrs["mass_numbers"]

    n_events: int = input_data_group.attrs["n_events"]  # type: ignore
    writer.set_number_of_events(n_events)

    for event_number in trange(n_events):  # type: ignore
        dataset: h5.Dataset = input_data_group[f"event_{event_number}"]  # type: ignore
        sim = SimEvent(
            dataset[:].copy(),  # type: ignore
            dataset.attrs["distance"],  # type: ignore
            proton_numbers,  # type: ignore
            mass_numbers,  # type: ignore
        )

        cloud = sim.digitize(config)
        if len(cloud) == 0:
            continue

        writer.write(cloud, config, event_number)

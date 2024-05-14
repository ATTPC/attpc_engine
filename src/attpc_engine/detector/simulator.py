import numpy as np
import h5py as h5
import math

from tqdm import tqdm
from .parameters import Config
from .solver import (
    equation_of_motion,
    stop_condition,
    forward_z_bound_condition,
    backward_z_bound_condition,
    rho_bound_condition,
)
from scipy.integrate import solve_ivp
from .transporter import make_traces, transport_track
from random import normalvariate
from ..constants import NUM_TB, E_CHARGE
from .. import nuclear_map
from .writer import SimulationWriter


def get_response(config: Config) -> np.ndarray:
    """
    Theoretical response function of GET electronics provided by the
    chip manufacturer. See https://doi.org/10.1016/j.nima.2016.09.018.

    Parameters
    ----------
    params: Parameters
        All parameters for simulation.

    Returns
    -------
    np.ndarray
        Returns 1xNUM_TB array of the response function evaluated
        at each time bucket
    """
    response = np.zeros(NUM_TB)
    for tb in range(NUM_TB):
        c1 = (
            4095 * E_CHARGE / config.electronics.amp_gain / 1e-15
        )  # Should be 4096 or 4095?
        c2 = tb / (
            config.electronics.shaping_time * config.electronics.clock_freq * 0.001
        )
        response[tb] = c1 * math.exp(-3 * c2) * (c2**3) * math.sin(c2)

    # Cannot have negative ADC values
    response[response < 0] = 0

    return response


class SimEvent:
    """A simulated event from the kinematics pipeline.

    Parameters
    ----------
    params: Parameters
        All parameters for simulation.

    Attributes
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
    nuclei: list[SimParticle]
        Instance of SimParticle for each nuclei in the exit channel of the pipeline.
    data: np.ndarray
        AT-TPC data for the simulated event.
    """

    def __init__(
        self,
        config: Config,
        kine: np.ndarray,
        distance: float,
        proton_numbers: np.ndarray,
        mass_numbers: np.ndarray,
    ):
        self.kine = kine
        self.distance = distance
        self.nuclei: list[SimParticle] = []

        # Only simulate the nuclei in the exit channel
        counter = 0
        while counter < (kine.shape[0] - 1):
            counter += 2
            if counter > (kine.shape[0] - 1):
                counter -= 1
            if (
                proton_numbers[counter] == 0
            ):  # pycatima cannot do energy loss for neutrons!
                continue
            self.nuclei.append(
                SimParticle(
                    config,
                    kine[counter],
                    distance,
                    proton_numbers[counter],
                    mass_numbers[counter],
                )
            )

        self.data = self.digitize(config)

    def digitize(self, config: Config) -> np.ndarray | None:
        """
        Digitizes the simulated event, converting the number of
        electrons detected on each pad to a signal in ADC units.
        The hardware ID of each pad is included to make the complete
        AT-TPC trace.

        Parameters
        ----------
        params: Parameters
            All parameters for simulation.

        Returns
        -------
        np.ndarray | None
            Returns a Nx(NUM_TB+5) array where each row is the
            complete trace of one pad.
        """
        # Sum traces from all particles
        points: np.ndarray = self.nuclei[0].hits  # These are simulation points
        for nuc in self.nuclei:
            points = np.vstack((points, nuc.hits))

        # Remove dead points
        points = points[points[:, 0] != -1.0]

        # Cannot digitize event where no pads are hit
        if len(points) == 0:
            return None

        # # Convolve traces with response function to get digitized traces
        # response: np.ndarray = np.tile(get_response(config), (traces.shape[0], 1))
        # digi_traces = fftconvolve(traces[:, 5:], response, mode="full", axes=(1,))
        # traces[:, 5:] = digi_traces[:, :NUM_TB]

        # # Set saturated pads to max ADC
        # traces[:, 5:][traces[:, 5:] > 4095] = 4095

        return points


class SimParticle:
    """
    A nucleus in a simulated event

    Parameters
    ----------
    params: Parameters
        All parameters for simulation.
    distance: float
        Linear distance the incoming nucleus travelled before it underwent a reaction.
        The angle between the incoming particle and the detector is 0 degrees.
    proton_number: int
        Number of protons in nucleus
    mass_number: int
        Number of nucleons in nucleus

    Attributes
    ----------
    data: np.ndarray
        Simulated four vector of nucleus from pipeline.
    nucleus: spyral_utils.nuclear.NucleusData
        Data of the simulated nucleus
    track: scipy.integrate.solve_ivp bunch object
        Simulated track of nucleus in the gas target
    electrons: np.ndarray
        Array of electrons created at each point of
        the nucleus' track.

    Methods
    ----------
    generate_track -> scipy.integrate.solve_ivp bunch object
        Solves EoM for track of nucleus in gas target

    """

    def __init__(
        self,
        config: Config,
        data: np.ndarray,
        distance: float,
        proton_number: int,
        mass_number: int,
    ):
        self.data = data
        self.nucleus = nuclear_map.get_data(proton_number, mass_number)
        self.distance = distance
        self.hits = self.generate_hits(config)

    def generate_track(self, config: Config, distance: float) -> np.ndarray:
        """
        Solve the EoM of the nucleus in the AT-TPC.

        Parameters
        ----------
        params: Parameters
            All parameters for simulation.
        distance: float
            Linear distance the incoming nucleus travelled before it underwent a reaction.
            The angle between the incoming particle and the detector is 0 degrees.

        Returns
        -------
        np.ndarray
            6xN array where each row is a solution to one of the ODEs evaluated at
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

        # Time steps to solve ODE at. Each step is 1e-10 s
        time_steps = np.linspace(0, 10e-7, 10001)

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
            t_eval=time_steps,
            args=(
                config.detector.bfield * -1.0,
                config.detector.efield * -1.0,
                config.detector.gas_target,
                self.nucleus,
            ),
        )

        return track.y

    def generate_electrons(
        self, config: Config, track: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find the number of electrons made at each point of the nucleus' track.

        Parameters
        ----------
        params: Parameters
            All parameters for simulation.
        track: np.ndarray
            6xN array where each row is a solution to one of the ODEs evaluated at
            the Nth time step.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Element 0 is the 6xN track array and element 1 is a 1xN array
            of electrons created at each point in the track.
        """
        # Find energy of nucleus at each point of its track
        gv = np.linalg.norm(track[3:], axis=0)
        beta = np.sqrt(gv**2.0 / (1.0 + gv**2.0))
        gamma = gv / beta
        energy = self.nucleus.mass * (gamma - 1.0)  # MeV

        # Find number of electrons created at each point of its track
        electrons = np.zeros_like(energy)
        electrons[1:] = abs(np.diff(energy))  # Magnitude of energy lost
        electrons *= (
            1e6 / config.detector.w_value
        )  # Convert to eV to match units of W-value

        # Adjust number of electrons by Fano factor, can only have integer amount of electrons
        electrons = np.array(
            [
                normalvariate(point, np.sqrt(config.detector.fano_factor * point))
                for point in electrons
            ],
            dtype=np.int64,
        )

        # Remove points in trajectory that create less than 1 electron
        mask = electrons >= 1
        track = track[:, mask]
        electrons = electrons[mask]

        return track, electrons

    def z_to_tb(self, config: Config, track) -> np.ndarray:
        """
        Converts the z-coordinate of each point in the track
        from physical space (m) to time (time buckets). Note that
        the time buckets returned are floats and will be rounded
        down to their correct bins when the electrons at each
        point are transported to the pad plane.

        Parameters
        ----------
        params: Parameters
            All parameters for simulation.
        track: np.ndarray
            6xN array where each row is a solution to one of the ODEs evaluated at
            the Nth time step.

        Returns
        -------
        np.ndarray[floats]
            1xN array of time buckets for N points in the track
            that produce one or more electrons.
        """
        # Z coordinates of points in track
        zpos: np.ndarray = track[2]

        dv: float = config.calculate_drift_velocity()

        # Convert from m to time buckets
        zpos = np.abs(zpos - config.detector.length)  # z relative to the micromegas
        zpos /= dv
        zpos += config.electronics.micromegas_edge

        return zpos

    def generate_hits(self, config: Config) -> np.ndarray:
        """
        Finds the pads hit by the electrons transported from each point
        of the nucleus' trajectory to the pad plane.

        Parameters
        ----------
        params: Parameters
            All parameters for simulation.

        Returns
        -------
        numpy.ndarray
            Array of points (point cloud)
        """
        # Generate nucleus' track and calculate the electrons made at each point
        track: np.ndarray = self.generate_track(config, self.distance)
        track, electrons = self.generate_electrons(config, track)

        # Apply gain factor from micropattern gas detectors
        electrons *= config.detector.mpgd_gain

        # Convert z position of trajectory to time buckets
        track[2] = self.z_to_tb(config, track)

        points = transport_track(
            config.pad_map,
            config.pads.map_params,
            config.detector.diffusion,
            config.detector.efield,
            config.calculate_drift_velocity(),
            track,
            electrons,
        )
        return points


def run_simulation(
    config: Config,
    input_path: str,
    writer: SimulationWriter,
):
    """
    Runs the AT-TPC simulation with the input parameters on the specified
    kinematic data hdf5 file generated by the kinematic pipeline.

    Parameters
     ----------
    params: Parameters
        All parameters for simulation.
    input_path: str
        Path to HDF5 file containing kinematics
    """

    input = h5.File(input_path, "r")
    input_data_group = input["data"]

    evt_counter: int = 0
    for event in tqdm(range(input_data_group.attrs["n_events"])):  # type: ignore
        sim = SimEvent(
            config,
            input_data_group[f"event_{event}"],  # type: ignore
            input_data_group[f"event_{event}"].attrs["distance"],  # type: ignore
            input_data_group.attrs["proton_numbers"],  # type: ignore
            input_data_group.attrs["mass_numbers"],  # type: ignore
        )

        if sim.data is None:
            continue

        writer.write(sim.data, config, evt_counter)
        evt_counter += 1
    writer.set_number_of_events(evt_counter)

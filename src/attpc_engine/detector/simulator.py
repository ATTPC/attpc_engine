import numpy as np
import h5py as h5

from .parameters import *
from .solver import *
from pathlib import Path
from scipy.integrate import solve_ivp
from spyral_utils.nuclear.target import GasTarget
from random import normalvariate
from .. constants import MEV_2_JOULE, MEV_2_KG, C, E_CHARGE
from .. import nuclear_map

TIME_WINDOW: float = 1.0e-10  # 0.1 ns window

class SimEvent:
    """A simulated event from the kinematics pipeline.

    Parameters
    ----------
    params: Parameters
        All parameters for simulation.

    Attributes
    ----------
    data: np.ndarray
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
    """

    def __init__(
        self,
        params: Parameters,
        data: np.ndarray,
        distance: float,
        proton_numbers: np.ndarray,
        mass_numbers: np.ndarray
    ):
        self.data = data
        self.distance = distance
        self.nuclei: list[SimParticle] = []

        # Only simulate the nuclei in the exit channel
        counter = 2
        while counter < data.shape[0]:
            self.nuclei.append(SimParticle(params, data[counter], distance,
                                           proton_numbers[counter], mass_numbers[counter]))
            counter += 3
        self.nuclei.append(SimParticle(params, data[counter - 2], distance,
                                       proton_numbers[counter - 2], mass_numbers[counter - 2]))


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
    
    Methods
    ----------
    initial_state -> np.ndarray
        Calculates 6x1 initial state vector of nucleus
        to be used in ODE solver.
    generate_track -> scipy.integrate.solve_ivp bunch object
        Solves EoM for track of nucleus in gas target

    """
    def __init__(
            self,
            params: Parameters,
            data: np.ndarray,
            distance: float,
            proton_number: int,
            mass_number: int
    ):
        self.data = data
        self.nucleus = nuclear_map.get_data(proton_number, mass_number)

        self.initial_state(distance)
        #self.track = self.generate_track(params)
        #self.electrons = self.generate_electrons()

    def initial_state(self, distance):
        """
        Make the 1x6 vector of the initial state of the nucleus.

        Parameters
        ----------
        distance: float
            Linear distance the incoming nucleus travelled before it underwent a reaction.
            The angle between the incoming particle and the detector is 0 degrees.

        Returns
        ----------
        np.ndarray
            1x6 array of the initial state of the nucleus. Format is [x, y, z, vx, vy, vz].
        """
        initial_state: np.ndarray = np.zeros(6)
        initial_state[2] = distance # Vertex
        print(self.data)
        initial_state[3:] = self.data[:3] / self.data[3] # Velocity CONFIRM UNITS HEREEEEEEEEEEE
        return initial_state

    def generate_track(
            self,
            params: Parameters
    ):
        """
        Solve the EoM of the nucleus in the AT-TPC.

        Returns
        -------
        scipy.integrate.solve_ivp bunch object
            Solution to ODE
        """
        track = solve_ivp(
            equation_of_motion,
            (0.0, TIME_WINDOW),
            self.initial_state(),
            method="RK45",
            events=[
                stop_condition,
                forward_z_bound_condition,
                backward_z_bound_condition,
                rho_bound_condition,
                ],
            args=(params.detector.bfield, params.detector.efield, params.gas_target, self.nucleus),
        )
        return track
    
    def generate_electrons(self,
                           params: Parameters
    ):
        """
        Find the number of electrons made at each point of the nucleus' track.
        """
        # Find energy of nucleus at each point of its track
        velocity_squared = np.sum((track.y[3:] ** 2), axis=0)
        energy = self.nucleus.mass * (-1 + 1 / np.sqrt(1 - (velocity_squared / C) ** 2))    # in MeV

        # Find number of electrons created at each point of its track
        electrons = np.zeros(energy.shape[0] + 1)
        electrons[1:] = np.diff(energy)
        electrons *= 1e6 / params.w_value   # Convert to eV to match units of W-value

        # Adjust number of electrons by Fano factor
        for idx in range(electrons.shape[0]): electrons[idx] += normalvariate(0.0, np.sqrt(params.fano_factor * electrons[idx]))

def run_simulation(params: Parameters,
                   input_path: Path):
    """
    Runs the AT-TPC simulation with the input parameters on the specified
    kinematic data hdf5 file generated by the kinematic pipeline.

    Parameters
     ----------
    params: Parameters
        All parameters for simulation.
    input_path: Path
        Path to HDF5 file containing kinematics
    """

    input = h5.File(input_path, 'r')
    input_data_group = input['data']

    for event in range(input_data_group.attrs['n_events']):
        result =SimEvent(params,
                 input_data_group[f'event_{event}'],
                 input_data_group[f'event_{event}'].attrs['distance'],
                 input_data_group.attrs['proton_numbers'],
                 input_data_group.attrs['mass_numbers']
                )
        print(result.nuclei[0].initial_state)

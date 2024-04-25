import numpy as np
import h5py as h5
import vector
import shapely

from numba import njit
from .parameters import *
from .solver import *
from .diffusion import *
from pathlib import Path
from scipy.integrate import solve_ivp
from spyral_utils.nuclear.target import GasTarget
from random import normalvariate
from .. constants import MEV_2_JOULE, MEV_2_KG, C, E_CHARGE
from .. import nuclear_map

NUM_TB: float = 512

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
            params: Parameters,
            data: np.ndarray,
            distance: float,
            proton_number: int,
            mass_number: int
    ):
        self.data = data
        self.nucleus = nuclear_map.get_data(proton_number, mass_number)
        self.track = self.generate_track(params, distance)
        self.electrons = self.generate_electrons(params)
        self.generate_hits(params)

    def generate_track(
            self,
            params: Parameters,
            distance: float
    ):
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
        scipy.integrate.solve_ivp bunch object
            Solution to ODE
        """
        # Find initial state of nucleus. (x, y, z, vx, vy, vz)
        initial_state: np.ndarray = np.zeros(6)
        initial_state[2] = distance
        initial_state[3:] = self.data[:3] / self.data[3] * C
    
        #Set ODE stop conditions. See SciPy solve_ivp docs
        stop_condition.terminal = True
        stop_condition.direction = -1.0
        forward_z_bound_condition.terminal = True
        forward_z_bound_condition.direction = 1.0
        backward_z_bound_condition.terminal = True
        backward_z_bound_condition.direction = -1.0
        rho_bound_condition.terminal = True
        rho_bound_condition.direction = 1.0

        # Time steps to solve ODE at. Assume maximum time for nucleus to 
        # cross detector is the time taken travelling at 0.01 * C. Evaluate 
        # the time every 1e-11 sec.
        time_steps = np.linspace(0, 3.34e-7, 33401)#3341

        track = solve_ivp(
            equation_of_motion,
            (0.0, 1.0),   # Set upper time limit to 1 sec. This should never be reached
            initial_state,
            method="RK45",
            events=[
                stop_condition,
                forward_z_bound_condition,
                backward_z_bound_condition,
                rho_bound_condition,
                ],
            t_eval = time_steps,
            args=(params.detector.bfield, params.detector.efield,\
                  params.detector.gas_target, self.nucleus),
        )

        return track.y
    
    def generate_electrons(self,
                           params: Parameters
    ):
        """
        Find the number of electrons made at each point of the nucleus' track.
        """
        # Find energy of nucleus at each point of its track
        velocity_squared = np.sum((self.track[3:] ** 2), axis=0)
        energy = self.nucleus.mass * (-1 + 1 / np.sqrt(1 - velocity_squared / (C ** 2)))    # in MeV

        # Find number of electrons created at each point of its track
        electrons = np.zeros_like(energy)
        electrons[1:] = abs(np.diff(energy))    # Magnitude of energy lost
        electrons *= 1e6 / params.detector.w_value  # Convert to eV to match units of W-value

        # Adjust number of electrons by Fano factor
        fano_adjusted: list = [normalvariate(point, np.sqrt(params.detector.fano_factor * point))
                           for point in electrons]
        fano_adjusted: np.ndarray = np.array(fano_adjusted)
        
        # Must have an integer amount of electrons at each point
        fano_adjusted = np.round(fano_adjusted)

        # Remove points in trajectory that create less than 1 electron
        mask = fano_adjusted > 1.0
        self.track = self.track[:, mask]
        fano_adjusted = fano_adjusted[mask]
    
        return fano_adjusted
    
    def z_to_tb(self,
                   params: Parameters):
        """
        Converts the z-coordinate of each point in the track
        from physical space (m) to time (time buckets).

        Parameters
        ----------
        params: Parameters
            All parameters for simulation.

        Returns
        -------
        np.ndarray
            1xN array of time buckets for N points in the track
            that produce one or more electrons.
        """
        # Z coordinates of points in track
        zpos: np.ndarray = self.track[2]

        dv: float = params.calculate_drift_velocity()
        
        # Convert from m to time buckets
        zpos = np.abs(zpos - params.detector.length)  # We want z relative to the micromegas
        zpos /= dv
        zpos += params.electronics.micromegas_edge

        return zpos
     
    def generate_hits(self,
                      params: Parameters
    ):
        """
        """
        # Store results
        # traces = {key: np.zeros(NUM_TB) for key, _ in params.pads.items()}
        # print(traces.keys())
        results = {}
        
        # Apply gain factor from micropattern gas detectors
        electrons: np.ndarray = self.electrons * params.detector.mpgd_gain

        # Find time bucket of each point in track
        point_tb: np.ndarray = self.z_to_tb(params)

        dv: float = params.calculate_drift_velocity()

        for idx, tb in enumerate(point_tb):
            # Standard deviation of diffusion gaussians
            sigma_transverse: float = np.sqrt(2 * params.detector.diffusion[0] *
                                              dv * tb / params.detector.efield)
            sigma_longitudinal: float = np.sqrt(2 * params.detector.diffusion[1] *
                                                dv * tb / params.detector.efield)
            
            # Make boundary of transverse diffusion gaussian
            boundary: shapely.Polygon = diffusion_boundary(
                                            params,
                                            (self.track[0][idx],
                                             self.track[1][idx]),
                                            sigma_transverse)
            
            # Find pads hit
            pads_hit = {key: pad for key, pad in params.pads.items()
                        if boundary.intersects(pad)}

            do_diffusion(results,
                         pads_hit,
                         electrons[idx],
                         tb,
                         sigma_transverse,
                         sigma_longitudinal)
            if idx == 0:
                break
    
        
def run_simulation(params: Parameters,
                   input_path: Path
    ):
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
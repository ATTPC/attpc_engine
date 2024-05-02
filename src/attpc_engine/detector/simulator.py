import numpy as np
import h5py as h5
import math
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

NUM_TB: int = 512

def get_response(params: Parameters):
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
        c1 = 4095 * E_CHARGE / params.electronics.amp_gain / 1e-15 # Should be 4096 or 4095?
        c2 = tb / (params.electronics.shaping_time *
                   params.electronics.clock_freq * 0.001)
        response[tb] = c1 * math.exp(-3 * c2) * (c2 ** 3) * math.sin(c2)

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
        params: Parameters,
        kine: np.ndarray,
        distance: float,
        proton_numbers: np.ndarray,
        mass_numbers: np.ndarray):
        self.kine = kine
        self.distance = distance
        self.nuclei: list[SimParticle] = []

        # Only simulate the nuclei in the exit channel
        counter = 2
        while counter < kine.shape[0]:
            self.nuclei.append(SimParticle(params, kine[counter], distance,
                                           proton_numbers[counter], mass_numbers[counter]))
            counter += 3
        self.nuclei.append(SimParticle(params, kine[counter - 2], distance,
                                       proton_numbers[counter - 2], mass_numbers[counter - 2]))
        
        self.data = self.digitize(params)
        
    def digitize(self,
                 params: Parameters):
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
        np.ndarray
            Returns a Nx(NUM_TB+5) array where each row is the 
            complete trace of one pad.
        """
        # Find all pads hit in simulated event
        pads_hit = {}
        for particle in self.nuclei:
            pads_hit = merge_dicts(pads_hit, particle.hits)
        
        # Cannot digitize event where no pads are hit
        if not bool(pads_hit):
            return None
        
        # Response function of GET electronics
        response: np.ndarray = get_response(params)

        # Make AT-TPC traces
        data = np.zeros((len(pads_hit), NUM_TB+5))
        for idx in enumerate(pads_hit.items()):
            # Convolve electrons with response function
            signal = np.convolve(idx[1][1],
                                 response,
                                 mode='full').astype(int)
            
            # Convolution cannot extend past the number of time buckets
            signal = signal[:NUM_TB]

            # Set saturated pads to max ADC
            signal[signal > 4095] = 4095

            # Combine hardware ID with signal for complete AT-TPC trace
            trace = np.concatenate((params.elec_map[idx[1][0]],
                                    signal),
                                   axis=None)
            data[idx[0]] = trace

        return data

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
            mass_number: int):
        self.data = data
        self.nucleus = nuclear_map.get_data(proton_number, mass_number)
        self.track = self.generate_track(params, distance)
        self.electrons = self.generate_electrons(params)
        self.hits = self.generate_hits(params)

    def generate_track(
            self,
            params: Parameters,
            distance: float):
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
        initial_state[2] = distance # m
        initial_state[3:] = self.data[:3] * MEV_2_KG * C   # kg * m/s
    
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
        time_steps = np.linspace(0, 3.34e-7, 33401)

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
                           params: Parameters):
        """
        Find the number of electrons made at each point of the nucleus' track.

        Parameters
        ----------
        params: Parameters
            All parameters for simulation.

        Returns
        -------
        np.ndarray
            6xN array where each row is a solution to one of the ODEs evaluated at
            the Nth time step, or point, of its trajectory.
        """
        # Find energy of nucleus at each point of its track
        momentum: float = np.sqrt(np.sum((self.track[3:] ** 2), axis=0))
        mass_kg = self.nucleus.mass * MEV_2_KG
        p_m = momentum / mass_kg
        speed = p_m / np.sqrt((1.0 + (p_m * 1.0 / C) ** 2.0))
        energy = self.nucleus.mass * (-1 + 1 / np.sqrt(1 - (speed / C) ** 2))   # in MeV

        # Find number of electrons created at each point of its track
        electrons = np.zeros_like(energy)
        electrons[1:] = abs(np.diff(energy))    # Magnitude of energy lost
        electrons *= 1e6 / params.detector.w_value  # Convert to eV to match units of W-value

        # Adjust number of electrons by Fano factor
        fano_adjusted = np.array([normalvariate(point, np.sqrt(params.detector.fano_factor *
                                                               point))
                                                               for point in electrons])
        
        # Must have an integer amount of electrons at each point
        fano_adjusted = fano_adjusted.astype(int)

        # Remove points in trajectory that create less than 1 electron
        mask = fano_adjusted >= 1
        self.track = self.track[:, mask]
        fano_adjusted = fano_adjusted[mask]
    
        return fano_adjusted
    
    def z_to_tb(self,
                params: Parameters):
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

        Returns
        -------
        np.ndarray[floats]
            1xN array of time buckets for N points in the track
            that produce one or more electrons.
        """
        # Z coordinates of points in track
        zpos: np.ndarray = self.track[2]

        dv: float = params.calculate_drift_velocity()
        
        # Convert from m to time buckets
        zpos = np.abs(zpos - params.detector.length)  # z relative to the micromegas
        zpos /= dv
        zpos += params.electronics.micromegas_edge

        return zpos
     
    def generate_hits(self,
                      params: Parameters):
        """
        Finds the pads hit by the electrons transported from each point
        of the nucleus' trajectory to the pad plane.

        Parameters
        ----------
        params: Parameters
            All parameters for simulation.

        Returns
        -------
        dict[int: np.ndarray[int]]
            Dictionary of pads hit by transporting the liberated electrons.
            The key is the pad number and the value is a 1xNUM_TB array 
            of the number of electrons detected at each time bucket.
        """
        results = {}
        
        # Apply gain factor from micropattern gas detectors
        electrons: np.ndarray = self.electrons * params.detector.mpgd_gain

        # Find time bucket of each point in track
        point_tb: np.ndarray = self.z_to_tb(params)

        for idx, tb in enumerate(point_tb):
            point = transport_point(params,
                                  (self.track[0][idx],
                                   self.track[1][idx]),
                                  electrons[idx],
                                  tb)
            intermediate: dict[int: np.ndarray] = point.pads
            results = merge_dicts(results, intermediate)

        return results

def merge_dicts(old: dict,
                new: dict):
    """
    """
    if not bool(old) and not bool(new):
        return {}
    
    elif not bool(old):
        return new
    
    elif not bool(new):
        return old
    
    else:
        combined = {key: old.get(key, 0) + new.get(key, 0)
                  for key in set(old) | set(new)}  
        return combined

def run_simulation(params: Parameters,
                   input_path: str,
                   output_path: str
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

    # Questions: should I  make input group, should i make header, should i include timestamps
    # Construct output HDF5 file in AT-TPC format
    output_file = h5.File(output_path, 'w')
    get_group = output_file.create_group('get')
    meta_group = output_file.create_group('meta')

    input = h5.File(input_path, 'r')
    input_data_group = input['data']

    evt_counter: int = 0
    for event in range(input_data_group.attrs['n_events']):
        sim = SimEvent(params,
                          input_data_group[f'event_{event}'],
                          input_data_group[f'event_{event}'].attrs['distance'],
                          input_data_group.attrs['proton_numbers'],
                          input_data_group.attrs['mass_numbers'])

        if sim.data is None:
            continue

        get_group.create_dataset(f'evt{evt_counter}_data', data=sim.data)

    # Make event range dataset
    meta_data = np.array([0, 0, evt_counter, 0])
    meta_group.create_dataset('meta', data=meta_data)
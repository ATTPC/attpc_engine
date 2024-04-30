from dataclasses import dataclass
from spyral_utils.nuclear.target import GasTarget

import numpy as np
import shapely

@dataclass
class Detector_Params:
    """
    Data class containing all detector parameters for simulation.

    Attributes
    ----------
    length: float (m)
        Length of active volume of detector.
    efield: float (V/m)
        Magnitude of the electric field. The electric field is 
        assumed to only have one component in the +z direction
        parallel to the incoming beam.
    bfield: float (T)
        Magnitude of the magnetic field. The magnetic field is 
        assumed to only have one component in the +z direction
        parallel to the incoming beam.
    mpgd_gain: int (unitless)
        Overall gain of all micropattern gas detectors used, e.g.
        a combination of a micromegas and THGEM.
    gas_target: GasTarget
        Target gas in the AT-TPC.
    drift_velocity: float (cm/us)
        Electron drift velocity in target gas.
    diffusion: tuple(float, float) (V,V)
        Diffusion coefficients of electrons in the target gas. The
        first element is the transverse coefficient and the second
        is the longitudinal coefficient.
    fano_factor: float (unitless)
        Fano factor of target gas.
    w_value: float (eV)
        W-value of gas. This is the average energy an ionizing
        loses to create one electron-ion pair in the gas.
    pad_vertices: str
        Path to location of file containing pads and their
        vertices. Each row is formatted as (pad number, x1,
        y1, x2, y2, ...,xn, yn) where x and y are the
        respective coordinates of the nth vertex given in mm.
    """
    length: float
    efield: float 
    bfield: float 
    mpgd_gain: int
    gas_target: GasTarget
    diffusion: tuple[float,float]
    fano_factor: float
    w_value: float
    pad_map: str
    pad_map_parameters: tuple[float, float, float]

@dataclass
class Electronics_Params:
    """
    Data class containing all electronics parameters for simulation.

    Attributes
    ----------
    clock_freq: float (MHz)
        Frequency of the GET clock.
    amp_gain: int (fC)
        Gain of GET amplifier.
    shaping_time: int (ns)
        Shaping time of GET.
    micromegas_edge: int (timebucket)
        The micromegas edge of the detector.
    windows_edge: int (timebucket)
        The windows edge of the detector.
    """
    clock_freq: float
    amp_gain: int
    shaping_time: int
    micromegas_edge: int
    windows_edge: int

class Parameters:
    """
    A wrapper class containing all the input detector and electronics parameters
    for the simulation.

    Attributes
    ----------
    detector: Detector_Params
        Detector parameters
    electronics: Electronics_Params
        Electronics parameters
    """
    def __init__(
            self,
            detector_params: Detector_Params,
            electronics_params: Electronics_Params
    ):
        self.detector = detector_params
        self.electronics = electronics_params
        self.pads = self.make_padplane()

    def calculate_drift_velocity(self):
        """
        Calculate drift velocity of electrons in the gas.

        Returns
        -------
        float
            Electron drift velocity in m / time bucket
        """
        dv: float = self.detector.length / (self.electronics.windows_edge -
                                            self.electronics.micromegas_edge)
        return dv 
        
    def make_padplane(self):
        """
        Makes a list of the pad plane pads.

        Returns
        -------
        pads: list[shapely.Polygon]
            List of polygons, one for each pad
        """
        pad_map: np.ndarray = np.loadtxt(self.detector.pad_map,
                                         dtype='int',
                                         delimiter=',',
                                         skiprows=0)

        return pad_map
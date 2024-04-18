from dataclasses import dataclass
from spyral_utils.nuclear.target import GasTarget

@dataclass
class Detector_Params:
    """
    Data class containing all detector parameters for simulation.

    Attributes
    ----------
    length: float (mm)
        Length of AT-TPC active volume.
    efield: float (V/m)
        Magnitude of the electric field. The electric field is 
        assumed to only have one component in the +z direction
        parallel to the incoming beam.
    bfield: float (T)
        Magnitude of the magnetic field. The magnetic field is 
        assumed to only have one component in the +z direction
        parallel to the incoming beam.
    gas_target: GasTarget
        Target gas in the AT-TPC
    drift_velocity: float (cm/us)
        Electron drift velocity in target gas.
    diffusion: tuple(float, float) (V,V)
        Diffusion coefficients of electrons in the target gas. The
        first element is the transvere coefficient and the second
        is the longitudinal coefficient.
    fano_factor: float (unitless)
        Fano factor of target gas.
    w_value: float (eV)
        W-value of gas. This is the average energy an ionizing
        loses to create one electron-ion pair in the gas.

    """
    length: float 
    efield: float 
    bfield: float 
    gas_target: GasTarget
    drift_velocity: float 
    diffusion: tuple(float,float)
    fano_factor: float
    w_value: float

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
    mpgd_gain: int (unitless)
        Overall gain of all micropattern gas detectors used, e.g.
        a combination of a micromegas and THGEM
    micromegas_edge: int (timebucket)
        The micromegas edge of the detector.
    """
    clock_freq: float
    amp_gain: int
    shaping_time: int
    mpgd_gain: int
    micromegas_edge: int


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
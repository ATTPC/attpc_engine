import numpy as np

from dataclasses import dataclass
from spyral_utils.nuclear.target import GasTarget
from numba import int64
from numba.typed import Dict
from importlib import resources

DEFAULT = "Default"
DEFAULT_LEGACY = "DefaultLegacy"


@dataclass
class DetectorParams:
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
    diffusion: tuple(float, float) (V,V)
        Diffusion coefficients of electrons in the target gas. The
        first element is the transverse coefficient and the second
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
    mpgd_gain: int
    gas_target: GasTarget
    diffusion: tuple[float, float]
    fano_factor: float
    w_value: float


@dataclass
class ElectronicsParams:
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
    adc_threshold: float


@dataclass
class PadParams:
    """
    Data class containing parameters related to the pads.

    Attributes
    ----------
    map: str
        Path to pad map LUT.
    map_params: tuple[float, float, float]
        LUT parameters. First element is the low edge of the
        grid, second element is the high edge, and the third
        element is the size of each pixel in the map.
    electronics: str
        Path to electronics file containing the hardware ID
        for each pad.
    geometry: str
        Path to pad geometry file, containing each pad's center position.
        Used for conversion to Spyral point cloud
    """

    grid_path: str = DEFAULT
    electronics_path: str = DEFAULT
    geometry_path: str = DEFAULT


class Config:
    """
    A wrapper class containing all the input detector and electronics parameters
    for the simulation.

    Attributes
    ----------
    detector: DetectorParams
        Detector parameters
    electronics: ElectronicsParams
        Electronics parameters
    pads: PadParams
        Pad parameters

    Methods
    -------
    calculate_drift_velocity()
        Calculates the electron drift velocity.
    load_pad_map()
        Loads pad map LUT.
    pad_to_hardwareid()
        Makes a mapping from pad number to hardware ID.
    """

    def __init__(
        self,
        detector_params: DetectorParams,
        electronics_params: ElectronicsParams,
        pad_params: PadParams,
    ):
        self.det_params = detector_params
        self.elec_params = electronics_params
        self.pad_params = pad_params
        self.pad_grid: np.ndarray | None = None
        self.pad_grid_edges: np.ndarray | None = None
        self.pad_centers: np.ndarray | None = None
        self.drift_velocity = 0.0
        self.calculate_drift_velocity()
        self.load_pad_grid()
        self.load_pad_centers()

    def calculate_drift_velocity(self) -> None:
        """
        Calculate drift velocity of electrons in the gas.

        Returns
        -------
        float
            Electron drift velocity in m / time bucket
        """
        self.drift_velocity = self.det_params.length / float(
            self.elec_params.windows_edge - self.elec_params.micromegas_edge
        )

    def load_pad_grid(self) -> None:
        """
        Loads pad map LUT as an array.

        Returns
        -------
        map: np.ndarray
            Array indexed by physical position that
            returns the pad number at that position.
        """
        if (
            self.pad_params.grid_path == DEFAULT
            or self.pad_params.grid_path == DEFAULT_LEGACY
        ):
            grid_handle = resources.files("attpc_engine.detector.data").joinpath(
                "pad_grid.npz"
            )
            with resources.as_file(grid_handle) as grid_path:
                data = np.load(grid_path)
                self.pad_grid = data["grid"]
                self.pad_grid_edges = data["edges"]
        else:
            data = np.load(self.pad_params.grid_path)
            self.pad_grid = data["grid"]
            self.pad_grid_edges = data["edges"]

    def load_pad_centers(self) -> None:
        self.pad_centers = np.zeros((10240, 2))
        if self.pad_params.geometry_path == DEFAULT:
            geom_handle = resources.files("attpc_engine.detector.data").joinpath(
                "padxy.csv"
            )
            with resources.as_file(geom_handle) as geopath:
                geofile = open(geopath, "r")
                geofile.readline()  # Remove header
                lines = geofile.readlines()
                for pad_number, line in enumerate(lines):
                    entries = line.split(",")
                    self.pad_centers[pad_number, 0] = float(entries[0])
                    self.pad_centers[pad_number, 1] = float(entries[1])
                geofile.close()
        elif self.pad_params.geometry_path == DEFAULT:
            geom_handle = resources.files("attpc_engine.detector.data").joinpath(
                "padxy_legacy.csv"
            )
            with resources.as_file(geom_handle) as geopath:
                geofile = open(geopath, "r")
                geofile.readline()  # Remove header
                lines = geofile.readlines()
                for pad_number, line in enumerate(lines):
                    entries = line.split(",")
                    self.pad_centers[pad_number, 0] = float(entries[0])
                    self.pad_centers[pad_number, 1] = float(entries[1])
                geofile.close()
        else:
            with open(self.pad_params.geometry_path, "r") as geofile:
                geofile.readline()  # Remove header
                lines = geofile.readlines()
                for pad_number, line in enumerate(lines):
                    entries = line.split(",")
                    self.pad_centers[pad_number, 0] = float(entries[0])
                    self.pad_centers[pad_number, 1] = float(entries[1])
                geofile.close()

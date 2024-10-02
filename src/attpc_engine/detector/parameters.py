import numpy as np
from dataclasses import dataclass
from spyral_utils.nuclear.target import GasTarget
from importlib import resources
from pathlib import Path

DEFAULT = "Default"


@dataclass
class DetectorParams:
    """Dataclass containing all detector parameters for simulation.

    Attributes
    ----------
    length: float
        Length of active volume of detector in meters
    efield: float
        Magnitude of the electric field in Volts/meter. The electric field is
        assumed to only have one component in the +z direction
        parallel to the incoming beam.
    bfield: float
        Magnitude of the magnetic field in Tesla. The magnetic field is
        assumed to only have one component in the +z direction
        parallel to the incoming beam.
    mpgd_gain: int
        Overall gain of all micropattern gas detectors used, e.g.
        a combination of a micromegas and THGEM. Unitless.
    gas_target: spyral_utils.nuclear.GasTarget
        Target gas in the AT-TPC
    diffusion: float
        Transverse diffusion coefficient of electrons in the target gas.
        In units of Volts
    fano_factor: float
        Fano factor of target gas. Unitless.
    w_value: float
        W-value of gas in eV. This is the average energy an ionizing
        loses to create one electron-ion pair in the gas.
    """

    length: float
    efield: float
    bfield: float
    mpgd_gain: int
    gas_target: GasTarget
    diffusion: float
    fano_factor: float
    w_value: float


@dataclass
class ElectronicsParams:
    """Dataclass containing all electronics parameters for simulation.

    Attributes
    ----------
    clock_freq: float
        Frequency of the GET clock in MHz.
    amp_gain: int
        Gain of GET amplifier in lsb/fC.
    shaping_time: int
        Shaping time of GET in ns.
    micromegas_edge: int
        The micromegas edge of the detector in time buckets.
    windows_edge: int
        The windows edge of the detector in time buckets.
    adc_threshold: int
        Minimum ADC signal amplitude a point must have in the point cloud.
    """

    clock_freq: float
    amp_gain: int
    shaping_time: int
    micromegas_edge: int
    windows_edge: int
    adc_threshold: int


@dataclass
class PadParams:
    """Dataclass containing parameters related to the pads.

    Attributes
    ----------
    grid_path: Path | str
        The path to the pad grid. Default is to use the packaged grid.
    geometry_path: Path | str
        The path to the pad center geometry. Default is to use
        the packaged geometry (non-legacy).
    """

    grid_path: Path | str = DEFAULT
    geometry_path: Path | str = DEFAULT
    pad_size_path: Path | str = DEFAULT


class Config:
    """
    A wrapper class containing all the input detector and electronics parameters
    for the simulation.

    Parameters
    ----------
    detector_params: DetectorParams
        Detector parameters for simulation.
    electronics_params: ElectronicsParams
        Electronics parameters for simulation.
    pad_params: PadParams
        Parameters related to the pads.

    Attributes
    ----------
    detector_params: DetectorParams
        Detector parameters for simulation.
    electronics_params: ElectronicsParams
        Electronics parameters for simulation.
    pad_params: PadParams
        Parameters related to the pads.
    pad_grid: np.ndarray | None
        Mesh of the pad plane. See load_pad_grid()
        for more details.
    pad_grid_edges: np.ndarray | None
        a 1x3 array where the first element is the leftmost edge of the
        pad grid, the second element is the rightmost edge, and the third
        element is the step size in mm.
    pad_centers: np.ndarray | None
        10240x2 array where where each row is one pad and its first element
        is the x coordinate of its center and the second element is the
        y coordinate.
    drift_velocity: float
        Drift velocity in m / time bucket.
    pad_sizes: np.ndarray | None


    Methods
    -------
    calculate_drift_velocity() -> None
        Calculate drift velocity of electrons in the gas.
    load_pad_grid() -> None
        Load the pad grid from an npz file.
    load_pad_centers() -> None
        Load the pad plane centers from a csv file.
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
        # Set everything
        self.calculate_drift_velocity()
        self.load_pad_grid()
        self.load_pad_centers()
        self.load_pad_sizes()

    def calculate_drift_velocity(self) -> None:
        """Calculate drift velocity of electrons in the gas.

        Returns
        -------
        float
            Electron drift velocity in m / time bucket
        """
        self.drift_velocity = self.det_params.length / float(
            self.elec_params.windows_edge - self.elec_params.micromegas_edge
        )

    def load_pad_grid(self) -> None:
        """Load the pad grid from an npz file.

        The pad grid is a regular mesh which descretizes the pad plane into
        a NxN matrix. Each element contains the pad ID at that location
        in space. If that location corresponds to no pad, then the pad ID is -1.
        The grid file should also contain a 3 element array with
        the grid edges in mm and the step size of the grid.

        The pad grid is indexed by the lower edge of each spatial bin. Due to this,
        and because the pad grid must be symmetric, this means that the Nth column
        and row are not the same as the 0th row and column. For example, if the pad
        grid is to be symmetric from -280 to 280 mm with spatial bins of 0.1 mm,
        the Oth row and column correspond to bin edges of -280 (the bin is [-280, 279.9))
        while the Nth row and column correspond to bin edges of 279.9 (the bin is
        [-279.9, 280). In this sense, the pad grid is inclusive on the lowest bin
        edge and exclusive on the highest bin edge.
        """
        if self.pad_params.grid_path == DEFAULT:
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
        """Load the pad plane centers from a csv file.

        Load a csv file contains the x, y center position of each pad.
        """
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
        else:
            with open(self.pad_params.geometry_path, "r") as geofile:
                geofile.readline()  # Remove header
                lines = geofile.readlines()
                for pad_number, line in enumerate(lines):
                    entries = line.split(",")
                    self.pad_centers[pad_number, 0] = float(entries[0])
                    self.pad_centers[pad_number, 1] = float(entries[1])
                geofile.close()

    def load_pad_sizes(self) -> None:
        """
        Load the size of each pad from a csv file into a 1D numpy array
        where the index is the pad number and the entry is the size.
        """
        self.pad_sizes = np.zeros(10240)
        if self.pad_params.pad_size_path == DEFAULT:
            geom_handle = resources.files("attpc_engine.detector.data").joinpath(
                "pad_scale.csv"
            )
            with resources.as_file(geom_handle) as geopath:
                geofile = open(geopath, "r")
                geofile.readline()  # Remove header
                lines = geofile.readlines()
                for pad_number, line in enumerate(lines):
                    entries = line.split(",")
                    self.pad_sizes[pad_number] = float(entries[0])
                geofile.close()
        else:
            with open(self.pad_params.geometry_path, "r") as geofile:
                geofile.readline()  # Remove header
                lines = geofile.readlines()
                for pad_number, line in enumerate(lines):
                    entries = line.split(",")
                    self.pad_sizes[pad_number] = float(entries[0])
                geofile.close()

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
    adc_threshold: float
        Minimum ADC signal amplitude a point must have in the point cloud.
    """

    clock_freq: float
    amp_gain: int
    shaping_time: int
    micromegas_edge: int
    windows_edge: int
    adc_threshold: float


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
        """Load the pad grid from an npz file

        The pad grid is a regular mesh which descretizes the pad plane into
        a NxN matrix. Each element contains the pad ID at that location
        in space. The grid file should also contain a 3 element array with
        the grid edges in mm and the step size of the grid.

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
        """Load the pad plane centers from a csv file

        Load a csv file contains the x, y center position of each pad

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

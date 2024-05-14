import numpy as np

from dataclasses import dataclass
from spyral_utils.nuclear.target import GasTarget
from numba import int64
from numba.typed import Dict
from importlib import resources
from typing import Any

DEFAULT_PAD_GEOMETRY: str = "Default"
DEFAULT_LEGACY_PAD_GEOMETRY: str = "DefaultLegacy"


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

    map: str
    map_params: tuple[float, float, float]
    electronics: str
    geometry: str = DEFAULT_PAD_GEOMETRY


@dataclass
class PadData:
    x: float
    y: float


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
        self.detector = detector_params
        self.electronics = electronics_params
        self.pads = pad_params
        self.pad_map = self.load_pad_map()
        # self.hardwareid_map = self.pad_to_hardwareid()
        self.pad_data = self.load_pad_data()

    def calculate_drift_velocity(self) -> float:
        """
        Calculate drift velocity of electrons in the gas.

        Returns
        -------
        float
            Electron drift velocity in m / time bucket
        """
        dv: float = self.detector.length / (
            self.electronics.windows_edge - self.electronics.micromegas_edge
        )
        return dv

    def load_pad_map(self) -> np.ndarray:
        """
        Loads pad map LUT as an array.

        Returns
        -------
        map: np.ndarray
            Array indexed by physical position that
            returns the pad number at that position.
        """
        map: np.ndarray = np.loadtxt(
            self.pads.map, dtype=np.int64, delimiter=",", skiprows=0
        )

        return map

    def load_pad_data(self) -> dict[int, PadData]:
        map: dict[int, PadData] = {}
        if self.pads.geometry == DEFAULT_PAD_GEOMETRY:
            geom_handle = resources.files("attpc_engine.detector.data").joinpath(
                "padxy.csv"
            )
            with resources.as_file(geom_handle) as geopath:
                geofile = open(geopath, "r")
                geofile.readline()  # Remove header
                lines = geofile.readlines()
                for pad_number, line in enumerate(lines):
                    entries = line.split(",")
                    map[pad_number] = PadData(x=float(entries[0]), y=float(entries[1]))
                geofile.close()
            return map
        elif self.pads.geometry == DEFAULT_LEGACY_PAD_GEOMETRY:
            geom_handle = resources.files("attpc_engine.detector.data").joinpath(
                "padxy_legacy.csv"
            )
            with resources.as_file(geom_handle) as geopath:
                geofile = open(geopath, "r")
                geofile.readline()  # Remove header
                lines = geofile.readlines()
                for pad_number, line in enumerate(lines):
                    entries = line.split(",")
                    map[pad_number] = PadData(x=float(entries[0]), y=float(entries[1]))
                geofile.close()
            return map
        else:
            with open(self.pads.geometry, "r") as geofile:
                geofile.readline()  # Remove header
                lines = geofile.readlines()
                for pad_number, line in enumerate(lines):
                    entries = line.split(",")
                    map[pad_number] = PadData(x=float(entries[0]), y=float(entries[1]))
                geofile.close()
            return map

    def get_pad_data(self, pad_id: int) -> PadData | None:
        return self.pad_data[pad_id]

    def pad_to_hardwareid(self) -> dict[int, np.ndarray]:
        """
        Creates a dictionary mapping each pad number to its hardware ID.

        Returns
        -------
        numba.typed.Dict[int, np.ndarray]
            Numba typed dictionary mapping pad number to hardware ID. The hardware ID
            is a 1x5 array with signature (CoBo, AsAd, Aget, Aget channel, Pad).
        """
        map = Dict.empty(int64, int64[:])
        with open(self.pads.electronics, "r") as elecfile:
            elecfile.readline()
            lines = elecfile.readlines()
            for line in lines:
                entries = line.split(",")
                hardware = np.array(
                    (
                        int(entries[0]),
                        int(entries[1]),
                        int(entries[2]),
                        int(entries[3]),
                        int(entries[4]),
                    )
                )
                map[int(entries[4])] = hardware

        return map

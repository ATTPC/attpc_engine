from typing import Protocol
from pathlib import Path

import numpy as np
import h5py as h5

from .parameters import Config
from .response import apply_response, get_response
from numba import njit


class SimulationWriter(Protocol):

    def write(self, data: np.ndarray, config: Config, event_number: int) -> None:
        pass

    def set_number_of_events(self, n_events: int) -> None:
        pass

    def get_filename(self) -> Path: ...


@njit
def convert_to_spyral(
    points: np.ndarray,
    window_edge: float,
    mm_edge: float,
    length: float,
    response: np.ndarray,
    pad_centers: np.ndarray,
    adc_threshold: float,
) -> np.ndarray:

    storage = np.empty((len(points), 8))
    for idx, point in enumerate(points):
        pad = pad_centers[int(point[0])]
        amp, integral = apply_response(response, point[2])
        storage[idx, 0] = pad[0]
        storage[idx, 1] = pad[1]
        storage[idx, 2] = (
            (window_edge - point[1]) / (window_edge - mm_edge) * length * 1000.0
        )  # mm
        storage[idx, 3] = amp
        storage[idx, 4] = integral
        storage[idx, 5] = point[0]
        storage[idx, 6] = point[1]
        storage[idx, 7] = 1.0  # Idk we don't even really use this rn

    if adc_threshold >= 4095:
        raise ValueError(
            "adc_threshold cannot be equal to or greater than the max GET ADC value!"
        )
    return storage[storage[:, 3] > adc_threshold]


class SpyralWriter:

    def __init__(self, file_path: Path, config: Config):
        self.path = file_path
        self.file = h5.File(self.path, "w")
        self.cloud_group = self.file.create_group("cloud")
        self.response = get_response(config).copy()

    def write(self, data: np.ndarray, config: Config, event_number: int) -> None:
        if config.pad_centers is None:
            raise ValueError("Pad centers are not assigned at write!")
        spyral_format = convert_to_spyral(
            data,
            config.elec_params.windows_edge,
            config.elec_params.micromegas_edge,
            config.det_params.length,
            self.response,
            config.pad_centers,
            config.elec_params.adc_threshold,
        )

        dset = self.cloud_group.create_dataset(
            f"cloud_{event_number}", data=spyral_format
        )
        # No ic stuff from simulation
        dset.attrs["ic_amplitude"] = -1.0
        dset.attrs["ic_multiplicity"] = -1.0
        dset.attrs["ic_integral"] = -1.0
        dset.attrs["ic_centroid"] = -1.0

    def set_number_of_events(self, n_events: int) -> None:
        self.cloud_group.attrs["min_event"] = 0
        self.cloud_group.attrs["max_event"] = n_events - 1

    def get_filename(self) -> Path:
        return self.path

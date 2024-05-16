from typing import Protocol
from pathlib import Path

import numpy as np
import h5py as h5

from .parameters import Config
from .response import apply_response, get_response


class SimulationWriter(Protocol):

    def write(self, data: np.ndarray, config: Config, event_number: int) -> None:
        pass

    def set_number_of_events(self, n_events: int) -> None:
        pass


class SpyralWriter:

    def __init__(self, file_path: Path):
        self.path = file_path
        self.file = h5.File(self.path, "w")
        self.cloud_group = self.file.create_group("cloud")

    def convert_to_spyral(self, points: np.ndarray, config: Config) -> np.ndarray:
        spyral_points = np.empty((len(points), 8))
        response = get_response(config)

        for idx, point in enumerate(points):
            pad = config.get_pad_data(point[0])
            amp, integral = apply_response(response, point[2])
            if pad is None:
                continue
            spyral_points[idx, 0] = pad.x
            spyral_points[idx, 1] = pad.y
            spyral_points[idx, 2] = (
                (config.electronics.windows_edge - point[1])
                / (config.electronics.windows_edge - config.electronics.micromegas_edge)
                * config.detector.length
                * 1000.0
            )  # mm
            spyral_points[idx, 3] = amp  # Set them the same cause whatever
            spyral_points[idx, 4] = integral
            spyral_points[idx, 5] = point[0]
            spyral_points[idx, 6] = point[1]
            spyral_points[idx, 7] = 1.0  # Idk we don't even really use this rn

        return spyral_points

    def write(self, data: np.ndarray, config: Config, event_number: int) -> None:
        spyral_format = self.convert_to_spyral(data, config)
        self.cloud_group.create_dataset(f"cloud_{event_number}", data=spyral_format)

    def set_number_of_events(self, n_events: int) -> None:
        self.cloud_group.attrs["min_event"] = 0
        self.cloud_group.attrs["max_event"] = n_events - 1

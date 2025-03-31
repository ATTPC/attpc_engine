from typing import Protocol
from pathlib import Path

import numpy as np
import h5py as h5

from .parameters import Config
from .response import apply_response, get_response
from numba import njit


class SimulationWriter(Protocol):
    """
    Protocol class for what methods a simulation writer class should contain.

    Methods
    -------
    write(data, labels, config, event_number)
        Writes a simulated point cloud to the point cloud file.
    get_directory_name()
        Returns directory that point cloud files are written to.
    close()
        Closes the writer.
    """

    def write(
        self, data: np.ndarray, labels: np.ndarray, config: Config, event_number: int
    ) -> None:
        """Writes a simulated point cloud to the point cloud file.

        Parameters
        ----------
        data: numpy.ndarray
            An Nx3 array representing the point cloud. Each row is a point, with elements
            [pad id, time bucket, electrons].
        labels: numpy.ndarray
            A length N array containing labels for points in the pointcloud, indicating
            which nucleus produced the point.
        config: Config
            The simulation configuration.
        event_number: int
            Event number of simulated event from the kinematics file.
        """
        pass

    def get_directory_name(self) -> Path:  # type: ignore
        """Returns directory that point cloud files are written to.

        Returns
        -------
        pathlib.Path
            The path the output directory
        """
        pass

    def close(self) -> None:
        """Closes the writer."""
        pass


@njit
def convert_to_spyral(
    points: np.ndarray,
    window_edge: int,
    mm_edge: int,
    length: float,
    response: np.ndarray,
    pad_centers: np.ndarray,
    pad_sizes: np.ndarray,
) -> np.ndarray:
    """Converts a simulated point in the point cloud to Spyral formatting.

    Parameters
    ----------
    points: np.ndarray
        An Nx3 array representing the point cloud. Each row is a point, with elements
        [pad id, time bucket, electrons].
    window_edge: int
        The windows edge of the detector in time buckets.
    mm_edge: int
        The micromegas edge of the detector in time buckets.
    length: float
        Length of active volume of detector in meters
    response: np.ndarray
        Response of GET electronics.
    pad_centers: np.ndarray
        (x, y) coordinates of each pad's center on the pad plane in mm.
    pad_sizes: np.ndarray
        Contains size of each pad.

    Returns
    -------
    np.ndarray
        The point cloud
    """

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
        storage[idx, 7] = pad_sizes[int(point[0])]

    return storage


class SpyralWriter:
    """Writer for default Spyral analysis.

    Writes the simulated data into multiple files to take advantage of
    Spyral's multiprocessing.

    Parameters
    ----------
    directory_path: pathlib.Path
        Path to directory to store simulated point cloud files.
    config: Config
        The simulation configuration.
    max_events_per_file: int
        The maximum number of events per file. Once this limit is reached, a new file is opened.
        Default value is 5,000 events.
    first_run_number: int
        The starting run number. You can use this to change the starting point for run files
        (i.e. run_0000 or run_0008) to avoid overwritting previous results. Default is 0

    Attributes
    ----------
    directory_path: pathlib.Path
        The path to the directory data will be written to
    response: numpy.ndarray
        Response of GET electronics.
    max_events_per_file: int
        The maximum number of events per file
    run_number: int
        Run number of current point cloud file being written to.
    starting_event: int
        The first event number of the file currently being written to
    events_written: int
        The number of events that have been written
    file: h5py.File
        h5 file object. It is the actual point cloud file currently
        being written to.
    cloud_group: h5py.Group
        "cloud" group in current point cloud file.

    Methods
    -------
    write(data, config, event_number)
        Writes a simulated point cloud to the point cloud file.
    get_directory_name()
        Returns directory that point cloud files are written to.
    close()
        Closes the writer with metadata written
    """

    def __init__(
        self,
        directory_path: Path,
        config: Config,
        max_events_per_file: int = 5_000,
        first_run_number: int = 0,
    ):
        self.directory_path: Path = directory_path
        self.response: np.ndarray = get_response(config).copy()
        self.max_events_per_file: int = max_events_per_file
        self.run_number = first_run_number
        self.starting_event = 0  # Kinematics generator always starts with event 0
        self.last_event = 0  # What event number do we end on
        self.events_written = 0  # haven't written anything yet
        # initialize the first file
        path: Path = self.directory_path / f"run_{self.run_number:04d}.h5"
        self.file = h5.File(path, "w")
        self.cloud_group: h5.Group = self.file.create_group("cloud")

    def create_next_file(self) -> None:
        """Creates the next point cloud file

        Moves the run number forward and opens a new HDF5 file
        with the appropriate groups.
        """
        self.run_number += 1
        path: Path = self.directory_path / f"run_{self.run_number:04d}.h5"
        self.file = h5.File(path, "w")
        self.cloud_group: h5.Group = self.file.create_group("cloud")

    def write(
        self, data: np.ndarray, labels: np.ndarray, config: Config, event_number: int
    ) -> None:
        """
        Writes a simulated point cloud to the point cloud file.

        Parameters
        ----------
        data: numpy.ndarray
            An Nx3 array representing the point cloud. Each row is a point, with elements
            [pad id, time bucket, electrons].
        labels: numpy.ndarray
            A length N array of labels indicating which nucleus produced the point in
            the point cloud.
        config: Config
            The simulation configuration.
        event_number: int
            Event number of simulated event from the kinematics file.
        """
        # If we reach the event limit, make a new file
        if self.events_written == self.max_events_per_file:
            self.close()
            self.create_next_file()
            self.starting_event = event_number
            self.events_written = 0

        if config.pad_centers is None:
            raise ValueError("Pad centers are not assigned at write!")
        spyral_format = convert_to_spyral(
            data,
            config.elec_params.windows_edge,
            config.elec_params.micromegas_edge,
            config.det_params.length,
            self.response,
            config.pad_centers,
            config.pad_sizes,
        )
        # apply ADC threshold
        mask = spyral_format[:, 3] > config.elec_params.adc_threshold
        spyral_format = spyral_format[mask]
        labels = labels[mask]
        # Make sure we're still sorted in z
        indicies = np.argsort(spyral_format[:, 2])
        spyral_format = spyral_format[indicies]
        labels = labels[indicies]

        pc_dset = self.cloud_group.create_dataset(
            f"cloud_{event_number}", data=spyral_format
        )
        pc_dset.attrs["orig_run"] = self.run_number
        pc_dset.attrs["orig_event"] = event_number
        # No ic stuff from simulation
        pc_dset.attrs["ic_amplitude"] = -1.0
        pc_dset.attrs["ic_multiplicity"] = -1.0
        pc_dset.attrs["ic_integral"] = -1.0
        pc_dset.attrs["ic_centroid"] = -1.0

        _ = self.cloud_group.create_dataset(f"labels_{event_number}", data=labels)

        # We wrote an event
        self.last_event = event_number
        self.events_written += 1

    def set_number_of_events(self) -> None:
        """Writes event metadata

        Stores first and last event numbers in the attributes
        """
        self.cloud_group.attrs["min_event"] = self.starting_event
        self.cloud_group.attrs["max_event"] = self.last_event

    def get_directory_name(self) -> Path:
        """Returns directory that point cloud files are written to.

        Returns
        -------
        pathlib.Path
            The path to the point cloud directory
        """
        return self.directory_path

    def close(self) -> None:
        """Closes the writer with metadata written

        Ensures that the event range metadata is recorded
        """
        self.set_number_of_events()
        self.file.close()

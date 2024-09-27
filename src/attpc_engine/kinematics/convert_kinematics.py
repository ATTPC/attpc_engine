from .. import nuclear_map

import h5py as h5
import polars as pl
from pathlib import Path
from tqdm import trange
import numpy as np
import argparse


def convert_kinematics_hdf5_to_polars(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise Exception(f"Input path {input_path} does not exist!")

    hdf_file = h5.File(input_path, "r")

    input_data_group = hdf_file["data"]
    proton_numbers: np.ndarray = input_data_group.attrs["proton_numbers"]  # type: ignore
    mass_numbers: np.ndarray = input_data_group.attrs["mass_numbers"]  # type: ignore
    n_events: int = input_data_group.attrs["n_events"]  # type: ignore
    chunk_size: int = input_data_group.attrs["chunk_size"]  # type: ignore
    n_nuclei = len(proton_numbers)
    nuclei = [
        nuclear_map.get_data(proton_numbers[idx], mass_numbers[idx])  # type: ignore
        for idx in range(n_nuclei)
    ]

    # dataframe dictionary
    dataframe = {
        "event": [],
        "Z": [],
        "A": [],
        "isotope": [],
        "energy": [],
        "px": [],
        "py": [],
        "pz": [],
        "vertex_x": [],
        "vertex_y": [],
        "vertex_z": [],
    }

    for event in trange(n_events):
        chunk = event // chunk_size  # integer floor division
        dataset: h5.Dataset = input_data_group[f"chunk_{chunk}"][f"event_{event}"]  # type: ignore
        vx = dataset.attrs["vertex_x"]
        vy = dataset.attrs["vertex_y"]
        vz = dataset.attrs["vertex_z"]
        for idx in range(n_nuclei):
            dataframe["event"].append(event)
            dataframe["Z"].append(proton_numbers[idx])
            dataframe["A"].append(mass_numbers[idx])
            dataframe["isotope"].append(nuclei[idx].isotopic_symbol)
            dataframe["energy"].append(dataset[idx, 3])
            dataframe["px"].append(dataset[idx, 0])
            dataframe["py"].append(dataset[idx, 1])
            dataframe["pz"].append(dataset[idx, 2])
            dataframe["vertex_x"].append(vx)
            dataframe["vertex_y"].append(vy)
            dataframe["vertex_z"].append(vz)

    df = pl.DataFrame(data=dataframe)
    df.write_parquet(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert the simulation kinematics HDF5 data to a dataframe"
    )
    parser.add_argument("input", type=Path, help="The simulation HDF5 data")
    parser.add_argument(
        "output", type=Path, help="The output dataframe file path (parquet)"
    )
    args = parser.parse_args()
    convert_kinematics_hdf5_to_polars(args.input, args.output)

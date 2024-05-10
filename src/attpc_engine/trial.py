from attpc_engine.detector.simulator import run_simulation
from attpc_engine.detector.parameters import (
    DetectorParams,
    ElectronicsParams,
    PadParams,
    Config,
)
from attpc_engine.kinematics.reaction import Reaction, Decay
from attpc_engine.kinematics.pipeline import (
    KinematicsPipeline,
    KinematicsTargetMaterial,
    Excitation,
    run_kinematics_pipeline,
)
from attpc_engine import nuclear_map
from spyral_utils.nuclear.target import TargetData, GasTarget

import time

st = time.time()

target = GasTarget(
    TargetData(compound=[[1, 2, 2]], pressure=600, thickness=None), nuclear_map
)

# # Simulate 1 event of 10Be(d,p) with the pipeline
# pipeline = KinematicsPipeline(
#     [
#         Reaction(
#             target=nuclear_map.get_data(1, 2),
#             projectile=nuclear_map.get_data(4, 10),
#             ejectile=nuclear_map.get_data(1, 1),
#         )
#     ],
#     [Excitation(1.78)],
#     93.0,
#     KinematicsTargetMaterial(material=target, min_distance=0.0, max_distance=1.0),
# )
# run_kinematics_pipeline(pipeline, 100, "/Users/zachserikow/Desktop/yup.hdf5")

# Specify simulation parameters
detector = DetectorParams(
    length=1.0,
    efield=60000.0,
    bfield=3.0,
    mpgd_gain=175000,
    gas_target=target,
    diffusion=(0.277, 0.277),
    fano_factor=0.2,
    w_value=34.0,
)

electronics = ElectronicsParams(
    clock_freq=3.125,
    amp_gain=900,
    shaping_time=1000,
    micromegas_edge=66,
    windows_edge=400,
)

pads = PadParams(
    map="/Users/zachserikow/Desktop/LUT.txt",
    map_params=[-280.0, 279.9, 0.1],
    electronics="/Users/zachserikow/Desktop/pad_electronics_legacy.csv",
)

params = Config(detector, electronics, pads)

run_simulation(
    params,
    "/Users/zachserikow/Desktop/yup.hdf5",
    "/Users/zachserikow/Desktop/run_0100.hdf5",
)

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print("Execution time:", elapsed_time, "seconds")

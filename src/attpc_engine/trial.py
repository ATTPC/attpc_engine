from attpc_engine.detector.simulator import run_simulation
from attpc_engine.detector.parameters import (
    Detector_Params,
    Electronics_Params,
    Parameters
    )
from attpc_engine.kinematics.reaction import Reaction, Decay
from attpc_engine.kinematics.pipeline import (
    KinematicsPipeline,
    Excitation,
    run_kinematics_pipeline
    )
from attpc_engine import nuclear_map
from spyral_utils.nuclear.target import TargetData, GasTarget

# # Simulate 1 event of 10Be(d,p) with the pipeline
# pipeline = KinematicsPipeline(
#     [Reaction(
#         target=nuclear_map.get_data(1, 2),
#         projectile=nuclear_map.get_data(4, 10),
#         ejectile=nuclear_map.get_data(1, 1),
#         )
#     ],
#     [Excitation(0.0)#GS, no width
#     ],
#     93.0,
#     )
# run_kinematics_pipeline(pipeline, 1, '/Users/zachserikow/Desktop/yup.hdf5')

# Specify simulation parameters
detector = Detector_Params(
    length = 1.0,
    efield = 60000.0,
    bfield = 3.0,  
    mpgd_gain = 175000,
    gas_target = GasTarget(
        TargetData(
            compound = [[1, 2, 2]],
            pressure = 600,
            thickness = None
            ),
        nuclear_map
        ),
    diffusion = (0.277, 0.277),
    fano_factor = 0.2,
    w_value = 34.0,
    pad_x = '/Users/zachserikow/Desktop/x_coords.csv',
    pad_y = '/Users/zachserikow/Desktop/y_coords.csv'
)
electronics = Electronics_Params(
    clock_freq = 3.125,
    amp_gain = 900,
    shaping_time = 1000,
    micromegas_edge = 66,
    windows_edge = 400
    )
params = Parameters(detector, electronics)
run_simulation(params, '/Users/zachserikow/Desktop/yup.hdf5')
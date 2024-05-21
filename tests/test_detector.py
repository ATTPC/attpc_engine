from attpc_engine.detector import (
    DetectorParams,
    ElectronicsParams,
    PadParams,
    Config,
)
from attpc_engine.detector.simulator import SimEvent
from attpc_engine import nuclear_map

from spyral_utils.nuclear.target import TargetData, GasTarget
import numpy as np

gas = GasTarget(TargetData([(1, 2, 2)], pressure=300.0), nuclear_map)

detector = DetectorParams(
    length=1.0,
    efield=45000.0,
    bfield=2.85,
    mpgd_gain=175000,
    gas_target=gas,
    diffusion=0.277,
    fano_factor=0.2,
    w_value=34.0,
)

electronics = ElectronicsParams(
    clock_freq=6.25,
    amp_gain=900,
    shaping_time=1000,
    micromegas_edge=10,
    windows_edge=560,
    adc_threshold=40.0,
)

pads = PadParams()


# Can we load a default config
def test_config():
    config = Config(detector, electronics, pads)


# Does simulation event work ok
def test_simulation_event():
    # all protons bby
    fake_data = np.array(
        [
            [0.0, 0.0, 0.0, 938.0],
            [0.0, 0.0, 0.0, 938.0],
            [0.0, 0.0, 0.0, 938.0],
            [0.0, 0.0, 0.0, 938.0],
        ]
    )

    proton_numbers = np.array([1, 1, 1, 1])
    mass_numbers = np.array([1, 1, 1, 1])
    vertex = np.array([1.0, 1.0, 1.0])

    event = SimEvent(fake_data, vertex, proton_numbers, mass_numbers)

    assert len(event.nuclei) == 2

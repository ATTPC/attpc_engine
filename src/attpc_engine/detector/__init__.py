"""This is the parent module for all detector effects"""

from attpc_engine.detector.simulator import run_simulation
from attpc_engine.detector.parameters import (
    DetectorParams,
    ElectronicsParams,
    PadParams,
    Config,
)

from attpc_engine.detector.writer import SpyralWriter, SimulationWriter

__all__ = [
    "run_simulation",
    "DetectorParams",
    "ElectronicsParams",
    "PadParams",
    "Config",
    "SpyralWriter",
    "SimulationWriter",
]

"""This is the parent module for the kinematics phase space"""

from .pipeline import (
    KinematicsPipeline,
    run_kinematics_pipeline,
    KinematicsTargetMaterial,
)

from .excitation import (
    ExcitationDistribution,
    ExcitationGaussian,
    ExcitationUniform,
    ExcitationBreitWigner,
)

from .angle import PolarDistribution, PolarUniform, PolarArbitrary

from .reaction import Reaction, Decay

__all__ = [
    "KinematicsPipeline",
    "run_kinematics_pipeline",
    "KinematicsTargetMaterial",
    "ExcitationDistribution",
    "ExcitationGaussian",
    "ExcitationUniform",
    "ExcitationBreitWigner",
    "PolarDistribution",
    "PolarArbitrary",
    "PolarUniform",
    "Reaction",
    "Decay",
]

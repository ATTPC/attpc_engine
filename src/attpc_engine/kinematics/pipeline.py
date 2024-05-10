from .reaction import Reaction, Decay

from spyral_utils.nuclear.target import GasTarget
import numpy as np
import h5py as h5
from random import uniform, normalvariate
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Excitation:
    """Represents the sampling parameters for an excitation

    Attributes
    ----------
    centroid: float
        The state mean/centroid value in MeV
    width: float
        The state FWHM value in MeV

    Methods
    -------
    sigma()
        Calculates the standard deviation from the FWHM assuming a normal distribution
    """

    centroid: float = 0.0
    width: float = 0.0  # FWHM

    def sigma(self) -> float:
        """Calculates the standard deviation from the FWHM assuming a normal distribution

        Returns
        -------
        float
            The standard deviation
        """
        return self.width / 2.355


@dataclass
class KinematicsTargetMaterial:
    """Wrapper around GasTarget and sampling parameters

    Attributes
    ----------
    material: GasTarget
        The target material
    min_distance: float
        The minimum edge for sampling a beam distance
    max_distance: float
        The maximum edge for sampling a beam distance

    """

    material: GasTarget
    min_distance: float
    max_distance: float


class PipelineError(Exception):
    """Pipeline error class"""

    pass


class KinematicsPipeline:
    """The pipeline for generating kinematics data

    The pipeline handles the random sampling using the
    Python standard library random number generator (Mersenne-Twister)

    Parameters
    ----------
    steps: list[Reaction | Decay]
        The steps to be added into the pipeline. The first step should
        always be a Reaction. All subsequent steps should be Decays.
    excitations: list[Excitation]
        The excited state to populate in the Reaction residual or Decay residual_2.
        The number of excitations should be the same as the number of steps, and the order
        of the excitations and steps should be the same.
    beam_energy: float
        The initial (accelerator) beam energy in MeV
    target_material: KinematicsTargetMaterial | None
        Optional target material. If present energy loss will be applied to the beam.

    Attributes
    ----------
    reaction: Reaction
        The reaction in the pipeline. Always the first step in the
        pipeline
    decays: list[Decay]
        The subsequent decay steps in the pipeline.
    beam_energy: float
        The initial (accelerator) beam energy used for the
        reaction step
    target_material: KinematicsTargetMaterial | None
        Optional target material. If present energy loss will
        be applied to the beam.

    """

    def __init__(
        self,
        steps: list[Reaction | Decay],
        excitations: list[Excitation],
        beam_energy: float,
        target_material: KinematicsTargetMaterial | None = None,
    ):
        # Basic validation
        if len(steps) == 0:
            raise PipelineError("Pipeline must have at least one step (a Reaction)!")
        elif len(steps) != len(excitations):
            raise PipelineError(
                f"Pipeline must have the same number of steps (given {len(steps)}) and excitations (given {len(excitations)}!"
            )
        elif not isinstance(steps[0], Reaction):
            raise PipelineError("The first element in the pipeline must be a Reaction!")

        self.reaction: Reaction = steps[0]
        self.decays: list[Decay] = []
        self.excitations = excitations

        # Analyze pipeline for errors
        for idx in range(1, len(steps)):
            cur_step = steps[idx]
            if not isinstance(cur_step, Decay):
                raise PipelineError(
                    "All elements in the pipeline after the first element must be Decay!"
                )
            prev_step = steps[idx - 1]
            if isinstance(prev_step, Reaction):
                if (
                    prev_step.residual.isotopic_symbol
                    != cur_step.parent.isotopic_symbol
                ):
                    raise PipelineError(
                        "Broken step in pipeline! Step 0 residual does not match to Step 1 parent!"
                    )
            else:
                if (
                    prev_step.residual_2.isotopic_symbol
                    != cur_step.residual_2.isotopic_symbol
                ):
                    raise PipelineError(
                        f"Broken step in pipeline! Step {idx-1} residual_2 does not match Step {idx} parent!"
                    )
            self.decays.append(cur_step)

        returned_nuclei = 4 + (len(steps) - 1) * 2
        # storage for calculation result
        self.result = np.empty((returned_nuclei, 4), dtype=float)
        self.beam_energy = beam_energy
        self.target_material = target_material

    def __str__(self) -> str:
        """Return the full reaction chain as a string

        Returns
        -------
        str
            The chain as a reaction equation
        """
        chain = f"{self.reaction}"
        for decay in self.decays:
            chain += f", {str(decay)}"
        return chain

    def run(self) -> tuple[float, np.ndarray]:
        """The method simulate an event

        Returns
        -------
        float
            The distance within the gas target that this
            reaction occured. If there is no gas target,
            the distance is 0.0
        numpy.ndarray
            An Nx4 array of the nuclei 4-vectors. Each unique nucleus
            is a row in the array, and each row contains the px, py, pz, E.
        """

        # First step (reaction)

        # Sample
        ejectile_theta_cm = np.arccos(uniform(-1.0, 1.0))
        ejectile_phi_cm = uniform(0.0, np.pi * 2.0)
        projectile_energy = self.beam_energy
        distance = 0.0
        if self.target_material is not None:
            distance = uniform(
                self.target_material.min_distance,
                self.target_material.max_distance,
            )
            projectile_energy = (
                projectile_energy
                - self.target_material.material.get_energy_loss(
                    self.reaction.projectile,
                    projectile_energy,
                    np.array([distance]),
                )
            )
            projectile_energy = projectile_energy[0]  # Convert 1x1 array to float
        resid_ex = normalvariate(
            self.excitations[0].centroid, self.excitations[0].sigma()
        )

        rxn_result = self.reaction.calculate(
            projectile_energy,  # type: ignore
            ejectile_theta_cm,
            ejectile_phi_cm,
            resid_ex,
        )
        self.result[0] = np.array(
            [rxn_result[0].px, rxn_result[0].py, rxn_result[0].pz, rxn_result[0].E]
        )
        self.result[1] = np.array(
            [rxn_result[1].px, rxn_result[1].py, rxn_result[1].pz, rxn_result[1].E]
        )
        self.result[2] = np.array(
            [rxn_result[2].px, rxn_result[2].py, rxn_result[2].pz, rxn_result[2].E]
        )
        self.result[3] = np.array(
            [rxn_result[3].px, rxn_result[3].py, rxn_result[3].pz, rxn_result[3].E]
        )

        # Do all the decay steps
        prev_resid = rxn_result[3]
        for idx, decay in enumerate(self.decays):
            resid_1_theta_cm = np.arccos(uniform(-1.0, 1.0))
            resid_1_phi_cm = uniform(0.0, np.pi * 2.0)
            resid_2_ex = normalvariate(
                self.excitations[idx + 1].centroid, self.excitations[idx + 1].sigma()
            )
            decay_result = decay.calculate(
                prev_resid, resid_1_theta_cm, resid_1_phi_cm, resid_2_ex
            )
            result_pos = idx * 2 + 4
            self.result[result_pos] = np.array(
                [
                    decay_result[1].px,
                    decay_result[1].py,
                    decay_result[1].pz,
                    decay_result[1].E,
                ]
            )
            self.result[result_pos + 1] = np.array(
                [
                    decay_result[2].px,
                    decay_result[2].py,
                    decay_result[2].pz,
                    decay_result[2].E,
                ]
            )
            prev_resid = decay_result[2]
        return (distance, self.result)

    def get_proton_numbers(self) -> np.ndarray:
        """Get the array of proton numbers

        Returns
        -------
        numpy.ndarray
            An N dimensional array of proton numbers (integers)
        """
        z = np.empty(len(self.result), dtype=int)
        z[0] = self.reaction.target.Z
        z[1] = self.reaction.projectile.Z
        z[2] = self.reaction.ejectile.Z
        z[3] = self.reaction.residual.Z
        for idx, decay in enumerate(self.decays):
            offset = idx + 4
            z[offset] = decay.residual_1.Z
            z[offset + 1] = decay.residual_2.Z
        return z

    def get_mass_numbers(self) -> np.ndarray:
        """Get the array of mass numbers

        Returns
        -------
        numpy.ndarray
            An N dimensional array of mass numbers (integers)
        """
        a = np.empty(len(self.result), dtype=int)
        a[0] = self.reaction.target.A
        a[1] = self.reaction.projectile.A
        a[2] = self.reaction.ejectile.A
        a[3] = self.reaction.residual.A
        for idx, decay in enumerate(self.decays):
            offset = idx + 4
            a[offset] = decay.residual_1.A
            a[offset + 1] = decay.residual_2.A
        return a


def run_kinematics_pipeline(
    pipeline: KinematicsPipeline, n_events: int, output_path: Path
) -> None:
    """Run a pipeline for n events and write the result to disk

    Parameters
    ----------
    pipeline: KinematicsPipeline
        The pipeline to be run
    n_events: int
        The number of events to sample
    output_path: Path
        The path to which the data should be written (HDF5 format)
    """

    output_file = h5.File(output_path, "w")

    data_group = output_file.create_group("data")
    data_group.attrs["n_events"] = n_events
    data_group.attrs["proton_numbers"] = pipeline.get_proton_numbers()
    data_group.attrs["mass_numbers"] = pipeline.get_mass_numbers()

    for event in range(0, n_events):
        distance, result = pipeline.run()
        data = data_group.create_dataset(f"event_{event}", data=result)  # type: ignore
        data.attrs["distance"] = distance
    output_file.close()

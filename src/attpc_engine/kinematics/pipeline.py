from .reaction import Reaction, Decay
from .excitation import ExciationDistribution

from spyral_utils.nuclear.target import GasTarget
import numpy as np
import h5py as h5
from dataclasses import dataclass
from pathlib import Path
from numpy.random import default_rng
from tqdm import trange


@dataclass
class KinematicsTargetMaterial:
    """Wrapper around GasTarget and sampling parameters

    Attributes
    ----------
    material: GasTarget
        The target material
    z_range: tuple[float, float]
        The range of reaction verticies in the material. First value
        is the minimum, second value is the maximum, both in units of meters.
        The z_range is also used to sample beam energies within the target volume.
    rho_sigma: float
        The standard deviation of a normal distribution centered at 0.0 used to sample
        the reaction vertex &rho; in cylindrical coordinates. The distribution in
        cylindrical &theta; is assumed to be uniform.

    """

    material: GasTarget
    z_range: tuple[float, float]
    rho_sigma: float


class PipelineError(Exception):
    """Pipeline error class"""

    pass


class KinematicsPipeline:
    """The pipeline for generating kinematics data

    The pipeline handles the random sampling using a numpy Generator

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
        excitations: list[ExciationDistribution],
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
        self.rng = default_rng()

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

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        """The method simulate an event

        Returns
        -------
        numpy.ndarray
            The reaction vertex as a 3-vector. The array is
            [x,y,z], with each element in meters.
        numpy.ndarray
            An Nx4 array of the nuclei 4-vectors. Each unique nucleus
            is a row in the array, and each row contains the px, py, pz, E.
        """

        # First step (reaction)

        # Sample
        ejectile_theta_cm = np.arccos(self.rng.uniform(-1.0, 1.0))
        ejectile_phi_cm = self.rng.uniform(0.0, np.pi * 2.0)
        projectile_energy = self.beam_energy
        vertex = np.zeros(3)
        if self.target_material is not None:
            rho = np.abs(
                self.rng.normal(0.0, self.target_material.rho_sigma)
            )  # only need positive half of the distribution
            theta = self.rng.uniform(0.0, 2.0 * np.pi)
            vertex[0] = rho * np.cos(theta)
            vertex[1] = rho * np.sin(theta)
            vertex[2] = self.rng.uniform(
                self.target_material.z_range[0],
                self.target_material.z_range[1],
            )
            projectile_energy = (
                projectile_energy
                - self.target_material.material.get_energy_loss(
                    self.reaction.projectile,
                    projectile_energy,
                    vertex[2:],
                )
            )
            projectile_energy = projectile_energy[0]  # Convert 1x1 array to float

        # Sample an excitation, truncating the distribution based on
        # energetically allowed values
        allowed = False
        resid_ex: float
        while not allowed:
            resid_ex = self.excitations[0].sample(self.rng)
            allowed = self.reaction.is_excitation_allowed(projectile_energy, resid_ex)

        # Calculate

        # Primary reaction
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
            # Sample angles
            resid_1_theta_cm = np.arccos(self.rng.uniform(-1.0, 1.0))
            resid_1_phi_cm = self.rng.uniform(0.0, np.pi * 2.0)
            # sample an excitation, truncating the distribution based
            # on energetically allowed values
            resid_2_ex: float
            allowed = False
            while not allowed:
                resid_2_ex = self.excitations[idx].sample(self.rng)
                allowed = decay.is_excitation_allowed(prev_resid, resid_2_ex)

            # Calculate
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
        return (vertex, self.result)

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

    print("------- AT-TPC Simulation Engine -------")
    print(f"Sampling kinematics from reaction: {pipeline}")
    print(f"Running for {n_events} samples.")
    print(f"Output will be written to {output_path}.")

    output_file = h5.File(output_path, "w")

    data_group = output_file.create_group("data")
    data_group.attrs["n_events"] = n_events
    data_group.attrs["proton_numbers"] = pipeline.get_proton_numbers()
    data_group.attrs["mass_numbers"] = pipeline.get_mass_numbers()

    print("Start your engine!")
    for event in trange(0, n_events):
        vertex, result = pipeline.run()
        data = data_group.create_dataset(f"event_{event}", data=result)  # type: ignore
        data.attrs["vertex_x"] = vertex[0]
        data.attrs["vertex_y"] = vertex[1]
        data.attrs["vertex_z"] = vertex[2]
    output_file.close()
    print("Done.")
    print("----------------------------------------")

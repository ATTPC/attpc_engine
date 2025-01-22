from .reaction import Reaction, Decay
from .excitation import ExcitationDistribution
from .angle import PolarDistribution

from spyral_utils.nuclear.target import GasTarget
import numpy as np
import h5py as h5
from dataclasses import dataclass
from pathlib import Path
from numpy.random import default_rng
from tqdm import trange

CHUNK_SIZE: int = 1_000_000


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
        cylindrical &theta; is assumed to be uniform. In units of meters.
    """

    material: GasTarget
    z_range: tuple[float, float]
    rho_sigma: float


@dataclass
class Sample:
    """Complete set of sampled data for a pipeline

    Attributes
    ----------
    beam_energy: float
        Beam (projectile) energy in MeV
    reaction_excitation: float
        Exctation energy for reaction residual in MeV
    reaction_theta: float
        The reaction polar angle in radians
    reaction_phi: float
        The reaction azimuthal angle in radians
    vertex: numpy.ndarray
        The reaction vertex position in meters [x,y,z]
    decay_excitations: list[float]
        List of excitation energy for each decay residual in MeV
    decay_thetas: list[float]
        List of polar angle for each decay residual in radians
    decay_phis: list[float]
        List of azimuthal angle for each decay residual in radians
    """

    beam_energy: float
    reaction_excitation: float
    reaction_theta: float
    reaction_phi: float
    vertex: np.ndarray
    decay_excitations: list[float]
    decay_thetas: list[float]
    decay_phis: list[float]


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
    excitations: list[ExcitationDistribution]
        The excited state to populate in the Reaction residual or Decay residual_2.
        The number of excitations should be the same as the number of steps, and the order
        of the excitations and steps should be the same.
    polar_dists: list[PolarDistribution]
        The angular distributions to sample from. Currently we only support uniform distributions
        in cos(polar), however, it is possible that in the future we may support more complex distributions!
    beam_energy: float
        The initial (accelerator) beam energy in MeV
    target_material: KinematicsTargetMaterial | None
        Optional target material. If present energy loss will be applied to the beam.
    event_sample_limit: int
        The upper limit on the number of resamples per event. Resamples occur most commonly
        when an excitation is defined that dips below the energy threshold, making a section of the
        distribution invalid. The upper limit defined here stops us from sampling forever in
        the case where mutually exclusive steps in energy were defined. In general, the
        default value should suffice, but for extremely narrow phase spaces this may need to
        be expanded. But generally it would be better to define a clipped excitation
        distribution to draw from.

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
        excitations: list[ExcitationDistribution],
        polar_dists: list[PolarDistribution],
        beam_energy: float,
        target_material: KinematicsTargetMaterial | None = None,
        event_sample_limit: int = 1000,
    ):
        # Basic validation
        if len(steps) == 0:
            raise PipelineError("Pipeline must have at least one step (a Reaction)!")
        elif len(steps) != len(excitations):
            raise PipelineError(
                f"Pipeline must have the same number of steps (given {len(steps)}) and excitations (given {len(excitations)}!"
            )
        elif len(steps) != len(polar_dists):
            raise PipelineError(
                f"Pipeline must have the same number of steps (given {len(steps)}) and polar angle distributions (given {len(polar_dists)})!"
            )
        elif not isinstance(steps[0], Reaction):
            raise PipelineError("The first element in the pipeline must be a Reaction!")

        self.reaction: Reaction = steps[0]
        self.decays: list[Decay] = []
        self.excitations = excitations
        self.polar_dists = polar_dists
        self.rng = default_rng()
        self.event_sample_limit = event_sample_limit

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
                    != cur_step.parent.isotopic_symbol
                ):
                    raise PipelineError(
                        f"Broken step in pipeline! Step {idx - 1} residual_2 does not match Step {idx} parent!"
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

    def check_excitations_allowed(
        self, projectile_energy: float, excitations: list[float]
    ) -> bool:
        """Check if the total reaction system has enough energy to occur

        Parameters
        ----------
        projectile_energy: float
            The beam energy in MeV
        excitations: list[float]
            The excitations for each step in MeV

        Returns
        -------
        bool
            True if allowed, False otherwise
        """
        q_value = (
            (self.reaction.projectile.mass + projectile_energy)
            + self.reaction.target.mass
            - (
                self.reaction.ejectile.mass
                + self.reaction.residual.mass
                + excitations[0]
            )
        )
        for idx, decay in enumerate(self.decays):
            q_value += -1.0 * (
                decay.residual_1.mass + decay.residual_2.mass + excitations[idx + 1]
            )
        return q_value >= 0.0

    def sample(self) -> Sample:
        """Sample the pipeline parameters

        Returns
        -------
        Sample
            The sampled set of pipeline parameters
        """

        projectile_energy = self.beam_energy
        vertex = np.zeros(3)

        # If we have a target, do target stuff
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

        pi2 = np.pi * 2.0

        return Sample(
            beam_energy=projectile_energy,
            reaction_excitation=self.excitations[0].sample(self.rng),
            reaction_theta=self.polar_dists[0].sample(self.rng),
            reaction_phi=self.rng.uniform(0.0, pi2),
            vertex=vertex,
            decay_excitations=[
                self.excitations[idx].sample(self.rng)
                for idx in range(1, len(self.excitations))
            ],
            decay_thetas=[
                self.polar_dists[idx].sample(self.rng)
                for idx in range(1, len(self.excitations))
            ],
            decay_phis=[self.rng.uniform(0.0, pi2) for _ in self.decays],
        )

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        """The method simulate an event

        This method will re-sample from the given distributions until
        a valid event is created OR the single-event-sample-limit is
        reached. This achieves two goals:

        - The number of events sampled is guaranteed to match the requested amount
        - The shape of the sampled distributions is correct even in the case where
        one of the excitations has a region that is kinematically not allowed

        However, it does mean that without an upper limit we could sample forever
        in the case where an excitation is defined with a distribution that is never
        energetically allowed. Thus, we have a single-event-sample-limit that stops
        the calculation if the we sample too many times.

        Returns
        -------
        numpy.ndarray
            The reaction vertex as a 3-vector. The array is
            [x,y,z], with each element in meters.
        numpy.ndarray
            An Nx4 array of the nuclei 4-vectors. Each unique nucleus
            is a row in the array, and each row contains the px, py, pz, E.
        """

        # Count how many samples we've drawn
        sample_count = 0
        while True:
            sample_count += 1
            # If we reach the single-event-sample-limit, raise
            if sample_count > self.event_sample_limit:
                raise PipelineError(
                    f"Reached Sampling Limit ({self.event_sample_limit} samples) for a single event! You may have defined an illegal reaction!"
                )
            sample = self.sample()

            # If we're not energetically allowed, resample
            if not self.reaction.is_excitation_allowed(
                sample.beam_energy, sample.reaction_excitation
            ):
                continue

            rxn_result = self.reaction.calculate(
                sample.beam_energy,
                sample.reaction_theta,
                sample.reaction_phi,
                sample.reaction_excitation,
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

            prev_resid = rxn_result[3]
            allowed = True
            for idx, decay in enumerate(self.decays):
                # If we're not energetically allowed, resample
                if not decay.is_excitation_allowed(
                    prev_resid, sample.decay_excitations[idx]
                ):
                    allowed = False
                    break

                # Calculate
                decay_result = decay.calculate(
                    prev_resid,
                    sample.decay_thetas[idx],
                    sample.decay_phis[idx],
                    sample.decay_excitations[idx],
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

            # allowed by, everyone, we have a good event, leave the while
            if allowed:
                break

        return (sample.vertex, self.result)

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
            offset = idx * 2 + 4
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
            offset = idx * 2 + 4
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
    # ASCII art from https://emojicombos.com/race-flag-ascii-art
    print(
        r"""
            ⠀⠀⠀⠀⠀⠀⠀⣠⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣿⣄⠀⠀⠀⠀⠀⠀⠀
            ⠀⠀⠀⠀⢀⡔⢺⣿⣿⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⣿⣿⡗⠢⡀⠀⠀⠀⠀
            ⠀⠀⠀⠀⠀⢷⣶⡟⠉⢻⣷⡀⠀⠀⠀⠀⠀⠀⠀⣾⡿⠉⢻⣶⡾⠁⠀⠀⠀⠀
            ⣠⣤⣴⠖⠒⠹⣿⣷⣀⣸⣿⣧⠀⠀⠀⠀⠀⠀⣸⣿⣇⣀⣾⣿⠏⠒⠲⣶⣤⣄
            ⢹⣿⣿⡄⠀⣀⡝⠁⠘⣿⣿⣿⡆⠀⠀⠀⠀⢠⣿⣿⣿⠏⠈⢻⣀⠀⢠⣿⣿⡏
            ⠀⢃⠀⠘⣿⣿⣿⣄⣠⡟⠉⢿⣿⡀⠀⠀⠀⣾⡿⠉⢹⣄⣀⣿⣿⣿⠃⠀⡸⠀
            ⠀⠈⣦⣴⣾⠉⠁⠈⣿⣷⠀⠈⣿⣷⠀⠀⣼⣿⠃⠀⣾⣿⠃⠈⠉⣹⣦⣴⠁⠀
            ⠀⠀⠸⣿⣿⡦⠤⠐⠋⠁⠀⠀⠸⣿⣇⢰⣿⡏⠀⠀⠈⠙⠂⠤⢴⣿⣿⠇⠀⠀
            ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣿⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
            ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢿⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        """
    )
    chunk = 0
    chunk_event = 0
    data_group.attrs["chunk_size"] = CHUNK_SIZE
    chunk_group = data_group.create_group("chunk_0")
    chunk_group.attrs["min_event"] = 0
    miniters = int(0.01 * n_events)
    for event in trange(0, n_events, miniters=miniters):
        if chunk_event == CHUNK_SIZE:
            chunk_group.attrs["max_event"] = event - 1
            chunk_event = 0
            chunk += 1
            chunk_group = data_group.create_group(f"chunk_{chunk}")
            chunk_group.attrs["min_event"] = event
        vertex, result = pipeline.run()
        data = chunk_group.create_dataset(f"event_{event}", data=result)  # type: ignore
        data.attrs["vertex_x"] = vertex[0]
        data.attrs["vertex_y"] = vertex[1]
        data.attrs["vertex_z"] = vertex[2]
        chunk_event += 1
    chunk_group.attrs["max_event"] = n_events - 1
    data_group.attrs["n_chunks"] = chunk + 1
    output_file.close()
    print("Done.")
    print("----------------------------------------")

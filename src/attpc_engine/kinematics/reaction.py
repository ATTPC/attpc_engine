from .. import nuclear_map

from spyral_utils.nuclear import NucleusData
import vector
import numpy as np


class Reaction:
    """A two-body kinematics calculation

    Parameters
    ----------
    target: spyral_utils.nuclear.NucleusData
        The target nucleus
    projectile: spyral_utils.nuclear.NucleusData
        The projectile (beam) nucleus
    ejectile: spyral_utils.nuclear.NucleusData
        The ejectile (outgoing) nucleus

    Attributes
    ----------
    target: spyral_utils.nuclear.NucleusData
        The target nucleus
    projectile: spyral_utils.nuclear.NucleusData
        The projectile (beam) nucleus
    ejectile: spyral_utils.nuclear.NucleusData
        The outgoing (angle sampled) nucleus
    residual: spyral_utils.nuclear.NucleusData
        The excited outgoing (angle calculated) nucleus
    reaction_symbol: str
        The reaction equation

    """

    def __init__(
        self,
        target: NucleusData,
        projectile: NucleusData,
        ejectile: NucleusData,
    ):
        self.projectile = projectile
        self.target = target
        self.ejectile = ejectile
        resid_z = self.projectile.Z + self.target.Z - self.ejectile.Z
        resid_a = self.projectile.A + self.target.A - self.ejectile.A
        if resid_z < 0:
            raise ValueError(
                "Reaction calculated a residual Z (proton number) < 0, illegal reaction!"
            )
        if resid_a < 0:
            raise ValueError(
                "Reaction calculated a residual A (mass number) < 0, illegal reaction!"
            )
        self.residual = nuclear_map.get_data(resid_z, resid_a)

        self.reaction_symbol = (
            f"{self.target}({self.projectile},{self.ejectile}){self.residual}"
        )

    def __str__(self) -> str:
        """Get the reaction string

        Returns
        -------
        str
            The reaction equation as a string
        """
        return self.reaction_symbol

    def is_excitation_allowed(
        self, projectile_energy: float, residual_excitation: float
    ) -> bool:
        """Check if a given exctiation, projectile energy is energetically allowed

        Calculate center-of-mass energy of the system and verify that there is enough
        energy to make the outgoing products

        Parameters
        ----------
        projectile_energy: float
            The projectile kinetic energy in MeV
        residual_excitation: float
            The residual excitation energy in MeV

        Returns
        -------
        bool
            True if allowed, False if not
        """
        # System z-momentum in lab
        pz = np.sqrt(
            projectile_energy * (projectile_energy + 2.0 * self.projectile.mass)
        )
        # Lorentz invariant total length (i.e. CoM system energy)
        e_cm = np.sqrt(
            (self.target.mass + projectile_energy + self.projectile.mass) ** 2.0
            - pz**2.0
        )

        outgoing_mass = self.ejectile.mass + self.residual.mass + residual_excitation
        return outgoing_mass < e_cm

    def calculate(
        self,
        projectile_energy: float,
        ejectile_polar: float,
        ejectile_azimuthal: float,
        residual_excitation: float,
    ) -> list[vector.MomentumObject4D]:
        """Calculate the kinematics for a set of phase space parameters

        Parameters
        ----------
        projectile_energy: float
            The projectile kinetic energy in MeV
        ejectile_polar: float
            The ejectile polar angle in the reaction center-of-mass frame
            in radians
        ejectile_azimuthal: float
            The ejectile azimuthal angle in the reaction center-of-mass
            frame in radians
        residual_excitation: float
            The residual excitation energy in MeV

        Returns
        -------
        list[vector.MomentumObject4D]
            The set of calculated momentum 4-vectors in the lab frame
        """
        q_value = (
            self.target.mass
            + self.projectile.mass
            - (self.ejectile.mass + self.residual.mass + residual_excitation)
        )

        e_threshold = (
            -q_value
            * (self.ejectile.mass + self.residual.mass)
            / (self.ejectile.mass + self.residual.mass - self.projectile.mass)
        )

        if projectile_energy < e_threshold:
            raise ValueError("Beam energy below kinematic threshold!")

        # Create target, projectile vectors
        target_vec = vector.obj(px=0.0, py=0.0, pz=0.0, E=self.target.mass)
        proj_vec = vector.obj(
            px=0.0,
            py=0.0,
            pz=np.sqrt(
                projectile_energy * (projectile_energy + 2.0 * self.projectile.mass)
            ),
            E=projectile_energy + self.projectile.mass,
        )
        parent = target_vec + proj_vec
        parent_cm: vector.MomentumObject4D = parent.boostCM_of(parent)  # type: ignore

        # Do the ejectile in the CM where the calculation is simple
        eject_e_cm = (
            self.ejectile.mass**2.0
            - (self.residual.mass + residual_excitation) ** 2.0
            + parent_cm.E**2.0
        ) / (2.0 * parent_cm.E)
        eject_p = np.sqrt(eject_e_cm**2.0 - self.ejectile.mass**2.0)

        eject_vec_cm = vector.obj(
            px=eject_p * np.sin(ejectile_polar) * np.cos(ejectile_azimuthal),
            py=eject_p * np.sin(ejectile_polar) * np.sin(ejectile_azimuthal),
            pz=eject_p * np.cos(ejectile_polar),
            E=eject_e_cm,
        )

        # Boost ejectile back to the lab frame
        eject_vec = eject_vec_cm.boost(parent)  # type: ignore
        # Extract residual
        resid_vec = parent - eject_vec

        return [target_vec, proj_vec, eject_vec, resid_vec]  # type: ignore


class Decay:
    """A two-body decay kinematics calculation

    Parameters
    ----------
    parent: spyral_utils.nuclear.NucleusData
        The parent (decaying) nucleus
    residual_1: spyral_utils.nuclear.NucleusData
        The outgoing (angle sampled) nucleus

    Attributes
    ----------
    parent: spyral_utils.nuclear.NucleusData
        The parent (decaying) nucleus
    residual_1: spyral_utils.nuclear.NucleusData
        The outgoing (angle sampled) nucleus
    residual_2: spyral_utils.nuclear.NucleusData
        The excited outgoing (angle calculated) nucleus
    decay_symbol: str
        The decay equation
    """

    def __init__(self, parent: NucleusData, residual_1: NucleusData):
        self.parent = parent
        self.residual_1 = residual_1
        resid_2_z = self.parent.Z - self.residual_1.Z
        resid_2_a = self.parent.A - self.residual_1.A
        if resid_2_z < 0:
            raise ValueError(
                "Decay calculated a residual2 Z (proton number) < 0, illegal decay!"
            )
        if resid_2_a < 0:
            raise ValueError(
                "Decay calculated a residual2 A (mass number) < 0, illegal decay!"
            )

        self.residual_2 = nuclear_map.get_data(resid_2_z, resid_2_a)
        self.decay_symbol = f"{self.parent}->{self.residual_1}+{self.residual_2}"

    def __str__(self) -> str:
        """Get the decay string

        Returns
        -------
        str
            The decay equation as a string
        """
        return self.decay_symbol

    def is_excitation_allowed(
        self, parent_vector: vector.MomentumObject4D, residual_2_excitation: float
    ) -> bool:
        """Check if a given exctiation, parent vector is energetically allowed

        Parameters
        ----------
        parent_vector: vector.MomentumObject4D
            The momentum 4-vector of the parent (decaying) nucleus
        residual_2_excitation: float
            The residual excitation energy in MeV

        Returns
        -------
        bool
            True if allowed, False if not
        """
        q_value = parent_vector.M - (
            self.residual_1.mass + self.residual_2.mass + residual_2_excitation
        )
        return q_value > 0.0

    def calculate(
        self,
        parent_vector: vector.MomentumObject4D,
        residual_1_polar: float,
        residual_1_azimuthal: float,
        residual_2_excitation: float,
    ) -> list[vector.MomentumObject4D]:
        """Calculate the kinematics for a set of phase space parameters

        Parameters
        ----------
        parent_vector: vector.MomentumObject4D
            The momentum 4-vector of the parent (decaying) nucleus
        residual_1_polar: float
            The polar angle of residual_1 in the parent center-of-mass frame
            in radians
        residual_1_azimuthal: float
            The azimuthal angle of residual_1 in the parent center-of-mass
            frame in radians
        residual_2_excitation: float
            The excitation energy of residual_2 in MeV

        Returns
        -------
        list[vector.MomentumObject4D]
            The set of calculated momentum 4-vectors in the lab frame
        """
        q_value = parent_vector.M - (
            self.residual_1.mass + self.residual_2.mass + residual_2_excitation
        )
        if q_value < 0.0:
            raise ValueError("Parent doesn't have enough energy to decay!")

        parent_vec_cm = parent_vector.boostCM_of(parent_vector)

        e_resid_1_cm = (
            self.residual_1.mass**2.0
            - (self.residual_2.mass + residual_2_excitation) ** 2.0
            + parent_vec_cm.E**2.0
        ) / (2.0 * parent_vec_cm.E)
        p_resid_1_cm = np.sqrt(e_resid_1_cm**2.0 - self.residual_1.mass**2.0)

        resid_1_vec_cm = vector.obj(
            px=p_resid_1_cm * np.sin(residual_1_polar) * np.cos(residual_1_azimuthal),
            py=p_resid_1_cm * np.sin(residual_1_polar) * np.sin(residual_1_azimuthal),
            pz=p_resid_1_cm * np.cos(residual_1_polar),
            E=e_resid_1_cm,
        )

        resid_1_vec = resid_1_vec_cm.boost(parent_vector)
        resid_2_vec = parent_vector - resid_1_vec
        return [parent_vector, resid_1_vec, resid_2_vec]  # type: ignore

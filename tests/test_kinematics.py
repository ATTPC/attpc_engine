from attpc_engine.kinematics import (
    KinematicsPipeline,
    ExcitationGaussian,
    PolarUniform,
    Reaction,
    Decay,
)
from attpc_engine.kinematics.pipeline import PipelineError
from attpc_engine import nuclear_map
import numpy as np


def test_reaction():
    target = nuclear_map.get_data(6, 12)
    projectile = nuclear_map.get_data(1, 2)
    ejectile = nuclear_map.get_data(1, 1)

    rxn = Reaction(target, projectile, ejectile)

    proj_energy = 16.0  # MeV
    eject_polar = np.deg2rad(20.0)  # rad
    eject_azim = 0.0
    resid_ex = 0.0

    lise_val = 18.391  # LISE calculated kinetic energy

    result = rxn.calculate(
        proj_energy, eject_polar, eject_azim, residual_excitation=resid_ex
    )

    print(result)

    eject_ke = result[2].E - result[2].M

    # Try to match within 1 keV
    assert np.round(eject_ke, decimals=3) == lise_val


def test_pipeline():
    # Test if good pipeline works
    try:
        pipeline = KinematicsPipeline(
            [
                Reaction(
                    target=nuclear_map.get_data(5, 10),
                    projectile=nuclear_map.get_data(2, 3),
                    ejectile=nuclear_map.get_data(2, 4),
                ),
                Decay(
                    parent=nuclear_map.get_data(5, 9),
                    residual_1=nuclear_map.get_data(2, 4),
                ),
                Decay(
                    parent=nuclear_map.get_data(3, 5),
                    residual_1=nuclear_map.get_data(2, 4),
                ),
            ],
            [
                ExcitationGaussian(16.8, 0.2),
                ExcitationGaussian(0.0, 1.25),
                ExcitationGaussian(0.0, 0.0),
            ],
            [
                PolarUniform(0.0, np.pi),
                PolarUniform(0.0, np.pi),
                PolarUniform(0.0, np.pi),
            ],
            24.0,
        )
        vertex, result = pipeline.run()
        assert np.all(
            pipeline.get_proton_numbers() == np.array([5, 2, 2, 5, 2, 3, 2, 1])
        )
        assert np.all(
            pipeline.get_mass_numbers() == np.array([10, 3, 4, 9, 4, 5, 4, 1])
        )
        assert len(result) == 8
        assert np.all(vertex == 0.0)
    except PipelineError as e:
        print(f"Failed with error {e}")
        raise AssertionError() from e


def test_pipeline_ex_length():
    # Test if we catch length errors
    try:
        pipeline = KinematicsPipeline(
            [
                Reaction(
                    target=nuclear_map.get_data(5, 10),
                    projectile=nuclear_map.get_data(2, 3),
                    ejectile=nuclear_map.get_data(2, 4),
                ),
                Decay(
                    parent=nuclear_map.get_data(5, 9),
                    residual_1=nuclear_map.get_data(2, 4),
                ),
            ],
            [ExcitationGaussian(16.8, 0.2)],
            [
                PolarUniform(0.0, np.pi),
                PolarUniform(0.0, np.pi),
            ],
            24.0,
        )
        pipeline.run()
    except PipelineError:
        pass
    else:
        print("Failed test of matching Excitations/Steps")
        raise AssertionError()


def test_pipeline_pl_length():
    # Test if we catch length errors
    try:
        pipeline = KinematicsPipeline(
            [
                Reaction(
                    target=nuclear_map.get_data(5, 10),
                    projectile=nuclear_map.get_data(2, 3),
                    ejectile=nuclear_map.get_data(2, 4),
                ),
                Decay(
                    parent=nuclear_map.get_data(5, 9),
                    residual_1=nuclear_map.get_data(2, 4),
                ),
            ],
            [ExcitationGaussian(16.8, 0.2), ExcitationGaussian(0.0, 0.0)],
            [
                PolarUniform(0.0, np.pi),
            ],
            24.0,
        )
        pipeline.run()
    except PipelineError:
        pass
    else:
        print("Failed test of matching Polar Dists./Steps")
        raise AssertionError()


def test_pipeline_chain():
    # Test if we catch chaining errors
    try:
        pipeline = KinematicsPipeline(
            [
                Reaction(
                    target=nuclear_map.get_data(5, 10),
                    projectile=nuclear_map.get_data(2, 3),
                    ejectile=nuclear_map.get_data(2, 4),
                ),
                Decay(
                    parent=nuclear_map.get_data(4, 8),
                    residual_1=nuclear_map.get_data(2, 4),
                ),
            ],
            [ExcitationGaussian(16.8, 0.2), ExcitationGaussian(0.0, 0.0)],
            [
                PolarUniform(0.0, np.pi),
                PolarUniform(0.0, np.pi),
            ],
            24.0,
        )
        pipeline.run()
    except PipelineError:
        pass
    else:
        print("Failed test of matching Steps")
        raise AssertionError()


def test_pipeline_order():
    # Test if we catch out-of-order errors
    try:
        pipeline = KinematicsPipeline(
            [
                Decay(
                    parent=nuclear_map.get_data(5, 9),
                    residual_1=nuclear_map.get_data(2, 4),
                ),
                Reaction(
                    target=nuclear_map.get_data(5, 10),
                    projectile=nuclear_map.get_data(2, 3),
                    ejectile=nuclear_map.get_data(2, 4),
                ),
            ],
            [ExcitationGaussian(16.8, 0.2), ExcitationGaussian(0.0, 0.0)],
            [
                PolarUniform(0.0, np.pi),
                PolarUniform(0.0, np.pi),
            ],
            24.0,
        )
        result = pipeline.run()
        assert np.all(pipeline.get_proton_numbers() == np.array([5, 2, 2, 5, 2, 3]))
        assert np.all(pipeline.get_mass_numbers() == np.array([10, 3, 4, 9, 4, 5]))
        assert len(result) == 6
    except PipelineError as _:
        pass
    else:
        print("Failed out-of-order test")
        raise AssertionError()


def test_pipeline_sample_limit():
    # Test if we catch banned energetics
    # Define an illegal excitation for a given beam energy
    try:
        pipeline = KinematicsPipeline(
            [
                Reaction(
                    target=nuclear_map.get_data(5, 10),
                    projectile=nuclear_map.get_data(2, 3),
                    ejectile=nuclear_map.get_data(2, 4),
                ),
            ],
            [
                ExcitationGaussian(16.8, 0.2),
            ],
            [
                PolarUniform(0.0, np.pi),
                PolarUniform(0.0, np.pi),
            ],
            2.0,
        )
        result = pipeline.run()
        assert np.all(pipeline.get_proton_numbers() == np.array([5, 2, 2, 5, 2, 3]))
        assert np.all(pipeline.get_mass_numbers() == np.array([10, 3, 4, 9, 4, 5]))
        assert len(result) == 6
    except PipelineError as _:
        pass
    else:
        print("Failed out-of-order test")
        raise AssertionError()

from .constants import NUM_TB, E_CHARGE
from .parameters import Config

import numpy as np
from numba import njit


def get_response(config: Config) -> np.ndarray:
    """
    Theoretical response function of GET electronics provided by the
    chip manufacturer. See https://doi.org/10.1016/j.nima.2016.09.018.

    Parameters
    ----------
    config: Config
        The simulation configuration

    Returns
    -------
    np.ndarray
        Returns 1xNUM_TB array of the response function evaluated
        at each time bucket
    """
    response = np.zeros(NUM_TB)
    c1 = 4095 * E_CHARGE / config.elec_params.amp_gain / 1e-15
    tbs = np.linspace(0.0, NUM_TB, NUM_TB)
    c2 = tbs / (config.elec_params.shaping_time * config.elec_params.clock_freq * 0.001)
    response = c1 * np.exp(-3.0 * c2) * (c2**3) * np.sin(c2)
    # Cannot have negative ADC values
    response[response < 0] = 0

    return response


@njit
def apply_response(response: np.ndarray, electrons: float) -> tuple[float, float]:
    """Apply the response function to get a signal amplitude, integral

    Scale the response by the number of electrons. Retrieve the amplitude,
    integral of the response in ADC units. Signals above 11-bit max (4096) are clipped

    Parameters
    ----------
    response: numpy.ndarray
        The response signal
    electrons: float
        The number of electrons to scale by

    Returns
    -------
    tuple[float, float]
        The signal amplitude, integral respectively

    """
    resp_sig = response * electrons
    resp_sig[resp_sig > 4095] = 4095  # Set saturated pads to max ADC
    return (resp_sig.max(), resp_sig.sum())

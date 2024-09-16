import numpy as np
from numba import njit


@njit
def pair(tb: int, pad: int) -> int:
    """Combine two integers into one integer

    Uses Szudzik pairing function to combine Time Bucket and Pad
    into a id.

    Parameters
    ----------
    tb: int
        The time bucket of the data
    pad: int
        The pad id of the data

    Returns
    -------
    int
        The id of this data from the tb and pad

    """
    if tb < 0 or pad < 0:
        return -1
    return tb * tb + tb + pad if tb == max(tb, pad) else pad * pad + tb


@njit
def unpair(id: int) -> tuple[int, int]:
    """Inverse of the pairing function

    This is the inverse pairing function for Szudzik pairing

    Parameters
    ----------
    id: int
        A id generated from the pair function

    Returns
    -------
    tuple[int, int]
        A pair of numbers which was used to generate the original id.
        In our case the first is the time bucket, the second is the pad id.
        Note that the order is important!
    """
    if id < 0:
        return (-1, -1)

    sqrt_id = np.floor(np.sqrt(id))
    if id - sqrt_id**2 < sqrt_id:
        return (id - sqrt_id**2, sqrt_id)
    else:
        return (sqrt_id, id - sqrt_id**2 - sqrt_id)

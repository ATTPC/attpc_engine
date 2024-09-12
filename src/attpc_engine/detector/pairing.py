import numpy as np
from numba import njit


@njit
def pair(tb: int, pad: int) -> int:
    if tb < 0 or pad < 0:
        return -1
    return tb * tb + tb + pad if tb == max(tb, pad) else pad * pad + tb


@njit
def unpair(id: int) -> tuple[int, int]:
    if id < 0:
        return (-1, -1)

    sqrt_id = np.floor(np.sqrt(id))
    if id - sqrt_id**2 < sqrt_id:
        return (id - sqrt_id**2, sqrt_id)
    else:
        return (sqrt_id, id - sqrt_id**2 - sqrt_id)

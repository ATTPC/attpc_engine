import math
import numpy as np

from numba import njit
from .beam_pads import BEAM_PADS
from ..constants import NUM_TB

STEPS = 10


@njit(cache=True)
def make_traces(hardwareid_map, num_pads) -> np.ndarray:
    """
    Makes the full array of traces for each AT-TPC pad with an extra column
    for counting if that pad received a hit. The array is Nx(NUM_TB+6) and each row
    has the signature (CoBo, AsAd, AGET, AGET Channel, Pad, tb 1, tb 2, ...,
    tb NUM_TB, Hit Counter) where N is the number of pads.

    Parameters
    ----------
    hardwareid_map: numba.typed.Dict[int64: np.ndarray]
        Dictionary mapping pad number to hardware ID.
    num_pads: int
        Number of pads on pad plane.

    Returns
    -------
    np.ndarray
        Empty array of all AT-TPC traces with hit counter appended
        to the trace of each pad.
    """
    results = np.zeros((num_pads, NUM_TB + 6), dtype=np.int64)
    for pad in range(num_pads):
        results[pad, :5] = hardwareid_map[pad]

    return results


@njit(cache=True)
def normal_pdf(x: float, mu: float, sigma: float) -> float:
    """
    Equation of PDF corresponding to a 1D normal distribution.

    Parameters
    ----------
    x: float
        Value to evaluate PDF at.
    mu: float
        Mean of distribution.
    sigma: float
        Standard deviation of distribution.

    Returns
    -------
    float
        Value of distribution with the input mean and sigma
        at the input value.
    """
    c1: float = 1 / math.sqrt(2 * np.pi) / sigma
    c2: float = (-1 / 2) * ((x - mu) / sigma) ** 2

    return c1 * math.exp(c2)


@njit(cache=True)
def bivariate_normal_pdf(
    point: tuple[float, float], mu: tuple[float, float], sigma: float
) -> float:
    """
    Equation of PDF corresponding to a 2D normal distribution where sigma is the same
    for both x and y and there is no correlation between the two variables.

    Parameters
    ----------
    point: tuple[float, float]
        Point to evaluate PDF at.
    mu: tuple[float, float]
        Mean of distribution.
    sigma: float
        Standard deviation of distribution. Same for both
        x and y.

    Returns
    -------
    float
        Value of distribution with the input mean and sigma
        at the input point.
    """
    c1: float = 1 / 2 / np.pi / (sigma**2)
    c2: float = (-1 / 2 / sigma**2) * (
        ((point[0] - mu[0]) ** 2) + ((point[1] - mu[1]) ** 2)
    )

    return c1 * math.exp(c2)


@njit(cache=True)
def meshgrid(xarr: np.ndarray, yarr: np.ndarray) -> np.ndarray:
    """
    Creates a rectangular (x, y) grid from the input arrays and
    returns a Nx2 array where each row is a point in the mesh.

    Parameters
    ----------
    xarr: np.ndarray
        1xN array of x values of mesh.
    yarr: np.ndarray
        1xN array of y values of mesh.

    Returns
    -------
    np.ndarray
        Nx2 array of points in mesh where the
        first column is the x-value and the second
        column is the y-value.

    """
    grid = np.empty(shape=(xarr.size * yarr.size, 2), dtype=np.float64)
    idx: int = 0

    for x in xarr:
        for y in yarr:
            grid[idx][0] = x
            grid[idx][1] = y
            idx += 1

    return grid


@njit(cache=True)
def position_to_index(
    map_params: tuple[float, float, float], position: tuple[float, float]
) -> tuple[float, float]:
    """
    Given an input position in (x, y), outputs the index on the pad map
    corresponding to that position.

    Parameters
    ----------
    params: Parameters
        All parameters for simulation.
    position: tuple[float, float]
        Position in (x, y) to find pad map index of.

    Returns
    -------
    tuple[int, int]
        Index of pad map.
    """
    # Coordinates in mm
    x: float = position[0] * 1000.0
    y: float = position[1] * 1000.0

    low_edge: float = map_params[0]
    high_edge: float = map_params[1]
    bin_size: float = map_params[2]

    # Check if position is off pad plane. Return impossible index (for Numba)
    if (abs(math.floor(x)) > high_edge) or (abs(math.floor(y)) > high_edge):
        return (-1, -1)

    x_idx = int((math.floor(x) - low_edge) / bin_size)
    y_idx = int((math.floor(y) - low_edge) / bin_size)

    return (x_idx, y_idx)


@njit(cache=True)
def point_transport(
    pad_map: np.ndarray,
    map_params: tuple[float, float, float],
    time: float,
    center: tuple[float, float],
    electrons: int,
    sigma_l: float,
    traces: np.ndarray,
    beam_pads: np.ndarray,
) -> np.ndarray:
    """
    Transports all electrons created at a point in a simulated nucleus' track
    straight to the pad plane. If the input longitudinal diffusion is not 0, then
    longitudinal diffusion is applied as well.

    Parameters
    ----------
    pad_map: np.ndarray
        LUT of pads.
    map_params: tuple[float, float, float] (m, m, m)
        LUT parameters. First element is the low edge of the
        grid, second element is the high edge, and the third
        element is the size of each pixel in the map.
    time: float
        Time of point being transported.
    center: tuple[float, float]
        (x,y) position of point being transported.
    electrons: int
        Number of electrons made at point being transported.
    sigma_l: float
        Standard deviation of longitudinal diffusion at point
        being transported.
    traces: np.ndarray
        Array of all AT-TPC traces with hit counter appended
        to the trace of each pad.
    beam_pads: np.ndarray
        Pad numbers of pads in beam region.

    Returns
    -------
    np.ndarray
        Array of all AT-TPC traces with hit counter appended
        to the trace of each pad.
    """
    # Find pad number of hit pad, if it exists
    index: tuple[int, int] = position_to_index(map_params, center)
    if index == (-1, -1):
        return traces
    pad: int = pad_map[index[0], index[1]]

    # Ensure electron hits pad plane and hits a non-beam pad
    if pad != -1 and pad not in beam_pads:
        signal = np.zeros(NUM_TB + 1, dtype=np.int64)
        signal[int(time)] = electrons
        signal[NUM_TB] = 1  # Record that this pad was hit

        if sigma_l != 0:
            signal = add_longitudinal(signal, time, electrons, sigma_l)

        traces[pad, 5:] += signal

    return traces


@njit(cache=True)
def transverse_transport(
    pad_map: np.ndarray,
    map_params: tuple[float, float, float],
    time: float,
    center: tuple[float, float],
    electrons: int,
    sigma_t: float,
    sigma_l: float,
    traces: np.ndarray,
    beam_pads: list[int],
) -> np.ndarray:
    """
    Transports all electrons created at a point in a simulated nucleus'
    track to the pad plane with transverse diffusion applied. This is done by
    creating a square mesh roughly centered on the point where the electrons are
    created. The mesh extends from -3 sigma to 3 sigma in both the
    x and y directions. The number of steps in each direction is controlled by
    the STEPS parameter at the top of this file. Note, that because we specify
    only the number of steps this means that the step size varies from point to
    point in a trajectory. Each pixel of the mesh has a number of electrons in it
    calculated from the bivariate normal distribution PDF.

    Parameters
    ----------
    pad_map: np.ndarray
        LUT of pads.
    map_params: tuple[float, float, float] (m, m, m)
        LUT parameters. First element is the low edge of the
        grid, second element is the high edge, and the third
        element is the size of each pixel in the map.
    time: float
        Time of point being transported.
    center: tuple[float, float]
        (x,y) position of point being transported.
    electrons: int
        Number of electrons made at point being transported.
    sigma_t: float
        Standard deviation of transverse diffusion at point
        being transported.
    sigma_l: float
        Standard deviation of longitudinal diffusion at point
        being transported.
    traces: np.ndarray
        Array of all AT-TPC traces with hit counter appended
        to the trace of each pad.
    beam_pads: np.ndarray
        Pad numbers of pads in beam region.

    Returns
    -------
    np.ndarray
        Array of all AT-TPC traces with hit counter appended
        to the trace of each pad.
    """
    mesh_centerx: float = center[0]
    mesh_centery: float = center[1]

    # Create mesh
    xlims = (mesh_centerx - 3 * sigma_t, mesh_centerx + 3 * sigma_t)
    ylims = (mesh_centery - 3 * sigma_t, mesh_centery + 3 * sigma_t)
    xsteps = np.linspace(*xlims, STEPS)
    step_sizex: float = 2 * 3 * sigma_t / (STEPS - 1)
    ysteps = np.linspace(*ylims, STEPS)
    step_sizey: float = 2 * 3 * sigma_t / (STEPS - 1)
    mesh = meshgrid(xsteps, ysteps)

    for pixel in mesh:
        # Find pad number of hit pad, if it exists
        index: tuple[int, int] = position_to_index(map_params, pixel)
        if index == (-1, -1):
            continue
        pad: int = pad_map[index[0], index[1]]

        # Ensure electron hits pad plane and hits a non-beam pad
        if pad != -1 and pad not in beam_pads:
            signal = np.zeros(NUM_TB + 1, dtype=np.int64)
            signal[NUM_TB] = 1  # Record that this pad was hit
            pixel_electrons = int(
                (
                    bivariate_normal_pdf(pixel, center, sigma_t)
                    * (step_sizex * step_sizey)
                    * electrons
                )
            )
            signal[int(time)] = pixel_electrons

            if sigma_l != 0:
                signal = add_longitudinal(signal, time, pixel_electrons, sigma_l)

            traces[pad, 5:] += signal

    return traces


@njit(cache=True)
def add_longitudinal(
    signal: np.ndarray, time: float, electrons: int, sigma_l: float
) -> np.ndarray:
    """
    Applies longitudinal diffusion to electrons after they have been transported
    to the pad plane, hence the diffusion is applied on the time of arrival and not
    in physical space along the z direction. Similar to the transverse_transport function,
    a 1D mesh is created from -3 sigma to 3 sigma in the time bucket dimension. The
    number of steps is controlled by the STEPS parameter at the top of this file.
    Note, that because we specify only the number of steps this means that the step size
    varies from point to point in a trajectory. Each step in the mesh has a number of
    electrons in it calculated from the normal distribution PDF.

    Parameters
    ----------
    signal: np.ndarray
        1x(NUM_TB+1) array that corresponds to the time buckets of one pad.
        An extra column is tacked on the end to record that the pad
        received a hit.
    time: float
        Time of point being transported.
    electrons: int
        Number of electrons made at point being transported.
    sigma_l: float
        Standard deviation of longitudinal diffusion at point being transported.

    Returns
    -------
    np.ndarray
        1x(NUM_TB+1) array that corresponds to the time buckets of one pad with
        longitudinal diffusion applied. An extra column is tacked on the end to
        record that the pad received a hit.
    """
    centertb: float = time

    # Create time buckets to diffuse over
    tblims = (centertb - 3 * sigma_l, centertb + 3 * sigma_l)
    tbsteps = np.linspace(*tblims, STEPS)
    step_sizetb: float = 2 * 3 * sigma_l / (STEPS - 1)

    # Ensure diffusion is within allowed time buckets
    tbsteps = tbsteps[(0 <= tbsteps) & (tbsteps < NUM_TB)]

    # Remove electrons in preparation for diffusion
    signal[int(time)] = 0

    # Do longitudinal diffusion
    for step in tbsteps:
        signal[int(step)] += int(
            normal_pdf(step, centertb, sigma_l) * step_sizetb * electrons
        )

    return signal


@njit(cache=True)
def find_pads_hit(
    pad_map: np.ndarray,
    map_params: tuple[float, float, float],
    time: float,
    center: tuple[float, float],
    electrons: int,
    sigma_t: float,
    sigma_l: float,
    traces: np.ndarray,
    beam_pads: list[int],
) -> np.ndarray:
    """
    Finds the pads hit by transporting the electrons created at a point in
    the nucleus' trajectory to the pad plane and applies transverse and
    longitudinal diffusion, if selected.

    Parameters
    ----------
    pad_map: np.ndarray
        LUT of pads.
    map_params: tuple[float, float, float] (m, m, m)
        LUT parameters. First element is the low edge of the
        grid, second element is the high edge, and the third
        element is the size of each pixel in the map.
    time: float
        Time of point being transported.
    center: tuple[float, float]
        (x,y) position of point being transported.
    electrons: int
        Number of electrons made at point being transported.
    sigma_t: float
        Standard deviation of transverse diffusion at point
        being transported.
    sigma_l: float
        Standard deviation of longitudinal diffusion at point
        being transported.
    traces: np.ndarray
        Array of all AT-TPC traces with hit counter appended
        to the trace of each pad.
    beam_pads: np.ndarray
        Pad numbers of pads in beam region.

    Returns
    -------
    np.ndarray
        Array of all AT-TPC traces with hit counter appended
        to the trace of each pad.
    """
    # At least point transport
    if sigma_t == 0.0:
        traces: np.ndarray = point_transport(
            pad_map,
            map_params,
            time,
            center,
            electrons,
            sigma_l,
            traces,
            beam_pads,
        )

    # At least transverve diffusion transport
    elif sigma_t != 0.0:
        traces: np.ndarray = transverse_transport(
            pad_map,
            map_params,
            time,
            center,
            electrons,
            sigma_t,
            sigma_l,
            traces,
            beam_pads,
        )

    return traces


@njit(cache=True)
def transport_track(
    pad_map: np.ndarray,
    map_params: tuple[float, float, float],
    diffusion: tuple[float, float],
    efield: float,
    dv: float,
    track: np.ndarray,
    electrons: np.ndarray,
    empty_traces: np.ndarray,
    beam_pads: list[int] = np.array(BEAM_PADS),
):
    """
    High-level function that transports each point in a nucleus' trajectory
    to the pad plane, applying diffusion if specified.

    Parameters
    ----------
    pad_map: np.ndarray
        LUT of pads.
    map_params: tuple[float, float, float] (m, m, m)
        LUT parameters. First element is the low edge of the
        grid, second element is the high edge, and the third
        element is the size of each pixel in the map.
    diffusion: tuple[float, float] (V, V)
        Diffusion coefficients of electrons in the target gas. The
        first element is the transverse coefficient and the second
        is the longitudinal coefficient.
    efield: float (V / m)
        Magnitude of the electric field. The electric field is
        assumed to only have one component in the +z direction
        parallel to the incoming beam.
    dv: float (m / time bucket)
        Electron drift velocity.
    track: np.ndarray
        6xN array where each row is a solution to one of the ODEs evaluated at
        the Nth time step.
    electrons: np.ndarray
        1xN array of electrons created each time step (point) of the trajectory.
    traces: np.ndarray
        Array of all AT-TPC traces with hit counter appended
        to the trace of each pad.
    beam_pads: np.ndarray
        Pad numbers of pads in beam region.

    Returns
    -------
    np.ndarray
        Array of all AT-TPC traces with hit counter appended
        to the trace of each pad.
    """
    sigma_t: np.ndarray = np.sqrt(2 * diffusion[0] * dv * track[2] / efield)  # in m
    sigma_l: np.ndarray = np.sqrt(
        2 * diffusion[1] * track[2] / dv / efield
    )  # in time buckets

    traces = empty_traces
    for idx in range(track.shape[1]):
        time = track[2, idx]
        center = (track[0, idx], track[1, idx])
        point_electrons = electrons[idx]
        traces = find_pads_hit(
            pad_map,
            map_params,
            time,
            center,
            point_electrons,
            sigma_t[idx],
            sigma_l[idx],
            traces,
            beam_pads,
        )

    return traces

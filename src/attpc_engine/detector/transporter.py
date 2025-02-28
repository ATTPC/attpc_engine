from .pairing import pair
from .beam_pads import BEAM_PADS_ARRAY
from .typed_dict import NumbaTypedDict

import numpy as np
from numba import njit

STEPS = 10


@njit
def bivariate_normal_pdf(
    point: tuple[float, float], mu: tuple[float, float], sigma: float
) -> float:
    """Sample a bivariate normal pdf

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

    return c1 * np.exp(c2)


@njit
def meshgrid(xarr: np.ndarray, yarr: np.ndarray) -> np.ndarray:
    """JIT-ed implementation of meshgrid

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


@njit
def position_to_index(
    grid_edges: np.ndarray, position: tuple[float, float]
) -> tuple[int, int]:
    """Convert a position to pad map index

    Given an input position in (x, y), outputs the index on the pad map
    corresponding to that position. For information about the format of
    the pad map, see the load_pad_grid method of the Config class in
    the parameters file.

    Parameters
    ----------
    grid_edges: numpy.ndarray
        The edges of the pad grid
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

    low_edge: float = grid_edges[0]
    high_edge: float = grid_edges[1]
    bin_size: float = grid_edges[2]

    # Pad map is exclusive on highest edge
    if np.floor(x) >= high_edge or np.floor(y) >= high_edge:
        return (-1, -1)

    # Pad map is inclusive on lowest edge
    if np.floor(x) < low_edge or np.floor(y) < low_edge:
        return (-1, -1)

    x_idx = int((np.floor(x) - low_edge) / bin_size)
    y_idx = int((np.floor(y) - low_edge) / bin_size)

    return (x_idx, y_idx)


@njit
def point_transport(
    pad_grid: np.ndarray,
    grid_edges: np.ndarray,
    time: float,
    center: tuple[float, float],
    electrons: int,
    points: NumbaTypedDict[int, tuple[int, int]],
    label: int,
):
    """Transport electrons without diffusion

    Transports all electrons created at a point in a simulated nucleus' track
    straight to the pad plane.

    Parameters
    ----------
    pad_grid: numpy.ndarray
        Grid of pad id for a given index, where index is calculated from x-y position
    grid_edges: numpy.ndarray
        Edges of the pad grid in mm, as well as the step size of the grid in mm
        Allows conversion of position to grid index. 3 element array [low_edge, hi_edge, step]
    time: float
        Time of point being transported.
    center: tuple[float, float]
        (x,y) position of point being transported.
    electrons: int
        Number of electrons made at point being transported.
    points: numba.typed.Dict[int, tuple[int, int]]
        A dictionary mapping a unique pad,tb key to the number of electrons, which
        will be filled by this function
    label: int
        The label for all points created in this call
    """
    # Find pad number of hit pad, if it exists
    index_x, index_y = position_to_index(grid_edges, center)
    if index_x == -1 or index_y == -1:
        return
    pad = int(pad_grid[index_x, index_y])

    # Ensure electron hits pad plane and hits a non-beam pad
    if pad != -1 and pad not in BEAM_PADS_ARRAY:
        tb = int(time)  # Convert from absolute time bucket to discretized
        id = pair(tb, pad)
        charge, _ = points.get(id, (0, 0))  # The get returns 0 if the key doesn't exist
        charge += electrons
        points[id] = (charge, label)


@njit
def transverse_transport(
    pad_grid: np.ndarray,
    grid_edges: np.ndarray,
    time: float,
    center: tuple[float, float],
    electrons: int,
    sigma_t: float,
    points: NumbaTypedDict[int, tuple[int, int]],
    label: int,
):
    """Transport electrons with transverse diffusion

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
    pad_grid: numpy.ndarray
        Grid of pad id for a given index, where index is calculated from x-y position
    grid_edges: numpy.ndarray
        Edges of the pad grid in mm, as well as the step size of the grid in mm
        Allows conversion of position to grid index. 3 element array [low_edge, hi_edge, step]
    time: float
        Time of point being transported.
    center: tuple[float, float]
        (x,y) position of point being transported.
    electrons: int
        Number of electrons made at point being transported.
    sigma_t: float
        Standard deviation of transverse diffusion at point
        being transported.
    points: numba.typed.Dict[int, tuple[int, int]]
        A dictionary mapping a unique pad,tb key to the number of electrons and  label,
        which will be filled by this function
    label: int
        The label for all points in this call
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
        index_x, index_y = position_to_index(grid_edges, pixel)
        if index_x == -1 or index_y == -1:
            continue
        pad: int = int(pad_grid[index_x, index_y])

        # Ensure electron hits pad plane and hits a non-beam pad
        if pad != -1 and pad not in BEAM_PADS_ARRAY:
            tb = int(time)
            id = pair(tb, pad)
            pixel_electrons = int(
                (
                    bivariate_normal_pdf(pixel, center, sigma_t)
                    * (step_sizex * step_sizey)
                    * electrons
                )
            )
            charge, _ = points.get(id, (0, 0))
            charge += pixel_electrons
            points[id] = (charge, label)  # The get returns 0 if the key doesn't exist


@njit
def transport_track(
    pad_grid: np.ndarray,
    grid_edges: np.ndarray,
    diffusion: float,
    efield: float,
    dv: float,
    track: np.ndarray,
    electrons: np.ndarray,
    points: NumbaTypedDict[int, tuple[int, int]],
    label: int,
):
    """Transport electrons generated by a trajectory to the AT-TPC pad plane

    High-level function that transports each point in a nucleus' trajectory
    to the pad plane, applying transverse diffusion if specified.

    Parameters
    ----------
    pad_grid: numpy.ndarray
        Grid of pad id for a given index, where index is calculated from x-y position
    grid_edges: numpy.ndarray
        Edges of the pad grid in mm, as well as the step size of the grid in mm
        Allows conversion of position to grid index. 3 element array [low_edge, hi_edge, step]
    diffusion: float
        Transverse iffusion coefficient of electrons in the target gas. Units of V
    efield: float
        Magnitude of the electric field. The electric field is
        assumed to only have one component in the +z direction
        parallel to the incoming beam. Units of V/m
    dv: float
        Electron drift velocity. Units of m/TimeBucket
    track: np.ndarray
        Nx6 array where each row is a solution to one of the ODEs evaluated at
        the Nth time step.
    electrons: np.ndarray
        Length N array of electrons created each time step (point) of the trajectory.
    points: numba.typed.Dict[int, tuple[int, int]]
        A dictionary mapping a unique pad,tb key to the number of electrons and a label,
        which will be filled by this function.
    label: int
        The label for all points generated by this call
    """

    # Each point is a TB/pad combo in the TPC
    for idx, row in enumerate(track):
        time = row[2]
        center = (row[0], row[1])
        point_electrons = electrons[idx]
        sigma_t = np.sqrt(2.0 * diffusion * dv * time / efield)
        if sigma_t == 0.0:
            point_transport(
                pad_grid, grid_edges, time, center, point_electrons, points, label
            )
        # Transverse diffusion transport
        else:
            transverse_transport(
                pad_grid,
                grid_edges,
                time,
                center,
                point_electrons,
                sigma_t,
                points,
                label,
            )

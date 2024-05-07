import math
import numpy as np

from numba import njit, float64, int64
from numba.types import UniTuple, DictType
from numba.typed import Dict
from numba.experimental import jitclass
from .parameters import Detector_Params, Electronics_Params, Pad_Params, Parameters
from .beam_pads import BEAM_PADS as b_pads

NUM_TB: int = 512
STEPS = 10


@njit
def normal_pdf(x: float, mu: float, sigma: float):
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


@njit
def bivariate_normal_pdf(
    point: tuple[float, float], mu: tuple[float, float], sigma: float
):
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


@njit
def meshgrid(xarr: np.ndarray, yarr: np.ndarray):
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


key_type = int64
value_type = int64[:]

transport_spec = [
    ("center", UniTuple(float64, 2)),
    ("electrons", int64),
    ("time", float64),
    ("sigma_t", float64),
    ("sigma_l", float64),
    ("pads_hit", DictType(key_type, value_type)),
]


@jitclass(spec=transport_spec)
class TransportPoint:
    """
    Class to transport electrons generated at a point along the simulated
    trajectory of a nucleus' to the pad plane for detection.

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
    center: tuple[float, float] (m, m)
        (x, y) position of point in trajectory.
    electrons: int
        Number of electrons created at point in trajectory.
    time: float (time bucket)
        Time bucket of electrons created at point in trajectory.
    beam_pads: list[int]
        Pad numbers of pads in beam region.
    sigma_t: float (m)
        Standard deviation of transverse diffusion.
    sigma_l: float (time buckets)
        Standard deviation of transverse diffusion.

    Attributes
    ----------
    sigma_t: float (m)
        Standard deviation of transverse diffusion.
    sigma_l: float (time buckets)
        Standard deviation of transverse diffusion.
    pads_hit: numba.typed.dict[int: np.ndarray]
        Numba typed dictionary of pads hit by transporting the electrons.
        The key is the pad number and the value a 1xNUM_TB array
        of the time buckets with the number of electrons detected
        in each.
    """

    def __init__(
        self,
        pad_map: np.ndarray,
        map_params: tuple[float, float, float],
        diffusion: tuple[float, float],
        efield: float,
        dv: float,
        center: tuple[float, float],
        electrons: int,
        time: float,
        beam_pads: list[int] = b_pads,
    ):
        self.center = center
        self.electrons = electrons
        self.time = time

        self.sigma_t = np.sqrt(2 * diffusion[0] * dv * time / efield)  # in m
        self.sigma_l = np.sqrt(2 * diffusion[1] * time / dv / efield)  # in time buckets

        self.pads_hit = self.find_pads_hit(pad_map, map_params, diffusion, beam_pads)

    def find_pads_hit(
        self,
        pad_map: np.ndarray,
        map_params: tuple[float, float, float],
        diffusion_coef: tuple[float, float],
        beam_pads: list[int],
    ):
        """
        Finds the pads hit by transporting the electrons created
        at the point to the pad plane and applies transverse and
        longitudinal diffusion, if selected.

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
        beam_pads: list[int]
            Pad numbers of pads in beam region.

        Returns
        -------
        pads_hit: numba.typed.dict[int: np.ndarray]
            Numba typed dictionary of pads hit by transporting the electrons.
            The key is the pad number and the value a 1xNUM_TB array
            of the time buckets with the number of electrons detected
            in each.
        """
        # No transverse diffusion
        if diffusion_coef[0] == 0.0:
            pads_hit: dict = self.do_point(pad_map, map_params, beam_pads)

            # Only lateral diffusion
            if diffusion_coef[1] != 0.0:
                pads_hit: dict = self.do_longitudinal(pads_hit)

        # At least transverse diffusion
        elif diffusion_coef[0] != 0.0:
            pads_hit: dict = self.do_transverse(pad_map, map_params, beam_pads)

            # Transverse and lateral diffusion
            if diffusion_coef[1] != 0.0:
                pads_hit: dict = self.do_longitudinal(pads_hit)

        return pads_hit

    def do_point(
        self,
        pad_map: np.ndarray,
        map_params: tuple[float, float, float],
        beam_pads: list[int],
    ):
        """
        Transports all electrons created at a point in a simulated particle's
        track straight to the pad plane, i.e. there is no diffusion.

        Parameters
        ----------
        params: Parameters
            All parameters for simulation.

        Returns
        -------
        pads_hit: dict[int: np.ndarray[int]]
            Dictionary where key is the pad number of the hit pad
            and the value is a 1xNUM_TB array giving the number of electrons
            detected in time bucket.
        """
        pads_hit = Dict.empty(key_type, value_type)

        # Find pad number of hit pad, if it exists
        index: tuple[int, int] = position_to_index(map_params, self.center)
        if index == (-1, -1):
            return pads_hit
        pad: int = pad_map[index[0], index[1]]

        # Ensure electron hits pad plane and hits a non-beam pad
        if pad != -1 and pad not in beam_pads:
            trace = np.zeros(NUM_TB, dtype=np.int64)
            trace[int(self.time)] = self.electrons
            # pads_hit.update({pad: trace})
            pads_hit[pad] = trace

        return pads_hit

    def do_transverse(
        self,
        pad_map: np.ndarray,
        map_params: tuple[float, float, float],
        beam_pads: list[int],
    ):
        """
        Transports all electrons created at a point in a simulated particle's
        track to the pad plane with transverse diffusion applied. This is done by
        creating a square mesh roughly centered on the point where the electrons are
        created. The mesh extends from -3 sigma to 3 sigma in both the
        x and y directions. The size of one pixel of the mesh is controlled
        by PIXEL_SIZE at the top of this file. Each pixel of the mesh has a
        number of electrons in it calculated from the bivariate normal
        distribution PDF.

        Parameters
        ----------
        params: Parameters
            All parameters for simulation.

        Returns
        -------
        pads_hit: dict[int: np.ndarray[int]]
            Dictionary where the key is the pad number of the hit pad
            and the value is a 1xNUM_TB array giving the number of electrons
            detected in each time bucket.
        """
        pads_hit = Dict.empty(key_type, value_type)

        mesh_centerx: float = self.center[0]
        mesh_centery: float = self.center[1]

        # Create mesh
        xlims = (mesh_centerx - 3 * self.sigma_t, mesh_centerx + 3 * self.sigma_t)
        ylims = (mesh_centery - 3 * self.sigma_t, mesh_centery + 3 * self.sigma_t)
        xsteps = np.linspace(*xlims, STEPS)
        step_sizex: float = 2 * 3 * self.sigma_t / (STEPS - 1)
        ysteps = np.linspace(*ylims, STEPS)
        step_sizey: float = 2 * 3 * self.sigma_t / (STEPS - 1)
        mesh = meshgrid(xsteps, ysteps)

        for pixel in mesh:
            # Find pad number of hit pad, if it exists
            index: tuple[int, int] = position_to_index(map_params, pixel)
            if index == (-1, -1):
                continue
            pad: int = pad_map[index[0], index[1]]

            # Ensure electron hits pad plane and hits a non-beam pad
            if pad != -1 and pad not in beam_pads:
                signal: np.ndarray = np.zeros(NUM_TB, dtype=np.int64)
                signal[int(self.time)] = int(
                    (
                        bivariate_normal_pdf(pixel, self.center, self.sigma_t)
                        * (step_sizex * step_sizey)
                        * self.electrons
                    )
                )

                if pad not in pads_hit:
                    pads_hit[pad] = signal

                else:
                    pads_hit[pad] += signal

        return pads_hit

    def do_longitudinal(self, pads_hit: dict[int : np.ndarray]):
        """
        Applies longitudinal diffusion to electrons after they have been transported
        to the pad plane, hence the diffusion is applied on the time of arrival and not
        in physical space along the z direction. Similar to the do_transverse method,
        a 1D mesh is created from -3 sigma to 3 sigma in the time bucket dimension. The
        size of one step in the mesh is controlled by the STEPS parameters at the top
        of this file. Each step in the mesh has a number of electrons in it calculated
        from the normal distribution PDF.

        Parameters
        ----------
        params: Parameters
            All parameters for simulation.
        pads_hit: dict[int: np.ndarray[int]]
            Dictionary where the key is the pad number of the hit pad
            and the value is a 1xNUM_TB array giving the number of electrons
            detected in each time bucket.

        Returns
        -------
        pads_hit: dict[int: np.ndarray(int)]
            Dictionary where key is the pad number of the hit pad
            and the value is a 1xNUM_TB array giving the number of electrons
            detected in time bucket.
        """
        # Don't do longitudinal diffusion if no pads are hit
        if not bool(pads_hit):
            return pads_hit

        centertb: float = self.time

        # Create time buckets to diffuse over
        tblims = (centertb - 3 * self.sigma_l, centertb + 3 * self.sigma_l)
        tbsteps = np.linspace(*tblims, STEPS)
        step_sizetb: float = 2 * 3 * self.sigma_l / (STEPS - 1)

        # Ensure diffusion is within allowed time buckets
        tbsteps = tbsteps[(0 <= tbsteps) & (tbsteps < NUM_TB)]

        for pad, signal in pads_hit.items():
            # Find electrons deposited in time bucket of pad
            electrons: int = signal[int(self.time)]

            # Remove electrons in preparation for diffusion
            signal[int(self.time)] = 0

            # Do longitudinal diffusion
            for step in tbsteps:
                signal[int(step)] += int(
                    normal_pdf(step, centertb, self.sigma_l) * step_sizetb * electrons
                )

        return pads_hit


@njit
def position_to_index(
    map_params: tuple[float, float, float], position: tuple[float, float]
):
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

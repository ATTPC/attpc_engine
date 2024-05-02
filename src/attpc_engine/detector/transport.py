import scipy
import math

from numba import njit
from .parameters import *
from .beam_pads import BEAM_PADS

NUM_TB: int = 512
STEPS = 10


def normal_pdf(x: float,
               mu: float,
               sigma: float):
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
    float:
        Value of distribution with the input mean and sigma
        at the input value.
    """
    c1: float = 1 / math.sqrt(2 * np.pi) / sigma
    c2: float = (-1 / 2) * ((x - mu) / sigma) ** 2

    return (c1 * math.exp(c2))

def bivariate_normal_pdf(point: tuple[float, float],
                         mu: tuple[float, float],
                         sigma: float):
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
    float:
        Value of distribution with the input mean and sigma
        at the input point.
     """
    c1: float = 1 / 2 / np.pi / (sigma ** 2)
    c2: float = (-1 / 2 / sigma ** 2) * (((point[0] - mu[0]) ** 2) +
                                          ((point[1] - mu[1]) ** 2))
    
    return (c1 * math.exp(c2))

class transport_point:
    """
    """
    def __init__(self,
                 params: Parameters,
                 center: tuple[float, float],
                 electrons: int,
                 time: float
    ):
        self.center = center
        self.electrons = electrons
        self.time = time

        dv: float = params.calculate_drift_velocity()
        self.sigma_t = np.sqrt(2 * params.detector.diffusion[0] *
                               dv * time / params.detector.efield)  # in m
        self.sigma_l = np.sqrt(2 * params.detector.diffusion[1] *
                               time / dv / params.detector.efield)  # in time buckets
    
        self.pads = self.find_pads_hit(params)
    
    def find_pads_hit(self,
                      params: Parameters):
        """
        """
        # No transverse diffusion
        if params.detector.diffusion[0] == 0:
            pads_hit: dict = self.do_point(params)

            # Only lateral diffusion
            if params.detector.diffusion[1] != 0:
                pads_hit = self.do_longitudinal(params, pads_hit)
    
        # At least transverse diffusion
        elif params.detector.diffusion[0] != 0:
            pads_hit: dict = self.do_transverse(params)

            # Transverse and lateral diffusion
            if params.detector.diffusion[1] != 0:
                pads_hit = self.do_longitudinal(params, pads_hit)
        
        return pads_hit
    
    def do_point(self,
                 params: Parameters):
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
        pads_hit = {}

        # Find pad number of hit pad, if it exists
        index: tuple[int, int] | None = position_to_index(params,
                                                              self.center)
        if index is None:
            return pads_hit
        pad: int = params.pad_map[index[0], index[1]]

        # Ensure electron hits pad plane and hits a non-beam pad
        if pad != -1 and pad not in BEAM_PADS:
            trace: np.ndarray = np.zeros(NUM_TB)
            trace[math.floor(self.time)] = self.electrons
            pads_hit[pad] = trace

        return pads_hit
    
    def do_transverse(self,
                      params: Parameters):
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
        pads_hit = {}

        mesh_centerx: float = self.center[0]
        mesh_centery: float = self.center[1]

        # Number of pixels along the x and y axes of the mesh
        #steps: int = int(np.ceil((2 * 3 * self.sigma_t) / PIXEL_SIZE)) + 1
        steps: int = STEPS

        # # Diffusion is smaller than a pixel, so treat it as a point
        # if steps < 1:
        #     return self.do_point(params)

        # Create mesh
        xlims = (mesh_centerx - 3 * self.sigma_t, mesh_centerx + 3 * self.sigma_t)
        ylims = (mesh_centery - 3 * self.sigma_t, mesh_centery + 3 * self.sigma_t)
        xsteps, step_sizex = np.linspace(*xlims, STEPS, retstep=True)
        ysteps, step_sizey = np.linspace(*ylims, STEPS, retstep=True)
        xmesh, ymesh = np.meshgrid(xsteps, ysteps)

        for pixel in np.column_stack((xmesh.ravel(), ymesh.ravel())):
            # Find pad number of hit pad, if it exists
            index: tuple[int, int] | None = position_to_index(params,
                                                              pixel)
            if index is None:
                continue
            pad: int = params.pad_map[index[0], index[1]]

            # Ensure electron hits pad plane and hits a non-beam pad
            if pad != -1 and pad not in BEAM_PADS:
                signal: np.ndarray = np.zeros(NUM_TB)
                signal[int(self.time)] = int((bivariate_normal_pdf(pixel,
                                                                   self.center,
                                                                   self.sigma_t) *
                                                (step_sizex * step_sizey) * 
                                                self.electrons))

                if pad not in pads_hit:
                    pads_hit[pad] = signal

                else:
                    pads_hit[pad] += signal

        return pads_hit

    def do_longitudinal(self,
                        params: Parameters,
                        pads_hit: dict[int: np.ndarray]):
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
        tbsteps, step_sizetb = np.linspace(*tblims, STEPS, retstep=True)

        # Ensure diffusion is within allowed time buckets
        tbsteps = tbsteps[(0 <= tbsteps) & (tbsteps < NUM_TB)]

        for pad, signal in pads_hit.items():
            # Find electrons deposited in time bucket of pad
            electrons: int = signal[int(self.time)]

            # Remove electrons in preparation for diffusion
            signal[int(self.time)] = 0

            # Do longitudinal diffusion
            for step in tbsteps:
                signal[int(step)] += int(normal_pdf(step,
                                                    centertb,
                                                    self.sigma_l) *
                                      step_sizetb * electrons)

        return pads_hit

def position_to_index(params: Parameters,
                      position: tuple[float, float]):
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
    # coordinates in mm
    x: float = position[0] * 1000.0
    y: float = position[1] * 1000.0

    low_edge: float = params.pads.map_params[0]
    high_edge: float = params.pads.map_params[1]
    bin_size: float = params.pads.map_params[2]

    # Check if position is off pad plane 
    if (abs(math.floor(x)) > high_edge) or (abs(math.floor(y)) > high_edge):
        return None

    x_idx = int((math.floor(x) - low_edge) / bin_size)
    y_idx = int((math.floor(y) - low_edge) / bin_size)
    
    return (x_idx, y_idx)


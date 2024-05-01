import scipy
import math

from numba import njit
from .parameters import *
from .beam_pads import BEAM_PADS

NUM_TB: int = 512
PIXEL_SIZE: float = 1e-3    # m

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

class diffuse_point:
    """
    """
    def __init__(self,
                 params: Parameters,
                 center: tuple[float, float],
                 electrons: int,
                 time: float
    ):
        dv: float = params.calculate_drift_velocity()

        self.sigma_t = np.sqrt(2 * params.detector.diffusion[0] *
                               dv * time / params.detector.efield)
        self.sigma_l = np.sqrt(2 * params.detector.diffusion[1] *
                               dv * time / params.detector.efield)
        
        self.center = center
        self.electrons = electrons
        self.time = time
        self.pads = self.find_pads_hit(params)
    
    def find_pads_hit(self,
                      params: Parameters):
        """
        """
        # No diffusion
        if params.detector.diffusion == (0, 0):
            pads_hit: dict = self.do_point(params)
    
        # At least transverse diffusion
        elif params.detector.diffusion[0] != 0:
            pads_hit: dict = self.do_transverse(params)

            # Transverse and lateral diffusion
            if params.detector.diffusion[1] != 0:
                5

        # Only lateral diffusion
        elif params.detector.diffusion[1] != 0:
            pads_hit: dict = self.do_point(params)
            5 
        
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
        pads_hit: dict[int: np.ndarray(int)]
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
        pads_hit: dict[int: np.ndarray(int)]
            Dictionary where key is the pad number of the hit pad
            and the value is a 1xNUM_TB array giving the number of electrons
            detected in time bucket.
        """
        pads_hit = {}

        mesh_centerx: float = self.center[0]
        mesh_centery: float = self.center[1]

        # Number of pixels along the x and y axes of the mesh
        steps: float = int(np.ceil((2 * 3 * self.sigma_t) / PIXEL_SIZE)) + 1

        # Diffusion is smaller than a pixel, so treat it as a point
        if steps < 1:
            return self.do_point(params)

        # Create mesh
        xlims = (mesh_centerx - 3 * self.sigma_t, mesh_centerx + 3 * self.sigma_t)
        ylims = (mesh_centery - 3 * self.sigma_t, mesh_centery + 3 * self.sigma_t)
        xmesh, ymesh = np.meshgrid(np.linspace(*xlims, steps), np.linspace(*ylims, steps))

        for pixel in np.column_stack((xmesh.ravel(), ymesh.ravel())):
            # Find pad number of hit pad, if it exists
            index: tuple[int, int] | None = position_to_index(params,
                                                              pixel)
            if index is None:
                continue
            pad: int = params.pad_map[index[0], index[1]]

            # Ensure electron hits pad plane and hits a non-beam pad
            if pad != -1 and pad not in BEAM_PADS:
                trace: np.ndarray = np.zeros(NUM_TB)
                trace[math.floor(self.time)] = (bivariate_normal_pdf(pixel,
                                                                    self.center,
                                                                    self.sigma_t) *
                                                (PIXEL_SIZE ** 2) * self.electrons)

                if pad not in pads_hit:
                    pads_hit[pad] = trace

                else:
                    pads_hit[pad] += trace

        return pads_hit

    def do_longitudinal(self,
                        params: Parameters,
                        pads_hit: dict):
        """
        """

        mesh_centerx: float = self.center[0]
        mesh_centery: float = self.center[1]

        # Number of pixels along the x and y axes of the mesh
        steps: float = int(np.ceil((2 * 3 * self.sigma_l) / PIXEL_SIZE)) + 1

        # Diffusion is smaller than a pixel, so treat it as a point
        if steps < 1:
            return self.do_point(params)

        # Create mesh
        xlims = (mesh_centerx - 3 * self.sigma_t, mesh_centerx + 3 * self.sigma_t)
        ylims = (mesh_centery - 3 * self.sigma_t, mesh_centery + 3 * self.sigma_t)
        xmesh, ymesh = np.meshgrid(np.linspace(*xlims, steps), np.linspace(*ylims, steps))

        for pixel in np.column_stack((xmesh.ravel(), ymesh.ravel())):
            # Find pad number of hit pad, if it exists
            index: tuple[int, int] | None = position_to_index(params,
                                                              pixel)
            if index is None:
                continue
            pad: int = params.pad_map[index[0], index[1]]

            # Ensure electron hits pad plane and hits a non-beam pad
            if pad != -1 and pad not in BEAM_PADS:
                trace: np.ndarray = np.zeros(NUM_TB)
                trace[math.floor(self.time)] = (bivariate_normal_pdf(pixel,
                                                                    self.center,
                                                                    self.sigma_t) *
                                                (PIXEL_SIZE ** 2) * self.electrons)

                if pad not in pads_hit:
                    pads_hit[pad] = trace

                else:
                    pads_hit[pad] += trace

        return pads_hit



def position_to_index(params: Parameters,
                      position: tuple[float, float]):
    """
    #map currently not symmetric, ask daniel
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

    x_idx: int = int((math.floor(x) - low_edge) / bin_size)
    y_idx: int = int((math.floor(y) - low_edge) / bin_size)
    
    return (x_idx, y_idx)


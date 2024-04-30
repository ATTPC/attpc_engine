import shapely
import scipy
import math

from .parameters import *
from .beam_pads import BEAM_PADS
from random import normalvariate

NUM_TB: int = 512
PIXEL_SIZE: float = 1e-3    # m

def gaussian_2d(x:float,
                y:float,
                mu: tuple[float, float],
                sigma: float):
     """
     Equation of PDF corresponding to a 2D gaussian where sigma is the same
     for both x and y and there is no correlation between the two variables.
     """
     c1: float = 1 / 2 / np.pi / (sigma ** 2)
     a1: float = (-1 / 2) * (1 / sigma ** 2) * ((x - mu[0]) ** 2) + ((y - mu[1]) ** 2)
     
     return (c1 * np.exp(a1))

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

    def transverse_boundary(self):
        """
        The transverse diffusion is modeled as a 2D gaussian. This
        function makes a circle where the points are 3 sigma away 
        from the center of the gaussian.

        Parameters
        ----------
        params: Parameters
            All parameters for simulation.
        center_x: float
            Gaussian center x-coordinate.
        center_y: float
            Gaussian center y-coordinate.
        sigma: float
            Standard deviation of gaussian.
            Assumed to be the same in x and y.
        
        Returns
        -------
        shapely.Polygon
            Polygon of boundary of transverse diffusion gaussian.
        """
        # Make circlular boundary
        radius = 3 * self.sigma_t
        theta = np.linspace(0.0, 2.0 * np.pi, 1000)
        array = np.zeros(shape=(len(theta), 2))
        array[:, 0] = self.center[0] + np.cos(theta) * radius
        array[:, 1] = self.center[1] + np.sin(theta) * radius

        # Convert circular boundary to shapely polygon
        boundary = shapely.Polygon(array)

        return boundary
    
    def pads_hit(self,
                 params: Parameters):
        """
        """
        boundary = self.transverse_boundary()
        pads_hit = {key: pad for key, pad in params.pads.items()
                    if boundary.intersects(pad) or boundary.contains(pad)}
        
        return pads_hit
    
    def find_pads_hit(self,
                      params: Parameters):
        """
        """
        results = {}
        trace: np.ndarray = np.zeros(NUM_TB)
        if params.detector.diffusion == (0, 0):
            #No diffusion
            index: tuple[int, int] | None = position_to_index(params,
                                                              self.center)
            if index is None:
                return results
            
            pad: int = params.pads[index[0], index[1]]
            # Ensure electron hits pad plane and hits a non-beam pad
            if pad != -1 and pad not in BEAM_PADS:
                trace[math.floor(self.time)] = self.electrons
                results[pad] = trace
                return results
    
        elif params.detector.diffusion[0] != 0:
            # At least transverse
            5

        elif params.detector.diffusion[1] != 0:
            # only lateral
            5
    
        return 0
    
    def do_transverse(self):
        """
        """
        results = {}

        pixel_centerx: float = self.center[0]
        pixel_centery: float = self.center[1]
        distance: float = 0

        steps: float = int(np.ceil((2 * 3 * self.sigma_t) / PIXEL_SIZE)) + 1
        xlims = (pixel_centerx - 3 * self.sigma_t, pixel_centerx  + 3 * self.sigma_t)
        ylims = (pixel_centery - 3 * self.sigma_t, pixel_centery  + 3 * self.sigma_t)
        xx, yy = np.meshgrid(np.linspace(*xlims, steps), np.linspace(*ylims, steps))
        points = np.stack((xx, yy), axis=-1)
        pdf = scipy.stats.multivariate_normal.pdf(points,
                                                  mean=[pixel_centerx, pixel_centery],
                                                  cov=self.sigma_t**2)
        pdf *= (PIXEL_SIZE ** 2) * self.electrons
        pdf = pdf.astype(int)

        for key, pad in self.pads.items():
            pads_e = [pdf.ravel()[idx] for idx, point in 
                       enumerate(np.column_stack((xx.ravel(), yy.ravel())))
                       if pad.contains(shapely.Point(point[0], point[1]))]
                       #and dist(self.center, point) > 3 * self.sigma_t]
            results[key] = {'electrons': [np.sum(pads_e)], 'time': [self.time]}
        
        return results

    
    def do_longitudinal(self,
                        results: dict[dict]):
        """
        """



    def do_diffusion(self,
                     ):
        """
        """
        #case where we want both lateral and longitudinal
        results_t = self.do_transverse()
        results_l = self.do_longitudinal(results_t)

def position_to_index(params: Parameters,
                      position: tuple[float, float]):
    """
    #map currently not symmetric, ask daniel
    """
    # coordinates in mm
    x: float = position[0] * 1000.0
    y: float = position[1] * 1000.0

    low_edge: float = params.detector.pad_map_parameters[0]
    high_edge: float = params.detector.pad_map_parameters[1]
    bin_size: float = params.detector.pad_map_parameters[2]

    # Check if position is off pad plane 
    if (abs(math.floor(x)) > high_edge) or (abs(math.floor(y)) > high_edge):
        return None

    x_idx: int = int((math.floor(x) - low_edge) / bin_size)
    y_idx: int = int((math.floor(y) - low_edge) / bin_size)
    
    return (x_idx, y_idx)


import shapely
import scipy

from .parameters import *
from random import normalvariate

NUM_TB: float = 512
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
     a1: float = (-1 / 2) * (1 / sigma ** 2) * ((x - mu[0]) ** 2) * ((y - mu[1]) ** 2)
     
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
        self.pads = self.pads_hit(params)

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
        theta = np.linspace(0.0, 2.0 * np.pi, 100000)
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
                    if boundary.intersects(pad)}
        
        return pads_hit

    def do_transverse(self):
        """
        """
        results = {}

        pixel_size: float = 1e-4    # m
        pixel_centerx: float = self.center[0]
        pixel_centery: float = self.center[1]
        distance: float = 0

        # Catch case where diffusion is so small it is 
        # contained in one pixel
        if (pixel_size / 2) > (3 * self.sigma_t):
            pixel = shapely.Point(pixel_centerx, pixel_centery)

            for _, (key, pad) in enumerate(self.pads.items()):
                pixel_electrons = int(np.floor(self.electrons *
                                               gaussian_2d(pixel_centerx, 
                                                           pixel_centery,
                                                           self.center,
                                                           self.sigma_t) *
                                               pixel_size ** 2))
                
                if pad.contains(pixel) and key not in results:
                    results[key] = {'electrons': pixel_electrons,
                                    'timebucket': self.time}
                    
                elif pad.contains(pixel) and key in results:
                    results[key]['electrons'] += pixel_electrons
        
            return results
             
        # Loop only in postive x and positive y quandrant
        while True:
            pixel_centerx += pixel_size / 2
            if (pixel_centerx - self.center[0]) > (3 * self.sigma_t):
                break

            while True:
                pixel_centery += pixel_size / 2
                distance = np.sqrt((pixel_centerx - self.center[0]) ** 2 +
                                   (pixel_centery - self.center[1]) ** 2)
                
                if distance > (3 * self.sigma_t):
                    break

                # Construct array of pixels in four quadrants
                pixels = np.array([[pixel_centerx, pixel_centery],    # +x, +y
                                    [-1 * pixel_centerx, pixel_centery],   # -x, +y
                                    [-1 * pixel_centerx, -1 * pixel_centery],  # -x, -y
                                    [pixel_centerx, -1 * pixel_centery]])  # +x, -y

                for quadrant in pixels:
                    pixel = shapely.Point(quadrant)

                    for _, (key, pad) in enumerate(self.pads.items()):
                        pixel_electrons = int(np.floor(self.electrons *
                                                       gaussian_2d(pixel_centerx, 
                                                                   pixel_centery,
                                                                   self.center,
                                                                   self.sigma_t) *
                                                       pixel_size ** 2))

                        if pad.contains(pixel) and key not in results:
                                results[key] = {'electrons': pixel_electrons,
                                                'timebucket': self.time}
                                
                        elif pad.contains(pixel) and key in results:
                                results[key]['electrons'] += pixel_electrons

        return results
    
    def do_transversenew(self):
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

        # results = {key: {'electrons': , 'timebucket':}] for key, pad in self.pads.items()
        #             if boundary.intersects(pad)}
        # for key, pad in self.pads.items():

        #     results = {key: {'electrons': } for x,y in np.column_stack((xx.ravel(), yy.ravel)) 
        #                if pad.contains(shapely.Point(x,y))}

    
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
       
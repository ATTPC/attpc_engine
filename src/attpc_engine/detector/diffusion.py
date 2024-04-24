import shapely
from .parameters import *

def transverse_diffusion_boundary(params: Parameters,
                                  center_x: float,
                                  center_y: float,
                                  sigma: float
):
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
    radius = 3 * sigma
    theta = np.linspace(0.0, 2.0 * np.pi, 100000)
    array = np.zeros(shape=(len(theta), 2))
    array[:, 0] = center_x + np.cos(theta) * radius
    array[:, 1] = center_y + np.sin(theta) * radius

    # Convert circular boundary to shapely polygon
    boundary = shapely.Polygon(array)

    return boundary

def do_transverse_diffusion(pads_hit: list[int],
                            electrons: int,
                            sigma: float
):
    """
    """

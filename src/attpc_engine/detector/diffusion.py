import shapely

from .parameters import *
from random import normalvariate

NUM_TB: float = 512

def diffusion_boundary(params: Parameters,
                       center: tuple[float, float],
                       sigma_t: float
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
    radius = 3 * sigma_t
    theta = np.linspace(0.0, 2.0 * np.pi, 100000)
    array = np.zeros(shape=(len(theta), 2))
    array[:, 0] = center[0] + np.cos(theta) * radius
    array[:, 1] = center[1] + np.sin(theta) * radius

    # Convert circular boundary to shapely polygon
    boundary = shapely.Polygon(array)

    return boundary

def do_diffusion(results: dict,
                 pads_hit: dict[int: shapely.Polygon],
                 electrons: int,
                 time: float, 
                 sigma_t: float,
                 sigma_l: float
):
    """
    """
    # Store results
    # traces = {'pad_number': [],
    #           'timebuckets': []}
    
    # for count, (key, pad) in enumerate(pads_hit.items()):
    #     intersection = boundary.intersection(pad,grid_size=1e-6) # 1 micron
    #     tb = np.zeros(512)

    #     print(intersection)
    #     if count==0:
    #         break

    pixel_size: float = 10e-5 # 10 micron
    pixel_centerx: float = 0
    pixel_centery: float = 0
    distance: float = 0

    # Loop only in postive x and positive y quandrant
    while True:
        pixel_centerx += pixel_size / 2

        if pixel_centerx > (3 * sigma_t):
            break

        while True:
            pixel_centery += pixel_size / 2
            distance = np.sqrt(pixel_centerx ** 2 + pixel_centery ** 2)
            
            if distance > (3 * sigma_t):
                break

            gaussian_pixels(results,
                            pads_hit,
                            pixel_centerx,
                            pixel_centery,
                            time,
                            electrons,
                            sigma_l)

def gaussian_pixels(results: dict,
                    pads_hit: dict[int: shapely.Polygon],
                    pixel_centerx: float,
                    pixel_centery: float,
                    time: float,
                    electrons: int,
                    sigma_l: float):
    """
    """
    # Construct array of pixels in four quadrants
    pixels = np.ndarray([[pixel_centerx, pixel_centery],    # +x, +y
                         [-1 * pixel_centerx, pixel_centery],   # -x, +y
                         [-1 * pixel_centerx, -1 * pixel_centery],  # -x, -y
                         [pixel_centerx, -1 * pixel_centery]])  # +x, -y

    for quadrant in pixels:
        pixel = shapely.Point(quadrant)

        for _, (key, pad) in enumerate(pads_hit.items()):
            if pad.contains(pixel):
                trace = np.ndarray(NUM_TB)
                time = normalvariate(time, sigma_l)
                if key not in results:
                    results[key] = trace
                if key in results:
                    results[key] += trace
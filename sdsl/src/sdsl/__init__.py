from ._sdsl import __doc__, __version__
from ._sdsl import R2xS1, R2xS1_Voxel, Env_R2_Arrangement, localize_R2, localize_R2_dynamic_naive, post_processing
from ._sdsl import R3xS1, R3xS1_Voxel, R3xS2, Env_R3_PCD, localize_R3_pcd
from ._sdsl import seed, max_min_on_trig_range
from . import loaders
from . import visualization

from typing import List

from .dynamic_obstacles import *
# from .simulation import *

import numpy as np

__all__ = [
    "R2xS1", "R2xS1_Voxel", "Env_R2_Arrangement", "localize_R2", "post_processing",
    "R3xS1", "R3xS1_Voxel", "R3xS2", "Env_R3_PCD", "localize_R3_pcd",
    "seed", "max_min_on_trig_range",
]

def load_poly_file(filename: str) -> Env_R2_Arrangement:
    polygons = []
    curr_polygon = []
    with open(filename, "r") as fp:
        for line in fp.readlines():
            line = line.strip()
            if line == "":
                polygons.append(curr_polygon)
                curr_polygon = []
            else:
                x, y = line.split()
                x, y = float(x), float(y)
                curr_polygon.append([x, y])
    if len(curr_polygon) > 0:
        polygons.append(curr_polygon)

    # Convert to numpy array of segments
    segments = []
    for polygon in polygons:
        for i in range(len(polygon)):
            segments.append([*polygon[i], *polygon[(i+1) % len(polygon)]])
    segments = np.array(segments, dtype=np.double)

    return Env_R2_Arrangement(segments)

def sample_q0(env: Env_R2_Arrangement):
    """
    Sample random valid configuration that is inside the environment
    """
    MAX_TRY = 1000
    cnt = 0
    bb = env.bounding_box()
    while True:
        q0 = bb.sample()
        if env.is_inside(q0):
            return q0
        cnt += 1
        if cnt > MAX_TRY:
            raise RuntimeError("Could not sample valid q0 configuration inside environment.")
    
from .sdsl import __doc__, __version__
from .sdsl import R2xS1, R2xS1_Voxel, Env_R2, localize_R2
from .sdsl import max_min_on_trig_range

import numpy as np

__all__ = [
    "R2xS1", "R2xS1_Voxel", "Env_R2", "localize_R2",
    "max_min_on_trig_range",
]

def load_poly_file(filename: str) -> Env_R2:
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

    return Env_R2(segments)
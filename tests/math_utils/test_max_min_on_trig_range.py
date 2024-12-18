from typing import List

import numpy as np

import sdsl

def fdg(d: float, g: List[float], q: List[float]):
    gx, gy, gt = g
    qx, qy, qt = q

    return (
        qx + (gx + d * np.cos(gt)) * np.cos(qt) + (-gy - d * np.sin(gt)) * np.sin(qt),
        qy + (gy + d * np.sin(gt)) * np.cos(qt) + (gx + d * np.cos(gt)) * np.sin(qt),
    )

def test_max_min_on_trig_range():
    d = 1.0
    g = [0.1, 0.2, 0.3]
    gx, gy, gt = g

    xmin = 0; xmax = 0.02
    ymin = 0.3; ymax = 0.35
    tmin = 0.4; tmax = 0.8

    qs = []
    for _ in range(50000):
        qx = np.random.uniform(xmin, xmax)
        qy = np.random.uniform(ymin, ymax)
        qt = np.random.uniform(tmin, tmax)
        qs.append([qx, qy, qt])
    
    # Compute the absolute bounding box
    maxx, minx = sdsl.max_min_on_trig_range(float(gx + d * np.cos(gt)), float(-gy - d * np.sin(gt)), tmin, tmax)
    maxy, miny = sdsl.max_min_on_trig_range(float(gy + d * np.sin(gt)), float(gx + d * np.cos(gt)), tmin, tmax)
    bottom_left = sdsl.R2xS1(minx + xmin, miny + ymin, tmin)
    top_right = sdsl.R2xS1(maxx + xmax, maxy + ymax, tmax)
    voxel = sdsl.R2xS1_Voxel(bottom_left, top_right)

    # Test the the bounding box is actually the bounding box
    for q in qs:
        x, y = fdg(d, g, q)
        assert voxel.contains(sdsl.R2xS1(x, y, q[2]))

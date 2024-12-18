from typing import List

import tqdm
import numpy as np
import matplotlib.pyplot as plt

import sdsl
from sdsl.visualization.viz2d import visualize_2d

MAP_PATH = "resources/maps/lab_lidar.poly"

if __name__ == "__main__":
    env = sdsl.load_poly_file(MAP_PATH)
    bb = env.bounding_box()

    points = []
    for _ in tqdm.tqdm(range(1000)):
        q = bb.sample()
        if not env.is_inside(q):
            continue
        points.append([q.x(), q.y()])
    points = np.array(points)

    visualize_2d(env, points=points)
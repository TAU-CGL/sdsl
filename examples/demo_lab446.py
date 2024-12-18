from typing import List

import numpy as np
import matplotlib.pyplot as plt

import sdsl
from sdsl.visualization.viz2d import visualize_2d

MAP_PATH = "resources/maps/lab_lidar.poly"

if __name__ == "__main__":
    env = sdsl.load_poly_file(MAP_PATH)
    visualize_2d(env)
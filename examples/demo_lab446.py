import time
from typing import List

import tqdm
import numpy as np
import matplotlib.pyplot as plt

import sdsl
from sdsl.visualization.viz2d import visualize_2d, voxel_to_segments

MAP_PATH = "resources/maps/lab_lidar.poly"
ARROW_LEN = 0.3
K = 8; K_ = 6
EPS = 0.0
RECURSION_DEPTH = 6

def get_odometry(k):
    odometry = []
    for i in range(k):
        odometry.append(sdsl.R2xS1(0.0, 0.0, 2.0 * np.pi * i / k))
    return odometry

def get_measurements(env, odometry, q0):
    measurements = []
    for g in odometry:
        measurements.append(env.measure_distance(g * q0))
    return measurements

if __name__ == "__main__":
    env = sdsl.load_poly_file(MAP_PATH)
    bb = env.bounding_box()

    sdsl.seed(100)

    for _ in range(1000):
        q0 = sdsl.sample_q0(env)
        odometry = get_odometry(K)
        measurements = get_measurements(env, odometry, q0)

        start = time.time()
        print("Starting localization...")
        # localization = sdsl.localize_R2(env, odometry, measurements, EPS, RECURSION_DEPTH)
        localization = sdsl.localize_R2_dynamic_naive(env, odometry, measurements, EPS, RECURSION_DEPTH, K_)
        end = time.time()
        print(f"End localization. Took {end - start}[sec]")
        segments = []
        for voxel in localization:
            segments += voxel_to_segments(voxel)
        segments = np.array(segments)


        q0_dir = [[q0.x(), q0.y(), q0.x() + ARROW_LEN * np.cos(q0.r()), q0.y() + ARROW_LEN * np.sin(q0.r())]]
        visualize_2d(env, red_segments=segments, blue_segments=np.array(q0_dir),points=np.array([[q0.x(), q0.y()]]))
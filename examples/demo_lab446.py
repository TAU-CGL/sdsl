import time
from typing import List

import tqdm
import numpy as np
import matplotlib.pyplot as plt

import sdsl
from sdsl.visualization.viz2d import visualize_2d, voxel_to_segments

MAP_PATH = "resources/maps/square_room.poly"
ARROW_LEN = 0.3
K = 16; K_ = 16
EPS = 0.02
RECURSION_DEPTH = 9

def get_odometry(k):
    odometry = []
    for i in range(k):
        odometry.append(sdsl.R2xS1(0.0, 0.0, 2.0 * np.pi * i / k))
    return odometry

def get_measurements(env, odometry, q0, dynamic_obstacles=[]):
    measurements = []
    for g in odometry:
        measurements.append(sdsl.measure_distance(env, g * q0, dynamic_obstacles))
    return measurements

if __name__ == "__main__":
    sdsl.seed(100)
    env = sdsl.load_poly_file(MAP_PATH)

    dynamic_obstacles = [
        sdsl.DynamicObstacle_Disc2D(-0.7, 0.7, 0.1),
    ]

    for _ in range(1000):
        bb = env.bounding_box()
        q0 = sdsl.sample_q0(env)
        odometry = get_odometry(K)
        measurements = get_measurements(env, odometry, q0, dynamic_obstacles)

        start = time.time()
        print("Starting localization...")
        # localization = sdsl.localize_R2(env, odometry, measurements, EPS, RECURSION_DEPTH)
        localization = sdsl.localize_R2_dynamic_naive(env, odometry, measurements, EPS, RECURSION_DEPTH, K_)
        end = time.time()
        print(f"End localization. Took {end - start}[sec]")

        print(sdsl.post_processing(env, odometry, measurements, EPS, localization))

        segments = []
        for voxel in localization:
            segments += voxel_to_segments(voxel)
            r = voxel.bottom_left().r()
            x = voxel.bottom_left().x()
            y = voxel.bottom_left().y()
            segments += [[x,y,x+ARROW_LEN * np.cos(r), y+ARROW_LEN * np.sin(r)]]

        blue_segments = [[q0.x(), q0.y(), q0.x() + ARROW_LEN * np.cos(q0.r()), q0.y() + ARROW_LEN * np.sin(q0.r())]]
        for g, d in zip(odometry, measurements):
            blue_segments += [[q0.x(), q0.y(), q0.x() + d * np.cos(q0.r() + g.r()), q0.y() + d * np.sin(q0.r() + g.r())]]

        visualize_2d(env, dynamic_obstacles=dynamic_obstacles, red_segments=np.array(segments), blue_segments=np.array(blue_segments),points=np.array([[q0.x(), q0.y()]]))

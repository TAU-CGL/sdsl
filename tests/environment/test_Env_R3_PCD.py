from typing import List

import numpy as np

import sdsl

import sdsl.visualization


TEST_FILE = "resources/maps/lab_lidar.poly"
TOLERANCE = 1e-7

def test_Env_R3_PCD_init():
    arr = sdsl.loaders.load_pcd_2d(TEST_FILE)
    env = sdsl.Env_R3_PCD(arr)
    assert np.allclose(
        np.sort(env.get_representation(), axis=0), 
        np.sort(arr, axis=0), 
        atol=TOLERANCE
    )

def test_Env_R3_PCD_measure_distance():
    arr = sdsl.loaders.load_pcd_2d(TEST_FILE)
    env = sdsl.Env_R3_PCD(arr)
    ray = sdsl.R3xS2(0, 0, 0, 1, 0, 0)
    dist = env.measure_distance(ray)
    assert abs(dist - 0.0389390000000005) < TOLERANCE



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    while True:
        arr = sdsl.loaders.load_pcd_2d(TEST_FILE)
        env = sdsl.Env_R3_PCD(arr)
        fig, ax = sdsl.visualization.visualize_pcd_2d(env)

        v = np.random.rand(2) * 2 - 1
        v /= np.linalg.norm(v)
        a, b = v[0], v[1]

        ray = sdsl.R3xS2(0, 0, 0, a, b, 0)
        dist = env.measure_distance(ray)
        print(a, b, dist)
        print(a * dist, b * dist)
        print(a ** 2 + b ** 2)
        if dist > 100:
            dist = 100
        ax.plot([0, dist * a], [0, dist * b], 'r-')

        plt.show()
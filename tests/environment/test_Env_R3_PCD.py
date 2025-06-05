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
    assert abs(dist - 2.53) < TOLERANCE



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    arr = sdsl.loaders.load_pcd_2d(TEST_FILE)
    env = sdsl.Env_R3_PCD(arr)
    fig = sdsl.visualization.visualize_pcd_2d(env)
    # fig.show()
    plt.show()
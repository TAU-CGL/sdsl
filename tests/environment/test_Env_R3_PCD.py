from typing import List

import numpy as np

import sdsl

import sdsl.visualization


TEST_FILE = "resources/maps/lab_lidar.poly"
TOLERANCE = 1e-7

def test_Env_R3_PCD_init_2d():
    arr = sdsl.loaders.load_pcd_2d(TEST_FILE)
    env = sdsl.Env_R3_PCD(arr)
    assert np.allclose(
        np.sort(env.get_representation(), axis=0), 
        np.sort(arr, axis=0), 
        atol=TOLERANCE
    )

def test_Env_R3_PCD_measure_distance_2d():
    arr = sdsl.loaders.load_pcd_2d(TEST_FILE)
    env = sdsl.Env_R3_PCD(arr)
    ray = sdsl.R3xS2(0, 0, 0, 1, 0, 0)
    dist = env.measure_distance(ray)
    assert abs(dist - 3.0389390000000005) < TOLERANCE

    # Cast a lot of random rays and check distances
    num_inf = 0
    for _ in range(10000):
        v = np.random.rand(2) * 2 - 1
        v /= np.linalg.norm(v)
        a, b = v[0], v[1]

        v = np.random.rand(2) * 2 - 1
        v *= 0.2
        c, d = v[0], v[1]

        ray = sdsl.R3xS2(c, d, 0, a, b, 0)
        dist = env.measure_distance(ray)
        if dist > 100:
            num_inf += 1
            continue

        query_pt = np.array([c + a * dist, d + b * dist])
        min_dist = None
        for i in range(arr.shape[0]):
            pt = arr[i, :2]
            tmp = np.linalg.norm(pt - query_pt)
            if min_dist is None or tmp < min_dist:
                min_dist = tmp
        assert min_dist < 0.05
    
    assert num_inf == 0



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    while True:
        arr = sdsl.loaders.load_pcd_2d(TEST_FILE)
        env = sdsl.Env_R3_PCD(arr)
        fig, ax = sdsl.visualization.visualize_pcd_2d(env)

        v = np.random.rand(2) * 2 - 1
        v /= np.linalg.norm(v)
        a, b = v[0], v[1]

        v = np.random.rand(2) * 2 - 1
        v *= 0.2
        c, d = v[0], v[1]

        ray = sdsl.R3xS2(c, d, 0, a, b, 0)
        dist = env.measure_distance(ray)
        if dist > 100:
            dist = 100
        else:
            continue
        ax.plot([c, c + dist * a], [d, d + dist * b], 'r-')

        plt.show()
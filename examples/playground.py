import numpy as np

import sdsl
from sdsl.visualization.viz2d import visualize_2d

if __name__ == "__main__":
    a = np.array(
        [
            [-1, -1,  0,  1],
            [ 0,  1,  1, -1],
            [ 1, -1, -1, -1],
        ], dtype=np.double)
    env = sdsl.Env_R2(a)

    theta = 45 / 180 * np.pi
    q = sdsl.R2xS1(0, 0, theta)
    dist = env.measure_distance(q)
    print(dist)
    red = np.array([[0, 0, dist * np.cos(theta), dist * np.sin(theta)]], dtype=np.double)

    visualize_2d(env, red)
    print(env.get_representation())
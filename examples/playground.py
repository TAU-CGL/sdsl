from typing import List

import numpy as np
import matplotlib.pyplot as plt

import sdsl
from sdsl.visualization.viz2d import visualize_2d

TRIANGLE = np.array(
        [
            [-1, -1,  0,  1],
            [ 0,  1,  1, -1],
            [ 1, -1, -1, -1],
        ], dtype=np.double)

def voxel_to_segments(v: sdsl.R2xS1_Voxel):
    return [
        [v.bottom_left().x(), v.bottom_left().y(), v.top_right().x(), v.bottom_left().y()],
        [v.top_right().x(), v.bottom_left().y(), v.top_right().x(), v.top_right().y()],
        [v.top_right().x(), v.top_right().y(), v.bottom_left().x(), v.top_right().y()],
        [v.bottom_left().x(), v.top_right().y(), v.bottom_left().x(), v.bottom_left().y()],
    ]


def fdg(d: float, g: List[float], q: List[float]):
    gx, gy, gt = g
    qx, qy, qt = q

    return (
        qx + (gx + d * np.cos(gt)) * np.cos(qt) + (-gy - d * np.sin(gt)) * np.sin(qt),
        qy + (gy + d * np.sin(gt)) * np.cos(qt) + (gx + d * np.cos(gt)) * np.sin(qt),
    )

if __name__ == "__main__":
    env = sdsl.Env_R2(TRIANGLE)

    bb = env.bounding_box()
    visualize_2d(env, np.array(voxel_to_segments(bb)))
    
    d = 0.5
    g = [0.0, 0.0, 0.0000001]
    gx, gy, gt = g

    xmin = 0; xmax = 0.02
    ymin = 0.3; ymax = 0.35
    tmin = 0.4; tmax = 0.8

    qs = []
    for _ in range(10000):
        qx = np.random.uniform(xmin, xmax)
        qy = np.random.uniform(ymin, ymax)
        qt = np.random.uniform(tmin, tmax)
        qs.append([qx, qy, qt])
    
    voxel = env.forward(d, sdsl.R2xS1(gx, gy, gt), sdsl.R2xS1_Voxel(
        sdsl.R2xS1(xmin, ymin, tmin),
        sdsl.R2xS1(xmax, ymax, tmax)
    ))
    voxel_segments = voxel_to_segments(voxel)

    # Plot fdg(q):
    fig, ax = plt.subplots()
    for q in qs:
        x, y = fdg(d, g, q)
        ax.plot(x, y, "r.")
        ax.plot(q[0], q[1], "g.")

    for segment in voxel_segments:
        ax.plot([segment[0], segment[2]], [segment[1], segment[3]], "b-")
    
    ax.set_aspect("equal")
    plt.show()


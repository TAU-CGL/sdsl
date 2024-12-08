import numpy as np

import sdsl
from sdsl.visualization.viz2d import visualize_2d

def voxel_to_segments(v: sdsl.R2xS1_Voxel):
    return [
        [v.bottom_left().x(), v.bottom_left().y(), v.top_right().x(), v.bottom_left().y()],
        [v.top_right().x(), v.bottom_left().y(), v.top_right().x(), v.top_right().y()],
        [v.top_right().x(), v.top_right().y(), v.bottom_left().x(), v.top_right().y()],
        [v.bottom_left().x(), v.top_right().y(), v.bottom_left().x(), v.bottom_left().y()],
    ]

if __name__ == "__main__":
    a = np.array(
        [
            [-1, -1,  0,  1],
            [ 0,  1,  1, -1],
            [ 1, -1, -1, -1],
        ], dtype=np.double)
    env = sdsl.Env_R2(a)

    v = sdsl.R2xS1_Voxel(
        sdsl.R2xS1(-1.2, -1.2, 0),
        sdsl.R2xS1(1.2, 1.2, 1)
    )
    red = []
    red += voxel_to_segments(v)
    print(v.split())
    for v_ in v.split():
        print(v_)
        red += voxel_to_segments(v_)
    visualize_2d(env, np.array(red))


    # theta = 10 / 180 * np.pi
    # q = sdsl.R2xS1(0, 0, theta)
    # dist = env.measure_distance(q)
    # print(dist)
    # red = np.array([[0, 0, dist * np.cos(theta), dist * np.sin(theta)]], dtype=np.double)
    # visualize_2d(env, red)

    # v1 = sdsl.R2xS1_Voxel(
    #     sdsl.R2xS1(-0.1, -0.1, 0),
    #     sdsl.R2xS1(0.1, 0.1, 0)
    # )
    # v2 = sdsl.R2xS1_Voxel(
    #     sdsl.R2xS1(0.4, -0.1, 0),
    #     sdsl.R2xS1(0.6, 0.1, 0)
    # )
    # v3 = sdsl.R2xS1_Voxel(
    #     sdsl.R2xS1(0.7, -0.1, 0),
    #     sdsl.R2xS1(0.9, 0.1, 0)
    # )
    # v4 = sdsl.R2xS1_Voxel(
    #     sdsl.R2xS1(-1.2, -1.2, 1),
    #     sdsl.R2xS1(1.2, 1.2, 1)
    # )

    # red = []
    # red += voxel_to_segments(v1)
    # red += voxel_to_segments(v2)
    # red += voxel_to_segments(v3)
    # red += voxel_to_segments(v4)

    # print("v1: ", env.intersects(v1))
    # print("v2: ", env.intersects(v2))
    # print("v3: ", env.intersects(v3))
    # print("v4: ", env.intersects(v4))

    # print(v4.bottom_left() <= v4.top_right())

    # # visualize_2d(env, np.array(red))
    # # print(env.get_representation())
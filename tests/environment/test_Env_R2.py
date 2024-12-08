import numpy as np

import sdsl


TRIANGLE = np.array(
        [
            [-1, -1,  0,  1],
            [ 0,  1,  1, -1],
            [ 1, -1, -1, -1],
        ], dtype=np.double)


def test_Env_R2_init():
    env = sdsl.Env_R2(TRIANGLE)
    assert np.allclose(
        np.sort(env.get_representation(), axis=0), 
        np.sort(TRIANGLE, axis=0), 
        atol=1e-7)
    
def test_Env_R2_measure_distance():
    env = sdsl.Env_R2(TRIANGLE)
    theta = 45 / 180 * np.pi
    q = sdsl.R2xS1(0, 0, theta)
    dist = env.measure_distance(q)
    p = [dist * np.cos(theta), dist* np.sin(theta)]

    # Check that this point is on the line y = 1 - 2x
    print(p)
    print(1 - 2 * p[0])
    print(p[1])
    assert np.allclose(1 - 2 * p[0], p[1], atol=1e-7)

def test_Env_R2_intersects():
    env = sdsl.Env_R2(TRIANGLE)

    v1 = sdsl.R2xS1_Voxel(
        sdsl.R2xS1(-0.1, -0.1, 0),
        sdsl.R2xS1(0.1, 0.1, 0)
    )
    v2 = sdsl.R2xS1_Voxel(
        sdsl.R2xS1(0.4, -0.1, 0),
        sdsl.R2xS1(0.6, 0.1, 0)
    )
    v3 = sdsl.R2xS1_Voxel(
        sdsl.R2xS1(0.7, -0.1, 0),
        sdsl.R2xS1(0.9, 0.1, 0)
    )
    v4 = sdsl.R2xS1_Voxel(
        sdsl.R2xS1(-1.2, -1.2, 1),
        sdsl.R2xS1(1.2, 1.2, 1)
    )

    assert not env.intersects(v1)
    assert env.intersects(v2)
    assert not env.intersects(v3)
    assert env.intersects(v4)
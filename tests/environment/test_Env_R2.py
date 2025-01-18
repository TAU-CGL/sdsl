from typing import List

import numpy as np

import sdsl


TRIANGLE = np.array(
        [
            [-1, -1,  0,  1],
            [ 0,  1,  1, -1],
            [ 1, -1, -1, -1],
        ], dtype=np.double)

TOLERANCE = 1e-7


def test_Env_R2_init():
    env = sdsl.Env_R2(TRIANGLE)
    assert np.allclose(
        np.sort(env.get_representation(), axis=0), 
        np.sort(TRIANGLE, axis=0), 
        atol=TOLERANCE)
    
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
    assert np.allclose(1 - 2 * p[0], p[1], atol=TOLERANCE)

def test_Env_R2_hausdorff_distance():
    env = sdsl.Env_R2(TRIANGLE)
    d1 = env.hausdorff_distance(sdsl.R2xS1(0, 0, 0))
    d2 = env.hausdorff_distance(sdsl.R2xS1(0.1, 0.3, 0))
    d3 = env.hausdorff_distance(sdsl.R2xS1(0.6, 0.5, 0))

    assert np.allclose([d1, d2, d3], [0.4472136, 0.2236068, 0.3130495], atol=TOLERANCE)

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

def test_Env_R2_is_inside():
    env = sdsl.Env_R2(TRIANGLE)
    q1 = sdsl.R2xS1(0, 0, 0)
    q2 = sdsl.R2xS1(2, 2, 0)
    q3 = sdsl.R2xS1(0, 1, 0)

    assert env.is_inside(q1)
    assert not env.is_inside(q2)
    assert env.is_inside(q3)


def fdg(d: float, g: List[float], q: List[float]):
    gx, gy, gt = g
    qx, qy, qt = q

    return (
        qx + (gx + d * np.cos(gt)) * np.cos(qt) + (-gy - d * np.sin(gt)) * np.sin(qt),
        qy + (gy + d * np.sin(gt)) * np.cos(qt) + (gx + d * np.cos(gt)) * np.sin(qt),
    )


def test_Env_R2_forward():
    env = sdsl.Env_R2(TRIANGLE)

    d = 0.5
    g = [0.0, 0.0, 0.0000001]
    gx, gy, gt = g

    xmin = 0; xmax = 0.02
    ymin = 0.3; ymax = 0.35
    tmin = 0.4; tmax = 0.8

    voxel = env.forward(d, sdsl.R2xS1(gx, gy, gt), sdsl.R2xS1_Voxel(
        sdsl.R2xS1(xmin, ymin, tmin),
        sdsl.R2xS1(xmax, ymax, tmax)
    ))

    # Test the the bounding box is actually the bounding box
    qs = []
    for _ in range(10000):
        qx = np.random.uniform(xmin, xmax)
        qy = np.random.uniform(ymin, ymax)
        qt = np.random.uniform(tmin, tmax)
        qs.append([qx, qy, qt])
    for q in qs:
        x, y = fdg(d, g, q)
        assert voxel.contains(sdsl.R2xS1(x, y, q[2]))
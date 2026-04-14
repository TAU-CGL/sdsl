import numpy as np

import sdsl

TRIANGLE = np.array(
        [
            [-1, -1,  0,  1],
            [ 0,  1,  1, -1],
            [ 1, -1, -1, -1],
        ], dtype=np.double)

TOLERANCE = 1e-7


def test_Env_R2_Arrangement_init():
    env = sdsl.Env_R2_Arrangement(TRIANGLE)
    assert np.allclose(
        np.sort(env.get_representation(), axis=0), 
        np.sort(TRIANGLE, axis=0), 
        atol=TOLERANCE)
    
def test_Env_R2_Arrangement_measure_distance():
    env = sdsl.Env_R2_Arrangement(TRIANGLE)
    theta = 45 / 180 * np.pi
    q = sdsl.R3(0, 0, theta)
    dist = env.measure_distance(q)
    p = [dist * np.cos(theta), dist* np.sin(theta)]

    # Check that this point is on the line y = 1 - 2x
    print(p)
    print(1 - 2 * p[0])
    print(p[1])
    assert np.allclose(1 - 2 * p[0], p[1], atol=TOLERANCE)

def test_Env_R2_Arrangement_hausdorff_distance():
    env = sdsl.Env_R2_Arrangement(TRIANGLE)
    d1 = env.hausdorff_distance(sdsl.R3(0, 0, 0))
    d2 = env.hausdorff_distance(sdsl.R3(0.1, 0.3, 0))
    d3 = env.hausdorff_distance(sdsl.R3(0.6, 0.5, 0))

    assert np.allclose([d1, d2, d3], [0.4472136, 0.2236068, 0.3130495], atol=TOLERANCE)

def test_Env_R2_Arrangement_intersects():
    env = sdsl.Env_R2_Arrangement(TRIANGLE)

    v1 = sdsl.Voxel_R3(
        sdsl.R3(-0.1, -0.1, 0),
        sdsl.R3(0.1, 0.1, 0)
    )
    v2 = sdsl.Voxel_R3(
        sdsl.R3(0.4, -0.1, 0),
        sdsl.R3(0.6, 0.1, 0)
    )
    v3 = sdsl.Voxel_R3(
        sdsl.R3(0.7, -0.1, 0),
        sdsl.R3(0.9, 0.1, 0)
    )
    v4 = sdsl.Voxel_R3(
        sdsl.R3(-1.2, -1.2, 1),
        sdsl.R3(1.2, 1.2, 1)
    )

    assert not env.intersects(v1)
    assert env.intersects(v2)
    assert not env.intersects(v3)
    assert env.intersects(v4)

def test_Env_R2_Arrangement_contains():
    env = sdsl.Env_R2_Arrangement(TRIANGLE)
    q1 = sdsl.R3(0, 0, 0)
    q2 = sdsl.R3(2, 2, 0)
    q3 = sdsl.R3(0, 1, 0)

    assert env.contains(q1)
    assert not env.contains(q2)
    assert env.contains(q3)

def test_Env_R2_Arrangement_collision_detection():
    # Triangle with vertices (-1,-1), (0,1), (1,-1).
    #
    # Collision: path from (-2, 0) to (2, 0) crosses the left wall
    # [(-1,-1)→(0,1)] at (-0.5, 0) and the right wall [(0,1)→(1,-1)]
    # at (0.5, 0).
    #
    # No collision: path from (0, -0.5) to (0, -0.1) stays entirely
    # inside the triangle without touching any wall.
    env = sdsl.Env_R2_Arrangement(TRIANGLE)

    q_left  = sdsl.R3(-2,   0,   0)
    q_right = sdsl.R3( 2,   0,   0)
    q_in1   = sdsl.R3( 0,  -0.5, 0)
    q_in2   = sdsl.R3( 0,  -0.1, 0)

    assert     env.collision_detection(q_left, q_right)   # crosses two walls
    assert not env.collision_detection(q_in1,  q_in2)     # wholly inside
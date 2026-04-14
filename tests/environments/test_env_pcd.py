import numpy as np

import sdsl

# Vertical wall of 5 points at x=0, y in [-1, -0.5, 0, 0.5, 1].
# The wall lies entirely in the plane x=0 (z=0 internally).
WALL = np.array(
    [
        [0.0, -1.0],
        [0.0, -0.5],
        [0.0,  0.0],
        [0.0,  0.5],
        [0.0,  1.0],
    ],
    dtype=np.double,
)


def test_Env_2D_PCD_init():
    env = sdsl.Env_2D_PCD(WALL)
    rep = env.get_representation()
    assert rep.shape == (5, 3)
    assert np.allclose(rep[:, 0], 0.0)        # all x-coords are 0
    assert np.allclose(rep[:, 2], 0.0)        # z column is 0 for 2-D cloud
    assert np.allclose(np.sort(rep[:, 1]),
                       np.sort(WALL[:, 1]))   # y-coords match


def test_Env_2D_PCD_collision_detection():
    # Wall of points at x=0.
    #
    # Collision: path from (-1, 0) to (1, 0) travels along y=0 and
    # passes through the wall point (0, 0) — perpendicular distance is 0.
    #
    # No collision: path from (1, 0) to (1, 1) runs parallel to the
    # wall at x=1; the closest wall point is (0, y) with perpendicular
    # distance 1.0, which exceeds the point-cloud proximity threshold.
    env = sdsl.Env_2D_PCD(WALL)

    q_before = sdsl.R3(-1, 0, 0)
    q_after  = sdsl.R3( 1, 0, 0)
    q_side1  = sdsl.R3( 1, 0, 0)
    q_side2  = sdsl.R3( 1, 1, 0)

    assert     env.collision_detection(q_before, q_after)  # path hits wall at (0,0)
    assert not env.collision_detection(q_side1,  q_side2)  # path clear of wall

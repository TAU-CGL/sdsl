import numpy as np

import sdsl

# 5-row × 10-column occupancy grid (uint8).
# Convention (negate=False): 0 → occupied (obstacle), 255 → free.
#
# Layout (row 0 is the top of the image; world y increases upward):
#
#   row 0  [255, 255, 255, 255, 255, 255, 255, 255, 255, 255]  free
#   row 1  [255, 255, 255, 255, 255, 255, 255, 255, 255, 255]  free
#   row 2  [255, 255, 255,   0,   0,   0,   0, 255, 255, 255]  obstacle at cols 3-6
#   row 3  [255, 255, 255, 255, 255, 255, 255, 255, 255, 255]  free
#   row 4  [255, 255, 255, 255, 255, 255, 255, 255, 255, 255]  free
#
# With resolution=1.0 m/px, origin_x=0, origin_y=0:
#   col = floor(x)
#   row = (height-1) - floor(y) = 4 - floor(y)
#
# Obstacle pixels at row=2, cols 3-6 correspond to world region
#   x ∈ [3, 7),  y ∈ [2, 3).

HEIGHT, WIDTH = 5, 10
GRID = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
GRID[2, 3:7] = 0   # horizontal obstacle wall

RESOLUTION = 1.0
ORIGIN_X, ORIGIN_Y = 0.0, 0.0


def _make_env():
    return sdsl.Env_2D_PGM(GRID, RESOLUTION, ORIGIN_X, ORIGIN_Y)


def test_Env_2D_PGM_contains():
    # contains() returns True when the pixel is FREE (navigable), False when
    # it is an obstacle — consistent with Env_R2_Arrangement where contains()
    # means "inside the traversable region".
    env = _make_env()
    assert not env.contains(sdsl.R3(5.5, 2.5, 0))   # obstacle pixel → not navigable
    assert     env.contains(sdsl.R3(1.5, 1.5, 0))   # free pixel    → navigable


def test_Env_2D_PGM_collision_detection():
    # Collision: path from (1.5, 2.5) to (7.5, 2.5) stays at y=2.5
    # (row=2) and crosses the obstacle columns 3-6 in the middle.
    #
    # No collision: path from (1.5, 0.5) to (8.5, 0.5) stays at y=0.5
    # (row=4), which is entirely free.
    env = _make_env()

    q_free_left  = sdsl.R3(1.5, 2.5, 0)
    q_free_right = sdsl.R3(7.5, 2.5, 0)
    q_clear1     = sdsl.R3(1.5, 0.5, 0)
    q_clear2     = sdsl.R3(8.5, 0.5, 0)

    assert     env.collision_detection(q_free_left,  q_free_right)  # path hits wall
    assert not env.collision_detection(q_clear1,     q_clear2)      # path stays free

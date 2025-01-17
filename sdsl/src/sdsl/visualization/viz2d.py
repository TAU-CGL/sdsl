from typing import List

import numpy as np
import matplotlib.pyplot as plt

from sdsl import Env_R2, R2xS1, R2xS1_Voxel, DynamicObstacle, DynamicObstacle_Disc2D

def visualize_2d(env: Env_R2, dynamic_obstacles: List[DynamicObstacle] = None , red_segments: np.array = None, blue_segments: np.array = None, points: np.array = None):
    """
    Visualize a 2D environment.
    """
    segments = env.get_representation()
    _, ax = plt.subplots()
    ax.set_aspect('equal')

    for i in range(segments.shape[0]):
        x1, y1, x2, y2 = segments[i]
        ax.plot([x1, x2], [y1, y2], 'k-')

    if dynamic_obstacles is not None:
        for obs in dynamic_obstacles:
            if isinstance(obs, DynamicObstacle_Disc2D):
                circle = plt.Circle((obs.x, obs.y), obs.radius, color='g', fill=False)
                ax.add_artist(circle)

    if red_segments is not None:
        for i in range(red_segments.shape[0]):
            x1, y1, x2, y2 = red_segments[i]
            ax.plot([x1, x2], [y1, y2], 'r-')
    if blue_segments is not None:
        for i in range(blue_segments.shape[0]):
            x1, y1, x2, y2 = blue_segments[i]
            ax.plot([x1, x2], [y1, y2], 'b-')
    if points is not None:
        for i in range(points.shape[0]):
            x, y = points[i]
            ax.plot(x, y, 'g.')
    
    min_x, max_x = min(min(segments[:,0]), min(segments[:,2])), max(max(segments[:,0]), max(segments[:,2]))
    min_y, max_y = min(min(segments[:,1]), min(segments[:,3])), max(max(segments[:,1]), max(segments[:,3]))
    ax.set_xlim(min_x - 0.5, max_x + 0.5)
    ax.set_ylim(min_y - 0.5, max_y + 0.5)

    plt.title("Sparse Distance Sampling Localization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(False)
    plt.show()

def voxel_to_segments(v: R2xS1_Voxel):
    return [
        [v.bottom_left().x(), v.bottom_left().y(), v.top_right().x(), v.bottom_left().y()],
        [v.top_right().x(), v.bottom_left().y(), v.top_right().x(), v.top_right().y()],
        [v.top_right().x(), v.top_right().y(), v.bottom_left().x(), v.top_right().y()],
        [v.bottom_left().x(), v.top_right().y(), v.bottom_left().x(), v.bottom_left().y()],
    ]
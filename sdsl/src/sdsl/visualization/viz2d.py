import numpy as np
import matplotlib.pyplot as plt

from sdsl import Env_R2, R2xS1

def visualize_2d(env: Env_R2, red_segments: np.array = None, blue_segments: np.array = None):
    """
    Visualize a 2D environment.
    """
    segments = env.get_representation()
    _, ax = plt.subplots()
    ax.set_aspect('equal')

    for i in range(segments.shape[0]):
        x1, y1, x2, y2 = segments[i]
        ax.plot([x1, x2], [y1, y2], 'k-')

    if red_segments is not None:
        for i in range(red_segments.shape[0]):
            x1, y1, x2, y2 = red_segments[i]
            ax.plot([x1, x2], [y1, y2], 'r-')
    if blue_segments is not None:
        for i in range(blue_segments.shape[0]):
            x1, y1, x2, y2 = blue_segments[i]
            ax.plot([x1, x2], [y1, y2], 'b-')
    
    min_x, max_x = min(min(segments[:,0]), min(segments[:,2])), max(max(segments[:,0]), max(segments[:,2]))
    min_y, max_y = min(min(segments[:,1]), min(segments[:,3])), max(max(segments[:,1]), max(segments[:,3]))
    ax.set_xlim(min_x - 0.5, max_x + 0.5)
    ax.set_ylim(min_y - 0.5, max_y + 0.5)

    plt.title("Sparse Distance Sampling Localization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(False)
    plt.show()
import matplotlib.pyplot as plt

from sdsl import Env_R3_PCD

def visualize_pcd_2d(env: Env_R3_PCD):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    points = env.get_representation()
    ax.plot(points[:, 0], points[:, 1], 'o', markersize=2, color='#000000')

    min_x, max_x = min(points[:, 0]), max(points[:, 0])
    min_y, max_y = min(points[:, 1]), max(points[:, 1])
    ax.set_xlim(min_x - 0.5, max_x + 0.5)
    ax.set_ylim(min_y - 0.5, max_y + 0.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(False)

    return fig, ax
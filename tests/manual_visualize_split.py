import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import sdsl


def make_random_2d_box():
    bl = sdsl.Config_2d()
    tr = sdsl.Config_2d()
    x0, y0 = random.uniform(-5, 0), random.uniform(-5, 0)
    x1, y1 = random.uniform(1, 6), random.uniform(1, 6)
    bl[0], bl[1] = min(x0, x1), min(y0, y1)
    tr[0], tr[1] = max(x0, x1), max(y0, y1)
    return bl, tr


def make_random_3d_box():
    bl = sdsl.Config_3d()
    tr = sdsl.Config_3d()
    x0, y0, z0 = random.uniform(-5, 0), random.uniform(-5, 0), random.uniform(-5, 0)
    x1, y1, z1 = random.uniform(1, 6), random.uniform(1, 6), random.uniform(1, 6)
    bl[0], bl[1], bl[2] = min(x0, x1), min(y0, y1), min(z0, z1)
    tr[0], tr[1], tr[2] = max(x0, x1), max(y0, y1), max(z0, z1)
    return bl, tr


def random_mid_2d(bl, tr):
    mid = sdsl.Config_2d()
    mid[0] = random.uniform(bl[0], tr[0])
    mid[1] = random.uniform(bl[1], tr[1])
    return mid


def random_mid_3d(bl, tr):
    mid = sdsl.Config_3d()
    mid[0] = random.uniform(bl[0], tr[0])
    mid[1] = random.uniform(bl[1], tr[1])
    mid[2] = random.uniform(bl[2], tr[2])
    return mid


def plot_2d_split(ax, voxel, midpoint):
    subvoxels = voxel.split(midpoint)
    ax.set_title("2D split")
    ax.set_aspect("equal", adjustable="box")

    for v in subvoxels:
        x0, y0 = v.bottom_left[0], v.bottom_left[1]
        x1, y1 = v.top_right[0], v.top_right[1]
        rect_x = [x0, x1, x1, x0, x0]
        rect_y = [y0, y0, y1, y1, y0]
        ax.plot(rect_x, rect_y, color="tab:blue", linewidth=1.5)

    ax.scatter([midpoint[0]], [midpoint[1]], color="tab:red", s=40, label="midpoint")
    ax.legend(loc="upper right")


def box_edges_3d(bl, tr):
    x0, y0, z0 = bl
    x1, y1, z1 = tr
    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    return [(corners[i], corners[j]) for i, j in edges]


def plot_3d_split(ax, voxel, midpoint):
    subvoxels = voxel.split(midpoint)
    ax.set_title("3D split")

    lines = []
    for v in subvoxels:
        bl = (v.bottom_left[0], v.bottom_left[1], v.bottom_left[2])
        tr = (v.top_right[0], v.top_right[1], v.top_right[2])
        lines.extend(box_edges_3d(bl, tr))

    lc = Line3DCollection(lines, colors="tab:green", linewidths=0.8)
    ax.add_collection3d(lc)
    ax.scatter([midpoint[0]], [midpoint[1]], [midpoint[2]], color="tab:red", s=35)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.autoscale()


def main():
    random.seed(42)

    bl2, tr2 = make_random_2d_box()
    mid2 = random_mid_2d(bl2, tr2)
    voxel2 = sdsl.Voxel_2d(bl2, tr2)

    bl3, tr3 = make_random_3d_box()
    mid3 = random_mid_3d(bl3, tr3)
    voxel3 = sdsl.Voxel_3d(bl3, tr3)

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    plot_2d_split(ax1, voxel2, mid2)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_3d_split(ax2, voxel3, mid3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

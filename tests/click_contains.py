"""
Interactive check of Env_2D_PGM.contains().

Shows the 4 experiment maps side by side. Clicking a point on any map prints the
world coordinates and the result of env.contains() at that location, and marks
the point on the map (green = contains True / free, red = False / obstacle).

Run from the repo root:
    python tests/click_contains.py
"""
import os

import matplotlib.pyplot as plt

import sdsl
from sdsl.loaders.load_pgm_map import load_pgm_map

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

MAP_LIST = [
    "resources/maps/2d/slam/simple_symmetry/square.yaml",
    "resources/maps/2d/slam/simple_symmetry/symmetry_2.yaml",
    "resources/maps/2d/slam/simple_symmetry/plus.yaml",
    "resources/maps/2d/slam/apt_20250913_1449/my_map.yaml",
]


def load_maps():
    maps = []
    for rel in MAP_LIST:
        path = os.path.join(REPO_ROOT, rel)
        pgm = load_pgm_map(path)
        env = sdsl.Env_2D_PGM(
            pgm.grid, pgm.resolution, pgm.origin_x, pgm.origin_y,
            pgm.occupied_thresh, pgm.negate)
        maps.append((rel, pgm, env))
    return maps


def main():
    maps = load_maps()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_to_map = {}

    for ax, (name, pgm, env) in zip(axes.flat, maps):
        extent = [
            pgm.origin_x,
            pgm.origin_x + pgm.width * pgm.resolution,
            pgm.origin_y,
            pgm.origin_y + pgm.height * pgm.resolution,
        ]
        # Image row 0 is the top of the map (max world y).
        ax.imshow(pgm.grid, cmap="gray", origin="upper", extent=extent)
        ax.set_title(os.path.basename(name))
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax_to_map[ax] = (name, env)

    def on_click(event):
        if event.inaxes not in ax_to_map or event.xdata is None:
            return
        name, env = ax_to_map[event.inaxes]
        x, y = event.xdata, event.ydata
        inside = env.contains(sdsl.R3(x, y, 0.0))
        print(f"{os.path.basename(name)}: ({x:.3f}, {y:.3f}) -> contains = {inside}")
        event.inaxes.plot(x, y, "o", color="lime" if inside else "red",
                          markersize=6, markeredgecolor="black")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.suptitle("Click a point to print env.contains() "
                 "(green = True, red = False)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

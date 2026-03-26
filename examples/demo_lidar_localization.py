"""
Demo: load lab_lidar.poly, click inside the map to cast 16 rays and run 3D localization.

The configuration space is 3D: (x, y, theta).
Red dots show the (x, y) projection of each localized voxel center.

Run from the repo root:
    python examples/demo_lidar_localization.py
"""
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import sdsl

MAP_PATH = "resources/maps/2d/checkpoint.poly"
N_RAYS = 16
RECURSION_DEPTH = 8  # 8^(depth-1) voxels with AlwaysTrue predicate
KK_PRIME_RATIO = 0.7
ERROR_BOUND = 0.05


def load_poly_as_segments(path):
    """Parse a .poly file (list of xy vertices) into a (N, 4) segment array [x1,y1,x2,y2]."""
    pts = np.loadtxt(path)
    dst = np.roll(pts, -1, axis=0)      # shift by one: next vertex in loop
    return np.column_stack([pts, dst])   # close the polygon automatically


def cast_rays(env, x, y, n=N_RAYS):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dists = np.array([env.measure_distance(sdsl.R3(x, y, theta)) for theta in angles])
    return angles, dists


def main():
    segments = load_poly_as_segments(MAP_PATH)
    env = sdsl.Env_R2_Arrangement(segments)
    bbox = env.bounding_box()   # Voxel_R3 covering full (x, y, theta) space

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")
    ax.set_title("Click inside the map to cast rays and localize")

    # Draw the map walls via the environment's own representation (CGAL-exact coords)
    rep = env.get_representation()          # (N, 4) = [x1,y1,x2,y2]
    walls = rep.reshape(-1, 2, 2)           # (N, 2-points, 2-coords)
    ax.add_collection(LineCollection(walls, colors="black", linewidths=1))
    ax.autoscale()
    ax.margins(0.05)

    state = {"rays": None, "dots": None}

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        x, y = event.xdata, event.ydata

        # --- Cast rays ---
        angles, dists = cast_rays(env, x, y)
        ends = np.column_stack([x + dists * np.cos(angles),
                                y + dists * np.sin(angles)])   # (N_RAYS, 2)
        origin = np.full((N_RAYS, 2), [x, y])
        ray_segs = np.stack([origin, ends], axis=1)             # (N_RAYS, 2, 2)

        if state["rays"] is not None:
            state["rays"].remove()
        state["rays"] = ax.add_collection(
            LineCollection(ray_segs, colors="steelblue", linewidths=0.8, alpha=0.8)
        )

        # --- Localize ---
        # Odometry: K offsets from robot's origin, one per measurement.
        # Each entry is (dx=0, dy=0, dtheta=angle_i) — the robot didn't move,
        # it just measured in each direction.
        odometry = [sdsl.R3(0.0, 0.0, theta) for theta in angles]
        pred = sdsl.Predicate_Fwd2D_Arr(env, odometry, list(dists), KK_PRIME_RATIO, ERROR_BOUND)
        start_time = time.time()
        voxels = sdsl.localize_omp_forkjoin_3d(bbox, pred, RECURSION_DEPTH, verbose=True)
        end_time = time.time()
        print(f"Took: {end_time - start_time:.4f} [sec]")

        # Red dot at each voxel center — project onto xy plane (ignore theta)
        if state["dots"] is not None:
            state["dots"].remove()
            state["dots"] = None

        if voxels:
            centers = np.array([[v.midpoint()[0], v.midpoint()[1]] for v in voxels])
            state["dots"] = ax.scatter(
                centers[:, 0], centers[:, 1],
                c="red", s=20, alpha=0.4, zorder=5, linewidths=0,
            )

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

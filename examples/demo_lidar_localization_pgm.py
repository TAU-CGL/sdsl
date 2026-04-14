"""
Demo: load a ROS2 SLAM PGM map, click inside to cast 16 rays and localize.

Uses Env_2D_PGM, which operates directly on the occupancy-grid pixel array.
Ray casting uses an Amanatides & Woo DDA traversal over the grid.
The configuration space is 3-D: (x, y, theta).
Red dots show the (x, y) projection of each localized voxel center.

Run from the repo root:
    python examples/demo_lidar_localization_pgm.py
"""
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import sdsl
from sdsl.loaders.load_pgm_map import load_pgm_map

MAP_YAML     = "resources/maps/2d/slam/apt.yaml"
N_RAYS       = 16
RECURSION_DEPTH = 8
KK_PRIME_RATIO  = 0.7
ERROR_BOUND     = 0.015
TIMEOUT         = 1.0


def corrupt_measurements(dists, kk_prime_ratio):
    """Corrupt (1 − kk_prime_ratio) fraction of measurements by a random factor."""
    noisy = dists.copy()
    n_corrupt = round((1 - kk_prime_ratio) * len(dists)) - 1
    idx = np.random.choice(len(dists), size=n_corrupt, replace=False)
    noisy[idx] *= np.random.uniform(0.1, 3.0, size=n_corrupt)
    noisy += np.random.normal(0.0, 0.5 * ERROR_BOUND, size=len(noisy))
    return noisy


def cast_rays(env, x, y, n=N_RAYS):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dists  = np.array([env.measure_distance(sdsl.R3(x, y, theta)) for theta in angles])
    return angles, dists


def main():
    # -----------------------------------------------------------------
    # Load map and build environment
    # -----------------------------------------------------------------
    pgm = load_pgm_map(MAP_YAML)
    print(f"Map: {pgm.width} × {pgm.height} px  |  "
          f"resolution: {pgm.resolution} m/px  |  "
          f"origin: ({pgm.origin_x:.2f}, {pgm.origin_y:.2f})")

    env  = sdsl.Env_2D_PGM(
        pgm.grid, pgm.resolution, pgm.origin_x, pgm.origin_y,
        pgm.occupied_thresh, pgm.negate)
    bbox = env.bounding_box()

    # -----------------------------------------------------------------
    # Draw the map
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")
    ax.set_title("PGM demo — click inside the map to cast rays and localize")

    # imshow: row 0 is the top of the image; origin='upper' + correct extent
    # maps pixels to world (x, y) coordinates.
    world_xmax = pgm.origin_x + pgm.width  * pgm.resolution
    world_ymax = pgm.origin_y + pgm.height * pgm.resolution
    extent = [pgm.origin_x, world_xmax, pgm.origin_y, world_ymax]

    ax.imshow(
        pgm.grid,
        cmap="gray",
        origin="upper",
        extent=extent,
        interpolation="nearest",
        vmin=0, vmax=255,
        alpha=0.8,
        zorder=1,
    )
    ax.set_xlim(pgm.origin_x, world_xmax)
    ax.set_ylim(pgm.origin_y, world_ymax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    state = {"rays": None, "dots": None}

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        x, y = event.xdata, event.ydata

        # --- Cast rays ---
        angles, dists = cast_rays(env, x, y)
        # Cap displayed ray length so INF rays don't throw off the plot
        display_dists = np.minimum(dists, 20.0)
        ends = np.column_stack([
            x + display_dists * np.cos(angles),
            y + display_dists * np.sin(angles),
        ])
        origin_pts = np.full((N_RAYS, 2), [x, y])
        ray_segs   = np.stack([origin_pts, ends], axis=1)   # (N_RAYS, 2, 2)

        if state["rays"] is not None:
            state["rays"].remove()
        state["rays"] = ax.add_collection(
            LineCollection(ray_segs, colors="steelblue", linewidths=0.8,
                           alpha=0.8, zorder=3)
        )

        # --- Localize ---
        noisy_dists = corrupt_measurements(dists, KK_PRIME_RATIO)
        odometry    = [sdsl.R3(0.0, 0.0, theta) for theta in angles]
        pred        = sdsl.Predicate_Fwd2D_Arr(
            env, odometry, list(noisy_dists), KK_PRIME_RATIO, ERROR_BOUND)

        t0     = time.time()
        voxels = sdsl.localize_omp_forkjoin_3d(
            bbox, pred, RECURSION_DEPTH, timeout=TIMEOUT, verbose=True)
        print(f"Localization: {len(voxels)} voxels in {time.time()-t0:.3f} s")

        if state["dots"] is not None:
            state["dots"].remove()
            state["dots"] = None

        if voxels:
            centers = np.array([[v.midpoint()[0], v.midpoint()[1]] for v in voxels])
            state["dots"] = ax.scatter(
                centers[:, 0], centers[:, 1],
                c="red", s=20, alpha=0.5, zorder=5, linewidths=0,
            )

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

"""
Demo: load a SLAM occupancy-grid map (PGM + YAML), treat every occupied pixel
as a point in a 2D PCD, click inside the map to cast 16 rays and run 3D
localization using Env_2D_PCD.

The configuration space is 3D: (x, y, theta).
Red dots show the (x, y) projection of each localized voxel center.

Run from the repo root:
    python examples/demo_lidar_localization_slam.py
"""
import time
import os

import numpy as np
import matplotlib.pyplot as plt
import yaml

import sdsl

MAP_DIR       = "resources/maps/2d/slam/fl4_20250813_1725"
MAP_YAML      = os.path.join(MAP_DIR, "my_map.yaml")
N_RAYS        = 8
RECURSION_DEPTH = 9
KK_PRIME_RATIO  = 12/16
ERROR_BOUND     = 0.015
TIMEOUT         = 100.0


def load_slam_map_as_pcd(yaml_path):
    """Parse a ROS-style occupancy-grid YAML+PGM and return occupied pixels
    as an (N, 2) float64 array of world-frame (x, y) coordinates.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    pgm_path = os.path.join(os.path.dirname(yaml_path), os.path.basename(cfg["image"]))
    img = plt.imread(pgm_path).astype(np.float64)   # uint8, shape (H, W)

    resolution = float(cfg["resolution"])
    origin_x, origin_y = float(cfg["origin"][0]), float(cfg["origin"][1])
    occupied_thresh = float(cfg["occupied_thresh"])
    negate = int(cfg.get("negate", 0))

    # ROS convention: occupancy probability = (255 - value)/255  (negate=0)
    if negate:
        occ = img / 255.0
    else:
        occ = (255.0 - img) / 255.0

    rows, cols = np.where(occ > occupied_thresh)

    # Row 0 is the top of the image; origin is the bottom-left corner
    height = img.shape[0]
    x = origin_x + cols * resolution
    y = origin_y + (height - 1 - rows) * resolution

    return np.column_stack([x, y]).astype(np.float64)


def corrupt_measurements(dists, kk_prime_ratio):
    """Randomly corrupt (1 - kk_prime_ratio) fraction of measurements by a factor in [0.1, 3]."""
    noisy = dists.copy()
    n_corrupt = round((1 - kk_prime_ratio) * len(dists)) - 1
    idx = np.random.choice(len(dists), size=n_corrupt, replace=False)
    noisy[idx] *= np.random.uniform(0.1, 3.0, size=n_corrupt)
    noisy += np.random.normal(0.0, 0.5 * ERROR_BOUND, size=len(noisy))
    return noisy


def cast_rays(env, x, y, n=N_RAYS):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dists = np.array([env.measure_distance(sdsl.R3(x, y, theta)) for theta in angles])
    return angles, dists


def main():
    points = load_slam_map_as_pcd(MAP_YAML)
    env = sdsl.Env_2D_PCD(points)
    bbox = env.bounding_box()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")
    ax.set_title("SLAM map demo — click inside the map to cast rays and localize")

    rep = env.get_representation()   # (N, 3): columns are x, y, z
    ax.scatter(rep[:, 0], rep[:, 1], c="black", s=2, linewidths=0, zorder=2)
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
                                y + dists * np.sin(angles)])
        from matplotlib.collections import LineCollection
        origin = np.full((N_RAYS, 2), [x, y])
        ray_segs = np.stack([origin, ends], axis=1)

        if state["rays"] is not None:
            state["rays"].remove()
        state["rays"] = ax.add_collection(
            LineCollection(ray_segs, colors="steelblue", linewidths=0.8, alpha=0.8)
        )

        # --- Localize ---
        noisy_dists = corrupt_measurements(dists, KK_PRIME_RATIO)
        odometry = [sdsl.R3(0.0, 0.0, theta) for theta in angles]
        pred = sdsl.Predicate_Fwd2D_Arr(env, odometry, list(noisy_dists), KK_PRIME_RATIO, ERROR_BOUND)
        start_time = time.time()
        voxels = sdsl.localize_omp_forkjoin_3d(bbox, pred, RECURSION_DEPTH, timeout=TIMEOUT, verbose=True)
        end_time = time.time()
        print(f"Took: {end_time - start_time:.4f} [sec]")

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

"""
Demo: load lab_lidar.poly, sample it into a point cloud, click inside the map
to cast 16 rays and run 3D localization using Env_2D_PCD.

The point cloud is built by sampling equidistant points along every polygon
edge at spacing POINT_SPACING.  The configuration space is 3D: (x, y, theta).
Red dots show the (x, y) projection of each localized voxel center.

Run from the repo root:
    python examples/demo_lidar_localization_pcd.py
"""
import time

import numpy as np
import matplotlib.pyplot as plt

import sdsl

MAP_PATH = "resources/maps/2d/checkpoint.poly"
N_RAYS = 16
RECURSION_DEPTH = 9  # 8^(depth-1) voxels with AlwaysTrue predicate
KK_PRIME_RATIO = 0.8
ERROR_BOUND = 0.05
POINT_SPACING = 0.15   # distance between consecutive samples along each edge
TIMEOUT = 0.3


def load_poly_as_pcd(path, spacing=POINT_SPACING):
    """Parse a .poly file and sample equidistant points along every edge.

    Returns an (N, 2) float64 array of (x, y) point-cloud coordinates.
    """
    pts = np.loadtxt(path)           # (V, 2) polygon vertices
    pts_closed = np.vstack([pts, pts[0]])  # close the loop

    samples = []
    for i in range(len(pts)):
        p0 = pts_closed[i]
        p1 = pts_closed[i + 1]
        edge_len = np.linalg.norm(p1 - p0)
        n = max(2, int(np.round(edge_len / spacing)) + 1)
        # endpoint=False avoids duplicating the vertex shared with the next edge
        ts = np.linspace(0.0, 1.0, n, endpoint=False)
        for t in ts:
            samples.append(p0 + t * (p1 - p0))

    return np.array(samples, dtype=np.float64)


def cast_rays(env, x, y, n=N_RAYS):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dists = np.array([env.measure_distance(sdsl.R3(x, y, theta)) for theta in angles])
    return angles, dists


def main():
    points = load_poly_as_pcd(MAP_PATH)
    env = sdsl.Env_2D_PCD(points)
    bbox = env.bounding_box()   # Voxel_R3 covering full (x, y, theta) space

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")
    ax.set_title("PCD demo — click inside the map to cast rays and localize")

    # Draw the point cloud
    rep = env.get_representation()   # (N, 3): columns are x, y, z
    ax.scatter(rep[:, 0], rep[:, 1], c="black", s=4, linewidths=0, zorder=2)
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
        from matplotlib.collections import LineCollection
        origin = np.full((N_RAYS, 2), [x, y])
        ray_segs = np.stack([origin, ends], axis=1)             # (N_RAYS, 2, 2)

        if state["rays"] is not None:
            state["rays"].remove()
        state["rays"] = ax.add_collection(
            LineCollection(ray_segs, colors="steelblue", linewidths=0.8, alpha=0.8)
        )

        # --- Localize ---
        odometry = [sdsl.R3(0.0, 0.0, theta) for theta in angles]
        pred = sdsl.Predicate_Fwd2D_Arr(env, odometry, list(dists), KK_PRIME_RATIO, ERROR_BOUND)
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

"""
Demo: progressive localization with odometry fusion on a ROS2 SLAM PGM map.

On the first click the localized voxels receive equal belief (1/N each).
On every subsequent click the previous belief is fused with the odometry
(ground-truth offset between the two click positions) via fusion_2d, which
uses a Gaussian motion model in SE(2).

Voxels are coloured by their normalised belief using the ROS-style HSV map:
  near-zero  →  red
  mid        →  yellow → green → cyan
  near-one   →  magenta

Run from the repo root:
    python examples/demo_lidar_localization_pgm_fusion.py
"""
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb

import sdsl
from sdsl.loaders.load_pgm_map import load_pgm_map

MAP_YAML        = "resources/maps/2d/slam/simple_symmetry/map.yaml"
# MAP_YAML        = "resources/maps/2d/slam/apt_20250913_1449/my_map.yaml"
N_RAYS          = 16
RECURSION_DEPTH = 9
KK_PRIME_RATIO  = 0.8
ERROR_BOUND     = 0.015
TIMEOUT         = 1.0
FUSION_EPS      = 0.015   # Gaussian std-dev for the motion model (metres / radians)
INITIAL_THETA = 0.707


# ---------------------------------------------------------------------------
# ROS-style HSV colourmap  — hue 0 (red) → 300/360 (magenta) as p: 0 → 1
# ---------------------------------------------------------------------------
def _make_ros_cmap(n=256):
    colors = [hsv_to_rgb([i / n * (300.0 / 360.0), 1.0, 1.0]) for i in range(n)]
    return LinearSegmentedColormap.from_list("ros_hsv", colors, N=n)

ROS_CMAP = _make_ros_cmap()


def corrupt_measurements(dists, kk_prime_ratio):
    """Corrupt (1 − kk_prime_ratio) fraction of measurements by a random factor."""
    noisy     = dists.copy()
    if kk_prime_ratio >= 1.0:
        return noisy
    n_corrupt = round((1 - kk_prime_ratio) * len(dists)) - 1
    idx       = np.random.choice(len(dists), size=n_corrupt, replace=False)
    noisy[idx] *= np.random.uniform(0.1, 3.0, size=n_corrupt)
    noisy += np.random.normal(0.0, 0.5 * ERROR_BOUND, size=len(noisy))
    return noisy


def cast_rays(env, x, y, n=N_RAYS):
    angles = np.linspace(INITIAL_THETA, 2 * np.pi + INITIAL_THETA, n, endpoint=False)
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
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.subplots_adjust(right=0.88)           # leave room for colourbar
    ax.set_aspect("equal")
    ax.set_title(
        "PGM fusion demo — click to localize; belief updates with odometry fusion")

    world_xmax = pgm.origin_x + pgm.width  * pgm.resolution
    world_ymax = pgm.origin_y + pgm.height * pgm.resolution
    extent     = [pgm.origin_x, world_xmax, pgm.origin_y, world_ymax]

    ax.imshow(
        pgm.grid,
        cmap="gray", origin="upper", extent=extent,
        interpolation="nearest", vmin=0, vmax=255,
        alpha=0.8, zorder=1,
    )
    ax.set_xlim(pgm.origin_x, world_xmax)
    ax.set_ylim(pgm.origin_y, world_ymax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # Colourbar (attach to a ScalarMappable so it updates automatically)
    sm = plt.cm.ScalarMappable(cmap=ROS_CMAP, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    fig.colorbar(sm, cax=cbar_ax, label="Belief (normalised)")

    state = {
        "prev_voxels": None,   # list[Voxel_R3] from previous click
        "prev_belief": None,   # list[float]    from previous click
        "prev_pos":    None,   # (x, y) ground-truth of previous click
        "scatter":     None,   # PathCollection artist
        "rays_lc":     None,   # LineCollection artist
        "gt_marker":   None,   # cross marker artist
    }

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        x, y = event.xdata, event.ydata

        # ----------------------------------------------------------------
        # 1. Cast rays and display them
        # ----------------------------------------------------------------
        angles, dists = cast_rays(env, x, y)
        display_dists = np.minimum(dists, 20.0)
        ends        = np.column_stack([
            x + display_dists * np.cos(angles),
            y + display_dists * np.sin(angles),
        ])
        origin_pts = np.full((N_RAYS, 2), [x, y])
        ray_segs   = np.stack([origin_pts, ends], axis=1)   # (N_RAYS, 2, 2)

        if state["rays_lc"] is not None:
            state["rays_lc"].remove()
        state["rays_lc"] = ax.add_collection(
            LineCollection(ray_segs, colors="steelblue", linewidths=0.8,
                           alpha=0.8, zorder=3)
        )

        # ----------------------------------------------------------------
        # 2. Ground-truth cross marker
        # ----------------------------------------------------------------
        # if state["gt_marker"] is not None:
        #     state["gt_marker"].remove()
        # state["gt_marker"], = ax.plot(
        #     x, y, "w+", markersize=14, markeredgewidth=2, zorder=7)

        # ----------------------------------------------------------------
        # 3. Localize
        # ----------------------------------------------------------------
        noisy_dists = corrupt_measurements(dists, KK_PRIME_RATIO)
        odometry    = [sdsl.R3(0.0, 0.0, theta - INITIAL_THETA) for theta in angles]
        pred        = sdsl.Predicate_Fwd2D_Arr(
            env, odometry, list(noisy_dists), KK_PRIME_RATIO, ERROR_BOUND)

        t0     = time.time()
        voxels = sdsl.localize_omp_forkjoin_3d(
            bbox, pred, RECURSION_DEPTH, timeout=TIMEOUT, verbose=True)
        print(f"Localization: {len(voxels)} voxels in {time.time()-t0:.3f} s")

        voxels = sdsl.cleanup_SE2(voxels)
        print(f"After cleanup: {len(voxels)} connected components")

        if not voxels:
            state["prev_pos"] = (x, y)
            fig.canvas.draw_idle()
            return

        # ----------------------------------------------------------------
        # 4. Compute belief
        # ----------------------------------------------------------------
        if state["prev_voxels"] is None or state["prev_belief"] is None:
            # First click — uniform belief
            n      = len(voxels)
            belief = [1.0 / n] * n
            print("First click — uniform belief assigned.")
        else:
            # Subsequent clicks — fuse with odometry
            prev_x, prev_y = state["prev_pos"]
            dx = x - prev_x
            dy = y - prev_y
            Ut = sdsl.R3(dx, dy, 0.0)   # robot frame ≈ world frame (theta=0)
            print(f"Fusing with Ut=({dx:.3f}, {dy:.3f}, 0)")
            belief = sdsl.fusion_2d(
                state["prev_voxels"], state["prev_belief"],
                voxels, Ut, FUSION_EPS,
            )

        # ----------------------------------------------------------------
        # 5. Draw voxel centres coloured by belief
        # ----------------------------------------------------------------
        centers = np.array([[v.midpoint()[0], v.midpoint()[1]] for v in voxels])
        beliefs = np.array(belief, dtype=float)

        for i, (v, b) in enumerate(zip(voxels, beliefs)):
            print(f"Voxel {i}: center=({v.midpoint()[0]:.6f}, {v.midpoint()[1]:.6f}), ({v.midpoint()[2]:.6f}) "
                  f"belief={b:.4f}, volume={v.volume():.4f}")

        if state["scatter"] is not None:
            state["scatter"].remove()
        state["scatter"] = ax.scatter(
            centers[:, 0], centers[:, 1],
            c=beliefs, cmap=ROS_CMAP, vmin=0.0, vmax=beliefs.max() or 1.0,
            s=20, alpha=0.2, zorder=5, linewidths=0,
        )

        # ----------------------------------------------------------------
        # 6. Save state for next fusion step
        # ----------------------------------------------------------------
        state["prev_voxels"] = voxels
        state["prev_belief"] = belief
        state["prev_pos"]    = (x, y)

        fig.canvas.draw_idle()

        vol = voxels[0].volume() / (voxels[0].top_right[2] - voxels[0].bottom_left[2])   
        vol = 1
        b = beliefs
        N = len(b)
        print(f"Belief distribution: {np.max(b)}, {np.min(b)}, {np.sum(b)}")
        b = b / b.sum()
        # print(f"Entropy: {-np.sum(b * np.log(b + 1e-12))}")
        mask = b > 0
        H = -np.sum(b[mask] * np.log(b[mask]) * vol)
        # H_norm = H / np.log(N)
        print(f"Entropy: {H}")

        # Fixed entropy
        H = sdsl.entropy_SE2(voxels, list(beliefs))
        print(f"Projected entropy: {H}")


    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


if __name__ == "__main__":
    main()

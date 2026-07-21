"""
Demonstration of 2D localization and motion simulation using SDSL.
"""
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb
from matplotlib.patches import Rectangle

import sdsl
from sdsl.loaders.load_pgm_map import load_pgm_map

try:
    from .ChooseMovement import ChooseMovement
except ImportError:  # pragma: no cover - direct script execution fallback
    from ChooseMovement import ChooseMovement

MAP_YAML    = "resources/maps/2d/slam/simple_symmetry/square.yaml"
IMG_PATH    = "resources/maps/2d/slam/simple_symmetry/square_c.png"
# MAP_YAML        = "resources/maps/2d/slam/simple_symmetry/symmetry_2.yaml"
# IMG_PATH        = "resources/maps/2d/slam/simple_symmetry/symmetry_2_c.png"
N_RAYS          = 16
RECURSION_DEPTH = 10
KK_PRIME_RATIO  = 1.0
ERROR_BOUND     = 0.015
TIMEOUT         = 1.0
FUSION_EPS      = 0.015   # Gaussian std-dev for the motion model (metres / radians)


# ---------------------------------------------------------------------------
# ROS-style HSV colourmap  — hue 0 (red) → 300/360 (magenta) as p: 0 → 1
# ---------------------------------------------------------------------------
def _make_ros_cmap(n=256):
    colors = [hsv_to_rgb([i / n * (300.0 / 360.0), 1.0, 1.0]) for i in range(n)]
    return LinearSegmentedColormap.from_list("ros_hsv", colors, N=n)

ROS_CMAP = _make_ros_cmap()

def relative_to_global_movement(dx,dy, theta):
    delta = np.asarray([dx,dy], dtype=float)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    return R @ delta

def corrupt_measurements(dists, kk_prime_ratio):
    """Corrupt (1 − kk_prime_ratio) fraction of measurements by a random factor."""
    noisy     = dists.copy()
    if kk_prime_ratio >= 1.0:
        return noisy
    #override random selection for debugging
    return noisy
    ##
    n_corrupt = round((1 - kk_prime_ratio) * len(dists)) - 1
    idx       = np.random.choice(len(dists), size=n_corrupt, replace=False)
    noisy[idx] *= np.random.uniform(0.1, 3.0, size=n_corrupt)
    noisy += np.random.normal(0.0, 0.5 * ERROR_BOUND, size=len(noisy))
    return noisy


def cast_rays(env, x, y, z, n=N_RAYS):
    angles = np.linspace(z, 2 * np.pi + z, n, endpoint=False)
    dists  = np.array([env.measure_distance(sdsl.R3(x, y, theta)) for theta in angles])
    return angles, dists

def localize(env, x, y, z, state):
    angles, dists = cast_rays(env, x, y, z) #glabal xyz, global angles
    noisy_dists   = corrupt_measurements(dists, KK_PRIME_RATIO)
    odometry      = [sdsl.R3(0.0, 0.0, theta - z) for theta in angles] #relative angles
    pred          = sdsl.Predicate_Fwd2D_Arr(
        env, odometry, list(noisy_dists), KK_PRIME_RATIO, ERROR_BOUND)

    bbox   = env.bounding_box()
    voxels = sdsl.localize_omp_forkjoin_3d(
        bbox, pred, RECURSION_DEPTH, timeout=TIMEOUT, verbose=True)
    voxels = sdsl.cleanup_SE2(voxels)

    if state["prev_voxels"] is None or state["prev_belief"] is None:
        # First click — uniform belief
        n      = len(voxels)
        belief = [1.0 / n] * n
        print("First click — uniform belief assigned.")
    else:
        # Subsequent clicks — fuse with odometry
        prev_x, prev_y, prev_z = state["prev_pos"]
        dx = x - prev_x
        dy = y - prev_y
        dz = (z - prev_z)%(2*np.pi) 
        print(f"calculated Odometry: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}")
        # z is the angle with (1,0 vector) and vf
        vf = np.array([np.cos(prev_z), np.sin(prev_z)])
        vl = np.array([-np.sin(prev_z), np.cos(prev_z)])
        displacement = np.array([dx, dy])
        print("displacement", displacement)

        vf_inner = np.dot(displacement, vf)  # inner product with forward direction
        vl_inner = np.dot(displacement, vl)  # inner product with right direction

        print("vf", vf, "inner:", vf_inner)
        print("vl", vl, "inner:", vl_inner)
        Ut = sdsl.R3(vf_inner, vl_inner, dz)   
        print(f"Fusing with Ut=({vf_inner:.3f}, {vl_inner:.3f}, {dz:.3f})")
        belief = sdsl.fusion_2d(
            state["prev_voxels"], state["prev_belief"],
            voxels, Ut, FUSION_EPS,
        )
    
    state["prev_voxels"] = voxels
    state["prev_belief"] = belief
    state["prev_pos"]    = (x, y, z)

    return angles, dists, voxels, belief
    
def visualize(fig,ax, state, angles, dists):
    x, y, z = state["prev_pos"]
    # display_dists = np.minimum(dists, 100.0)
    # ends        = np.column_stack([
    #     x + display_dists * np.cos(angles),
    #     y + display_dists * np.sin(angles),
    # ])
    # origin_pts = np.full((N_RAYS, 2), [x, y])
    # ray_segs   = np.stack([origin_pts, ends], axis=1)   # (N_RAYS, 2, 2)
    # if state["rays_lc"] is not None:
    #     state["rays_lc"].remove()
    # state["rays_lc"] = ax.add_collection(
    #     LineCollection(ray_segs, colors="steelblue", linewidths=0.8,
    #                     alpha=0.8, zorder=3)
    # )

    if state.get("z_arrow") is not None:
        state["z_arrow"].remove()
    arrow_length = 3.0
    arrow_dx = arrow_length * np.cos(z)
    arrow_dy = arrow_length * np.sin(z)
    state["z_arrow"] = ax.arrow(
        x, y, arrow_dx, arrow_dy, head_width=0.3, head_length=0.2,
        fc="red", ec="red", linewidth=2.0, zorder=6)
    if state.get("position") is not None:
        artists = state["position"] if isinstance(state["position"], (list, tuple)) else [state["position"]]
        for artist in artists:
            if artist is not None:
                artist.remove()
    state["position"] = ax.plot([x], [y], marker='+', color='black',
            markersize=14, markeredgewidth=2, zorder=7)
    
    fig.canvas.draw_idle()



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
        # "scatter":     None,   # PathCollection artist
        "rays_lc":     None,   # LineCollection artist
        # "gt_marker":   None,   # cross marker artist
        "z_arrow":     None,   # arrow showing z angle with x axis
        "position":    None,   # marker showing (x, y) position
        "x":           None,   # current x position
        "y":           None,   # current y position
        "z":           None,   # current z angle
    }

    mover = ChooseMovement(env, yaml_path=MAP_YAML, image_path=IMG_PATH)
    click_count = {"value": 0}

    def on_click(event):
        if click_count["value"] == 0:
            state["x"] = event.xdata
            state["y"] = event.ydata
            # state["x"] = 2.790
            # state["y"] = 8.166
            state["z"] = np.pi/2.0

        click_count["value"] += 1

        angles, dists, voxels, belief = localize(env, state["x"], state["y"], state["z"], state)
        visualize(fig, ax, state, angles, dists)
        sim_results = mover.simulate_movements(angles, dists, voxels, belief)
        print(f'Simulated {len(sim_results)} motion hypotheses')
        # mover.visualize_voxels(ax, voxels,
        #                    color=(1.0, 0.0, 0.0, 0.25),
        #                    edgecolor='green',
        #                    label='original')
        # for sim_idx, sim in enumerate(sim_results):
        #     moved = sim['cleaned_voxels'] or sim['surviving_voxels']
        #     mover.visualize_voxels(ax, moved,
        #                        color=(1.0, 0.65, 0.0, 0.30),
        #                        edgecolor='brown',
        #                        label=f'moved_{sim_idx}')
        # visualize(fig, ax, state, angles, dists)
        mover.visualize_simulation_result(ax, sim_results, voxels)
        def score(sim):
            entropy = sim.get('entropy')
            avg_dist = sim.get('avg_neighbor_room_distance')
            avg_wall_dist = sim.get('avg_wall_distance') # if samller than 0.2 then panelize
            if entropy is None or avg_dist is None or avg_wall_dist is None:
                return float('inf')
            return entropy + avg_dist + (0.2 - avg_wall_dist)*10.0 if avg_wall_dist < 0.2 else entropy + avg_dist 

        best_sim = min(
            sim_results,
            key=score,
            default=None,
        )
        # dx, dy = relative_to_global_movement(best_sim['dx'], best_sim['dy'], state['z']) if best_sim else (0, 0)
        # dx, dy = best_sim['dx'], best_sim['dy'] if best_sim else (0, 0)
        print(f"Best simulation: {best_sim if best_sim else 'None'}")
        step = float(np.minimum(np.maximum(best_sim["distance"]-0.05, 0), 0.3))
        dx = step * np.cos(best_sim['angle'] + state['z']) if best_sim else 0
        dy = step * np.sin(best_sim['angle'] + state['z']) if best_sim else 0
        print(f"Best movement: dx={dx:.3f}, dy={dy:.3f}, angle={best_sim['angle'] if best_sim else 0:.3f}")
        print(f"prev position: x={state['x']:.3f}, y={state['y']:.3f}, z={state['z']:.3f}")
        state["x"] += dx
        state["y"] += dy
        state["z"] = (best_sim['angle']+state['z'])% (2 * np.pi) if best_sim else state["z"]
        print(f"Updated position: x={state['x']:.3f}, y={state['y']:.3f}, z={state['z']:.3f}")
        print("..")

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


if __name__ == "__main__":
    main()

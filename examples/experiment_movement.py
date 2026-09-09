"""
Run headless experiments of the demo_movement logic across maps.

Produces average distance traveled until high-confidence localization.
"""
import argparse
import importlib.util
import math
import os
import random
import statistics
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# Defaults copied from demo_movement.py
DEFAULT_MAP = "resources/maps/2d/slam/simple_symmetry/square.yaml"
DEFAULT_IMG = "resources/maps/2d/slam/simple_symmetry/square_c.png"


def load_demo_module():
    here = os.path.dirname(__file__)
    demo_path = os.path.join(here, "demo_movement.py")
    spec = importlib.util.spec_from_file_location("demo_movement", demo_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_free_point(pgm, env, min_clearance=0.3, max_tries=200):
    xmin = pgm.origin_x
    ymin = pgm.origin_y
    xmax = pgm.origin_x + pgm.width * pgm.resolution
    ymax = pgm.origin_y + pgm.height * pgm.resolution
    for _ in range(max_tries):
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        # measure distance to nearest obstacle; accept if clear enough
        try:
            inside = env.contains(module.sdsl.R3(x, y, 0.0))
        except Exception:
            inside = False
        if inside:
            return x, y
    return None


def run_on_map(module, map_yaml, image_path, repeats=10, max_steps=200, threshold=0.55, seed=None, visualize=False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    pgm = module.load_pgm_map(map_yaml)
    env = module.sdsl.Env_2D_PGM(
        pgm.grid, pgm.resolution, pgm.origin_x, pgm.origin_y,
        pgm.occupied_thresh, pgm.negate)
    mover = module.ChooseMovement(env, yaml_path=map_yaml, image_path=image_path)

    results = []
    xmin = pgm.origin_x
    ymin = pgm.origin_y
    xmax = pgm.origin_x + pgm.width * pgm.resolution
    ymax = pgm.origin_y + pgm.height * pgm.resolution

    for rep in range(repeats):
        fig = None
        ax = None
        if visualize:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))
            world_xmax = pgm.origin_x + pgm.width * pgm.resolution
            world_ymax = pgm.origin_y + pgm.height * pgm.resolution
            extent = [pgm.origin_x, world_xmax, pgm.origin_y, world_ymax]
            ax.imshow(pgm.grid, cmap="gray", origin="upper", extent=extent, interpolation="nearest", vmin=0, vmax=255, alpha=0.8, zorder=1)
            ax.set_xlim(pgm.origin_x, world_xmax)
            ax.set_ylim(pgm.origin_y, world_ymax)
            ax.set_aspect('equal')
        # pick random start in free space
        start = None
        for _ in range(300):
            x0 = random.uniform(xmin, xmax)
            y0 = random.uniform(ymin, ymax)
            try:
                inside0 = env.contains(module.sdsl.R3(x0, y0, 0.0))
            except Exception:
                inside0 = False
            if inside0:
                start = (x0, y0)
                break
        if start is None:
            # fallback to center
            start = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)

        x, y = start
        z = random.uniform(0.0, 2 * math.pi)

        state = {
            "prev_voxels": None,
            "prev_belief": None,
            "prev_pos": None,
            "x": x,
            "y": y,
            "z": z,
        }

        total_distance = 0.0
        converged = False
        steps = 0

        for step in range(max_steps):
            steps += 1
            angles, dists, voxels, belief = module.localize(env, state["x"], state["y"], state["z"], state)

            # compute component confidence if possible
            max_conf = 0.0
            if belief:
                try:
                    comps, belief_sums = module.sdsl.connectedComponentsWithBeliefs(voxels, True, belief)
                    if belief_sums:
                        max_conf = max(belief_sums)
                except Exception:
                    max_conf = max(belief) if belief else 0.0

            if max_conf >= threshold:
                converged = True
                break

            sim_results = mover.simulate_movements(angles, dists, voxels, belief)

            if visualize and fig is not None and ax is not None:
                try:
                    module.visualize(fig, ax, state, angles, dists)
                    mover.visualize_simulation_result(ax, sim_results, voxels)
                    plt.pause(0.001)
                except Exception:
                    # don't fail the experiment on visualization errors
                    pass

            def score(sim):
                entropy = sim.get('entropy')
                avg_dist = sim.get('avg_neighbor_room_distance')
                avg_wall_dist = sim.get('avg_wall_distance')
                if entropy is None or avg_dist is None or avg_wall_dist is None:
                    return float('inf')
                return entropy + avg_dist + (0.2 - avg_wall_dist) * 10.0 if avg_wall_dist < 0.2 else entropy + avg_dist

            best_sim = min(sim_results, key=score, default=None)
            if best_sim is None:
                break

            step_len = float(np.minimum(np.maximum(best_sim["distance"] - 0.15, 0), 0.3))
            dx = step_len * math.cos(best_sim['angle'] + state['z'])
            dy = step_len * math.sin(best_sim['angle'] + state['z'])

            moved = math.hypot(dx, dy)
            total_distance += moved

            state["x"] += dx
            state["y"] += dy
            state["z"] = (best_sim['angle'] + state['z']) % (2 * math.pi)

        results.append({
            "converged": converged,
            "distance": total_distance,
            "steps": steps,
        })
        # If visualization was enabled, give a short non-blocking pause so the final
        # frame is visible, then close the figure and continue to the next repeat.
        if visualize and fig is not None:
            try:
                plt.pause(0.5)
            except Exception:
                pass
            try:
                plt.close(fig)
            except Exception:
                pass

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Run headless localization experiments")
    p.add_argument("--maps", nargs="+", default=[DEFAULT_MAP], help="YAML map files to run (space separated)")
    p.add_argument("--images", nargs="*", default=[DEFAULT_IMG], help="Optional image paths matching maps")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None, help="Optional CSV output path")
    # Visualization: default enabled; provide explicit disabling flag
    vis_group = p.add_mutually_exclusive_group()
    vis_group.add_argument("--visualize", dest="visualize", action="store_true", help="Enable visualization (default)")
    vis_group.add_argument("--no-visualize", dest="visualize", action="store_false", help="Disable visualization")
    p.set_defaults(visualize=True)
    return p.parse_args()


def main():
    global module
    module = load_demo_module()
    args = parse_args()
    maps = args.maps
    images = args.images or [None] * len(maps)

    all_summary = {}
    for i, map_yaml in enumerate(maps):
        img = images[i] if i < len(images) else None
        print(f"Running map: {map_yaml} repeats={args.repeats}")
        res = run_on_map(module, map_yaml, img, repeats=args.repeats, max_steps=args.max_steps, threshold=args.threshold, seed=args.seed, visualize=args.visualize)
        distances = [r['distance'] for r in res if r['converged']]
        conv_count = sum(1 for r in res if r['converged'])
        avg_distance = statistics.mean(distances) if distances else float('nan')
        all_summary[map_yaml] = {
            'results': res,
            'converged': conv_count,
            'avg_distance': avg_distance,
        }
        print(f"Map {map_yaml}: converged {conv_count}/{len(res)}; avg distance (converged) = {avg_distance:.3f}")

    if args.out:
        import csv
        with open(args.out, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['map', 'repeat', 'converged', 'distance', 'steps'])
            for m, info in all_summary.items():
                for idx, r in enumerate(info['results']):
                    w.writerow([m, idx, int(r['converged']), f"{r['distance']:.6f}", r['steps']])
    print("Done.")


if __name__ == '__main__':
    main()

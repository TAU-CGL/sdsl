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
import sdsl

# Defaults copied from demo_movement.py
DEFAULT_MAP_LIST = ["resources/maps/2d/slam/simple_symmetry/square.yaml",
                    "resources/maps/2d/slam/simple_symmetry/symmetry_2.yaml",
                    "resources/maps/2d/slam/simple_symmetry/plus.yaml",
                    "resources/maps/2d/slam/apt_20250913_1449/my_map.yaml"
                    ]
DEFAULT_IMG_LIST = ["resources/maps/2d/slam/simple_symmetry/square_c.png",
                    "resources/maps/2d/slam/simple_symmetry/symmetry_2_c.png",
                    "resources/maps/2d/slam/simple_symmetry/plus_c.png",
                    "resources/maps/2d/slam/apt_20250913_1449/my_map_c.png"]


def load_demo_module():
    examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")
    # demo_movement falls back to `from ChooseMovement import ...`, which needs examples/ on sys.path
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    demo_path = os.path.join(examples_dir, "demo_movement.py")
    spec = importlib.util.spec_from_file_location("demo_movement", demo_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_is_in_dominant_belief_voxel(voxels, belief, x, y, components_and_sums=None):
    """Return True only when the current pose lies inside the dominant belief voxel.

    components_and_sums: optional precomputed
    sdsl.connectedComponentsWithBeliefs(voxels, True, belief) result (avoids recomputing it).
    """
    if not voxels or not belief:
        return False

    try:
        if components_and_sums is None:
            components_and_sums = sdsl.connectedComponentsWithBeliefs(voxels, True, belief)
        components, belief_sums = components_and_sums
    except Exception:
        return False

    if not components or not belief_sums:
        return False

    best_idx = int(np.argmax(belief_sums))
    dominant_voxel = components[best_idx]
    bl = dominant_voxel.bottom_left
    tr = dominant_voxel.top_right
    return bl[0] <= x <= tr[0] and bl[1] <= y <= tr[1]


OUTCOMES = ("correct", "incorrect", "timeout", "stuck_no_step", "stuck_chance", "stuck_no_entropy", "collision")


def path_hits_wall(env, x0, y0, x1, y1, spacing):
    """Return True if any point on the straight segment (x0,y0)->(x1,y1) is outside free space."""
    n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / spacing)))
    for i in range(1, n + 1):
        t = i / n
        try:
            free = env.contains(sdsl.R3(x0 + t * (x1 - x0), y0 + t * (y1 - y0), 0.0))
        except Exception:
            free = False
        if not free:
            return True
    return False


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


def run_on_map(module, map_yaml, image_path, repeats=10, max_steps=200, threshold=0.55, seed=None, visualize=False, random_move=False, min_clearance=0.0, alpha=1.0, forward_safety_margin=0.2, entropy_eps=0.05):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    pgm = module.load_pgm_map(map_yaml)
    env = module.sdsl.Env_2D_PGM(
        pgm.grid, pgm.resolution, pgm.origin_x, pgm.origin_y,
        pgm.occupied_thresh, pgm.negate)
    mover = module.ChooseMovement(env, yaml_path=map_yaml, image_path=image_path, min_clearance=min_clearance)

    results = []
    xmin = pgm.origin_x
    ymin = pgm.origin_y
    xmax = pgm.origin_x + pgm.width * pgm.resolution
    ymax = pgm.origin_y + pgm.height * pgm.resolution

    # Draw every starting pose up front, before any run touches the RNG. Runs consume
    # `random` (random-mode move choice) and `np.random` (sensor noise), so sampling
    # starts lazily would make repeat k's start depend on what earlier repeats did.
    # Drawing them here means that, for a given seed, the best and random policies and
    # every grid-search combination start from exactly the same poses.
    starts = []
    for _ in range(repeats):
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
        starts.append((start[0], start[1], random.uniform(0.0, 2 * math.pi)))

    for rep in range(repeats):
        # Visited-room flags live on the mover and drive the room-distance tie-break;
        # clear them so each repeat starts with no rooms visited (like a fresh demo run).
        mover.reset_visited_flags()
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
        x, y, z = starts[rep]

        state = {
            "prev_voxels": None,
            "prev_belief": None,
            "prev_pos": None,
            "x": x,
            "y": y,
            "z": z,
        }

        total_distance = 0.0
        # stays "timeout" if the loop runs out of steps without another outcome
        outcome = "timeout"
        steps = 0

        for step in range(max_steps):
            steps += 1
            angles, dists, voxels, belief = module.localize(env, state["x"], state["y"], state["z"], state)

            # compute component confidence if possible
            max_conf = 0.0
            inside_dominant_voxel = False
            if belief:
                try:
                    # localize() already computed the connected components of this exact
                    # (voxels, belief); reuse them instead of recomputing (twice).
                    cc = state.get("cc") or module.sdsl.connectedComponentsWithBeliefs(voxels, True, belief)
                    comps, belief_sums = cc
                    if belief_sums:
                        max_conf = max(belief_sums)
                        inside_dominant_voxel = state_is_in_dominant_belief_voxel(
                            voxels, belief, state["x"], state["y"], components_and_sums=cc)
                except Exception:
                    max_conf = max(belief) if belief else 0.0
                    inside_dominant_voxel = False

            if max_conf >= threshold and inside_dominant_voxel:
                outcome = "correct"
                break
            if max_conf >= threshold and not inside_dominant_voxel:
                # the posterior is concentrated but not aligned with the current pose;
                # treat it as a failed convergence instead of a success.
                outcome = "incorrect"
                # Do not continue the run once the belief has been classified as
                # a definite outcome. This prevents a strong but wrong posterior
                # from being treated as a still-running trial.
                break

            # The random baseline never reads entropy / room distance, so skip computing them
            # (same result list, same RNG consumption, same chosen move).
            sim_results = mover.simulate_movements(angles, dists, voxels, belief, alpha=alpha,
                                                   forward_safety_margin=forward_safety_margin,
                                                   compute_metrics=not random_move)

            if visualize and fig is not None and ax is not None:
                try:
                    module.visualize(fig, ax, state, angles, dists)
                    mover.visualize_simulation_result(ax, sim_results, voxels)
                    plt.pause(0.001)
                except Exception:
                    # don't fail the experiment on visualization errors
                    pass

            if random_move:
                # pick a random feasible movement instead of the best-scored one
                best_sim = None
                try:
                    if sim_results:
                        best_sim = random.choice(sim_results)
                except Exception:
                    best_sim = None
            else:
                _, best_sim = mover.select_best_simulation(sim_results, entropy_eps=entropy_eps)

            if sim_results:
                e_min = min((s['entropy'] for s in sim_results if s.get('entropy') is not None), default=None)
                sorted_sims = sorted(
                    sim_results,
                    key=lambda s: (
                        s['entropy'] if s.get('entropy') is not None else float('inf'),
                        s['avg_neighbor_room_distance'] if s.get('avg_neighbor_room_distance') is not None else float('inf'),
                    ),
                )
                print(f"  step {step+1} — {len(sorted_sims)} candidate moves (sorted by entropy, then room_dist):", file=sys.stderr)
                for s in sorted_sims:
                    e = s.get('entropy')
                    rd = s.get('avg_neighbor_room_distance')
                    in_tie = e is not None and e_min is not None and e <= e_min + entropy_eps
                    chosen = s is best_sim
                    tag = " ← best" if chosen else ("  [tie]" if in_tie else "")
                    e_str = f"{e:.4f}" if e is not None else "None"
                    rd_str = f"{rd:.3f}" if rd is not None else "None"
                    print(
                        f"    angle={math.degrees(s['angle']):6.1f}° step={s['step']:.2f}m"
                        f"  entropy={e_str}  room_dist={rd_str}"
                        f"  env_rej={s.get('collision_count_env', '?')} mask_rej={s.get('collision_count_mask', '?')}"
                        f"{tag}",
                        file=sys.stderr,
                    )

            if best_sim is None:
                if not sim_results:
                    # simulate_movements returned nothing — either all steps were 0
                    # or all moves failed the alpha chance constraint
                    any_valid_step = any(
                        float(np.minimum(np.maximum(d - forward_safety_margin, 0), 0.3)) > 0.01
                        for d in dists
                    )
                    outcome = "stuck_no_step" if not any_valid_step else "stuck_chance"
                else:
                    # moves existed but select_best found none with valid entropy/room_distance
                    outcome = "stuck_no_entropy"
                break

            step_len = float(np.minimum(np.maximum(best_sim["distance"] - forward_safety_margin, 0), 0.3))
            dx = step_len * math.cos(best_sim['angle'] + state['z'])
            dy = step_len * math.sin(best_sim['angle'] + state['z'])

            moved = math.hypot(dx, dy)
            total_distance += moved

            if path_hits_wall(env, state["x"], state["y"], state["x"] + dx, state["y"] + dy, 0.5 * pgm.resolution):
                outcome = "collision"
                break

            state["x"] += dx
            state["y"] += dy
            state["z"] = (best_sim['angle'] + state['z']) % (2 * math.pi)

        results.append({
            "outcome": outcome,
            "converged": outcome == "correct",
            "distance": total_distance,
            "steps": steps,
        })
        if outcome == "correct":
            print(f"  repeat {rep+1}/{repeats}: converged in {steps} steps ({total_distance:.2f} m)", file=sys.stderr)
        elif outcome == "stuck_no_step":
            print(f"  repeat {rep+1}/{repeats}: stuck — all directions blocked (step=0) after {steps} steps", file=sys.stderr)
        elif outcome == "stuck_chance":
            print(f"  repeat {rep+1}/{repeats}: stuck — all moves failed alpha chance constraint after {steps} steps", file=sys.stderr)
        elif outcome == "stuck_no_entropy":
            print(f"  repeat {rep+1}/{repeats}: stuck — no move had valid entropy/room_distance after {steps} steps", file=sys.stderr)
        else:
            print(f"  repeat {rep+1}/{repeats}: {outcome} after {steps} steps ({total_distance:.2f} m)", file=sys.stderr)
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
    p.add_argument("--maps", nargs="+", default=DEFAULT_MAP_LIST, help="YAML map files to run (space separated)")
    p.add_argument("--images", nargs="*", default=DEFAULT_IMG_LIST, help="Optional image paths matching maps")
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None, help="Optional CSV output path")
    # Defaults = validation candidate A (validation/run_validation.sh, analyze_validation.py).
    p.add_argument("--min-clearance", type=float, default=0.0, help="Minimum wall clearance in metres (default: 0.0)")
    p.add_argument("--alpha", type=float, default=1.0, help="Chance-constraint threshold: max allowed collision probability (default: 1.0, i.e. disabled)")
    p.add_argument("--forward-safety-margin", type=float, default=0.2, help="Metres subtracted from forward ray before stepping (default: 0.2)")
    p.add_argument("--entropy-eps", type=float, default=0.05, help="Entropy tie window for best-move selection; ties broken by room distance (default: 0.05)")
    p.add_argument("--grid-search", action="store_true", help="Run grid search over min_clearance, alpha, forward_safety_margin, entropy_eps")
    p.add_argument("--grid-maps-count", type=int, default=2, help="Number of maps (from the top of the list) to use in grid search (default: 2)")
    p.add_argument("--grid-repeats", type=int, default=20, help="Repeats per combo in grid search (default: 20)")
    p.add_argument("--grid-max-steps", type=int, default=50, help="Max steps per run in grid search (default: 50)")
    # Visualization: default enabled; provide explicit disabling flag
    vis_group = p.add_mutually_exclusive_group()
    vis_group.add_argument("--visualize", dest="visualize", action="store_true", help="Enable visualization (default)")
    vis_group.add_argument("--no-visualize", dest="visualize", action="store_false", help="Disable visualization")
    p.set_defaults(visualize=False)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
# Phase 1 (current): sweep the three parameters most directly tied to the
# wall-avoidance vs. stuck trade-off, plus the move-selection tie window.
#
#   min_clearance (inflated wall mask margin; the map is assumed to be already
#   inflated by the robot's footprint, so this is the whole margin):
#     Controls the forbidden zone around walls. Each surviving belief voxel's
#     endpoint must be outside this margin. Larger values keep the robot
#     further from walls but reject more moves, raising the "stuck" rate in
#     narrow corridors and doorways.
#     Values: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25] m  (0.05 = demo_movement / ChooseMovement default)
#
#   alpha (chance-constraint threshold):
#     A candidate move is rejected if more than alpha of the total belief mass
#     would collide. Smaller alpha is more conservative (fewer moves accepted),
#     larger alpha allows moves where more hypotheses collide — which
#     artificially lowers entropy near walls for the wrong reasons.
#     Values: [0.05, 0.10, 0.15, 1.0]  (1.0 = constraint off: a direction is never
#     rejected for collision risk)
#
#   forward_safety_margin:
#     Subtracted from the forward ray distance before computing step length:
#     step = clip(dist - forward_safety_margin, 0, max_step). Larger values
#     keep more space in front of the robot but shorten every step, slowing
#     convergence. Too small risks stepping into walls on the forward ray.
#     Values: [0.05, 0.10, 0.20] m
#
#   entropy_eps (move-selection tie window):
#     Moves whose predicted entropy is within entropy_eps of the minimum are
#     treated as ties and broken by avg_neighbor_room_distance. Wider windows
#     let room distance matter more; 0-ish means pure entropy greedy.
#     Values: [0.005, 0.01, 0.05, 0.1]  (0.1 = demo_movement.py)
#
#   What to look for in results:
#     - Maximise "correct", minimise "collision" and "stuck".
#     - "stuck" rises with larger min_clearance or smaller alpha (too conservative).
#     - "collision" rises with smaller min_clearance or larger alpha (too permissive).
#     - avg_distance for correct runs: lower is faster convergence.
#     - Pick the combination with the best correct/collision balance; if two
#       combos tie on correct, prefer lower collision over lower stuck.
#
# Phase 2 (next): fix the best Phase-1 combination and sweep:
#   - max_step        [0.2,  0.3,  0.5]    — step-length cap; larger steps
#                                             explore faster but risk collisions
#
# Phase 3 (optional): localization quality parameters that affect what the
# mover sees as entropy:
#   - FUSION_EPS      [0.005, 0.015, 0.05] — motion-model noise spread
#   - KK_PRIME_RATIO  [0.5, 0.7, 1.0]      — fraction of uncorrupted rays
# ---------------------------------------------------------------------------

# 0.0 is ChooseMovement's default min_clearance (validation candidate A)
GRID_MIN_CLEARANCE      = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
GRID_ALPHA              = [0.05, 0.10, 0.15, 1.0]
GRID_FORWARD_SAFETY     = [0.05, 0.10, 0.20]
# 0.1 is the value hard-coded in demo_movement.py
GRID_ENTROPY_EPS        = [0.005, 0.01, 0.05, 0.1]


def run_grid_search(module, args):
    import contextlib
    import csv
    import io
    import itertools
    import os as _os

    @contextlib.contextmanager
    def _silence():
        """Suppress both Python-level and C-level stdout."""
        sys.stdout.flush()
        fd_out = _os.dup(1)           # save a copy of fd 1
        devnull = _os.open(_os.devnull, _os.O_WRONLY)
        _os.dup2(devnull, 1)          # point fd 1 → /dev/null
        _os.close(devnull)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                yield
        finally:
            sys.stdout.flush()
            _os.dup2(fd_out, 1)       # restore fd 1
            _os.close(fd_out)

    maps   = args.maps[:args.grid_maps_count]
    images = (args.images or [None] * len(args.maps))[:args.grid_maps_count]
    repeats   = args.grid_repeats
    max_steps = args.grid_max_steps

    combos = list(itertools.product(GRID_MIN_CLEARANCE, GRID_ALPHA, GRID_FORWARD_SAFETY, GRID_ENTROPY_EPS))
    total_tasks = len(combos) * len(maps)
    print(f"Grid search: {len(combos)} combinations × {len(maps)} maps × {repeats} repeats × up to {max_steps} steps/run")

    rows = []  # one row per (combo, map)
    task_idx = 0
    t_start = time.time()

    def _progress(done, total, label):
        bar_w = 30
        filled = int(bar_w * done / total) if total else 0
        bar = "█" * filled + "░" * (bar_w - filled)
        elapsed = time.time() - t_start
        eta_str = ""
        if done > 0:
            eta = elapsed / done * (total - done)
            m, s = divmod(int(eta), 60)
            h, m = divmod(m, 60)
            eta_str = f"  ETA {h:02d}:{m:02d}:{s:02d}"
        print(f"\r[{bar}] {done}/{total}{eta_str}  {label}", end="", flush=True)

    for combo_idx, (min_cl, alpha, fsm, eps) in enumerate(combos):
        # Re-seed before each combo so results are comparable across combinations.
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)

        for i, map_yaml in enumerate(maps):
            label = f"clr={min_cl} α={alpha} fsm={fsm} ε={eps} | {os.path.basename(map_yaml)}"
            _progress(task_idx, total_tasks, label)
            img = images[i] if i < len(images) else None
            with _silence():
                res = run_on_map(
                    module, map_yaml, img,
                    repeats=repeats,
                    max_steps=max_steps,
                    threshold=args.threshold,
                    seed=args.seed,
                    visualize=args.visualize,
                    random_move=False,
                    min_clearance=min_cl,
                    alpha=alpha,
                    forward_safety_margin=fsm,
                    entropy_eps=eps,
                )
            task_idx += 1
            counts   = {o: sum(1 for r in res if r['outcome'] == o) for o in OUTCOMES}
            correct  = [r['distance'] for r in res if r['converged']]
            avg_dist = statistics.mean(correct) if correct else float('nan')
            avg_str  = f"{avg_dist:.3f}" if not math.isnan(avg_dist) else "nan"
            _progress(task_idx, total_tasks, f"{label} → correct={counts['correct']} collision={counts['collision']} avg_dist={avg_str}")
            print()  # newline after each completed task
            row = {
                'min_clearance': min_cl,
                'alpha': alpha,
                'forward_safety_margin': fsm,
                'entropy_eps': eps,
                'map': map_yaml,
                **{o: counts[o] for o in OUTCOMES},
                'avg_distance_correct': avg_dist,
                'n': len(res),
            }
            rows.append(row)
            avg_str = f"{avg_dist:.3f}" if not math.isnan(avg_dist) else "nan"
            print(f"  {map_yaml}: {' '.join(f'{o}={counts[o]}' for o in OUTCOMES)} | avg_dist={avg_str}")

    # Write per-(combo, map) rows first so results survive any error in the summary below.
    if args.out:
        with open(args.out, 'w', newline='') as fh:
            fieldnames = ['min_clearance', 'alpha', 'forward_safety_margin', 'entropy_eps', 'map', *OUTCOMES, 'avg_distance_correct', 'n']
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote grid search results to {args.out}")

    # Aggregate across maps and print sorted by correct desc, collision asc
    from collections import defaultdict
    agg = defaultdict(lambda: {o: 0 for o in OUTCOMES})
    agg_dist  = defaultdict(list)
    agg_n     = defaultdict(int)
    for row in rows:
        key = (row['min_clearance'], row['alpha'], row['forward_safety_margin'], row['entropy_eps'])
        for o in OUTCOMES:
            agg[key][o] += row[o]
        if not math.isnan(row['avg_distance_correct']):
            agg_dist[key].append(row['avg_distance_correct'])
        agg_n[key] += row['n']

    print("\n=== Grid Search Summary (sorted by correct desc, collision asc) ===")
    header = f"{'min_cl':>6} {'alpha':>5} {'fsm':>5} {'eps':>6} | {'correct':>7} {'incorrect':>9} {'timeout':>7} {'stuck':>5} {'collision':>9} | avg_dist"
    print(header)
    print("-" * len(header))
    sorted_keys = sorted(
        agg.keys(),
        key=lambda k: (-agg[k]['correct'], agg[k]['collision']),
    )
    for key in sorted_keys:
        c = agg[key]
        avg = statistics.mean(agg_dist[key]) if agg_dist[key] else float('nan')
        avg_str = f"{avg:.3f}" if not math.isnan(avg) else "  nan"
        stuck = c['stuck_no_step'] + c['stuck_chance'] + c['stuck_no_entropy']
        print(f"{key[0]:>6} {key[1]:>5} {key[2]:>5} {key[3]:>6} | {c['correct']:>7} {c['incorrect']:>9} {c['timeout']:>7} {stuck:>5} {c['collision']:>9} | {avg_str}")


def main():
    global module
    module = load_demo_module()
    args = parse_args()

    if args.grid_search:
        run_grid_search(module, args)
        return

    maps = args.maps
    images = args.images or [None] * len(maps)

    all_summary = {}
    for i, map_yaml in enumerate(maps):
        img = images[i] if i < len(images) else None
        print(f"Running map: {map_yaml} repeats={args.repeats}")
        # run experiments with best-movement selection
        res_best = run_on_map(module, map_yaml, img, repeats=args.repeats, max_steps=args.max_steps, threshold=args.threshold, seed=args.seed, visualize=args.visualize, random_move=False, min_clearance=args.min_clearance, alpha=args.alpha, forward_safety_margin=args.forward_safety_margin, entropy_eps=args.entropy_eps)
        # run experiments with random-movement selection
        res_rand = run_on_map(module, map_yaml, img, repeats=args.repeats, max_steps=args.max_steps, threshold=args.threshold, seed=args.seed, visualize=False, random_move=True, min_clearance=args.min_clearance, alpha=args.alpha, forward_safety_margin=args.forward_safety_margin, entropy_eps=args.entropy_eps)

        def summarize(res):
            distances = [r['distance'] for r in res if r['converged']]
            counts = {o: sum(1 for r in res if r['outcome'] == o) for o in OUTCOMES}
            avg_distance = statistics.mean(distances) if distances else float('nan')
            return {'results': res, 'counts': counts, 'converged': counts['correct'], 'avg_distance': avg_distance}

        def format_counts(counts):
            return " ".join(f"{o}={counts[o]}" for o in OUTCOMES)

        sum_best = summarize(res_best)
        sum_rand = summarize(res_rand)

        all_summary[map_yaml] = {'best': sum_best, 'random': sum_rand}

        print(f"Map {map_yaml} (best): {format_counts(sum_best['counts'])} / {len(res_best)}; avg distance (correct) = {sum_best['avg_distance']:.3f}")
        print(f"Map {map_yaml} (random): {format_counts(sum_rand['counts'])} / {len(res_rand)}; avg distance (correct) = {sum_rand['avg_distance']:.3f}")

        print("\nCurrent results so far:")
        for current_map, info in all_summary.items():
            for mode in ('best', 'random'):
                summary = info[mode]
                total = len(summary['results'])
                avg = summary['avg_distance']
                avg_str = f"{avg:.3f}" if not math.isnan(avg) else "nan"
                print(f"  Map: {current_map} | Mode: {mode} | {format_counts(summary['counts'])} / {total} | Avg distance (correct): {avg_str}")

    if args.out:
        import csv
        with open(args.out, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['map', 'mode', 'repeat', 'outcome', 'distance', 'steps'])
            for m, info in all_summary.items():
                for mode in ('best', 'random'):
                    for idx, r in enumerate(info[mode]['results']):
                        w.writerow([m, mode, idx, r['outcome'], f"{r['distance']:.6f}", r['steps']])

        root, ext = os.path.splitext(args.out)
        summary_path = f"{root}_summary{ext or '.csv'}"
        with open(summary_path, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['map', 'mode', 'total', *OUTCOMES, 'avg_distance_correct'])
            for m, info in all_summary.items():
                for mode in ('best', 'random'):
                    summary = info[mode]
                    w.writerow([m, mode, len(summary['results']),
                                *(summary['counts'][o] for o in OUTCOMES),
                                f"{summary['avg_distance']:.6f}"])
        print(f"Wrote per-run results to {args.out} and counts to {summary_path}")
    # Print detailed human-readable summary of all statistics
    print("\n=== Final Experiment Summary ===")
    for m, info in all_summary.items():
        print(f"\nMap: {m}")
        for mode in ('best', 'random'):
            summary = info[mode]
            total = len(summary['results'])
            avg = summary['avg_distance']
            print(f"  Mode: {mode}")
            for o in OUTCOMES:
                print(f"    {o}: {summary['counts'][o]}/{total}")
            print(f"    Average distance (correct runs): {avg if not (avg!=avg) else 'nan'}")
            print(f"    Per-run details:")
            for i, r in enumerate(summary['results']):
                print(f"      - Repeat {i}: outcome={r['outcome']}, distance={r['distance']:.6f}, steps={r['steps']}")

    print("\nDone.")


if __name__ == '__main__':
    main()

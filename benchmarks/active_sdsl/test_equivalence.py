#!/usr/bin/env python3
"""Check that the optimized examples (working tree) are bit-identical to the original ones
at common.ORIG_REF (328423d by default; override with SDSL_ORIG_REF).

Compares, on every room-annotated map:
  1. map-static helpers: room_distance_lookup, get_room_at_point (every pixel + out of
     bounds), point_to_room_edge_distance, min_distance_to_neighbor_rooms
  2. simulate_movements on real localization outputs (uniform + random beliefs, alpha
     variants), field by field with exact equality, and compute_metrics=False
  3. lock-step active episodes: identical localize inputs are fed to both variants and
     every simulate result / chosen move must match exactly (incl. the driver's CC reuse)

Usage: python3 benchmarks/active_sdsl/test_equivalence.py [--quick]
"""
import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  MISMATCH:", msg)


def vox_t(v):
    return (v.bottom_left[0], v.bottom_left[1], v.bottom_left[2], v.top_right[0], v.top_right[1], v.top_right[2])


def same_val(a, b):
    """Exact equality including the Python type (np.float64 vs float matters downstream)."""
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b) and type(a) is type(b)
    return type(a) is type(b) and a == b


def compare_sims(ro, rp, label, metrics=True):
    check(len(ro) == len(rp), f"{label}: n sims {len(ro)} vs {len(rp)}")
    for i, (a, b) in enumerate(zip(ro, rp)):
        for k in a:
            if not metrics and k in ("entropy", "avg_neighbor_room_distance"):
                check(b[k] is None, f"{label}[{i}].{k} should be None when compute_metrics=False")
                continue
            va, vb = a[k], b[k]
            if k in ("surviving_voxels", "cleaned_voxels"):
                check([vox_t(v) for v in va] == [vox_t(v) for v in vb], f"{label}[{i}].{k}")
            elif isinstance(va, list):
                check(len(va) == len(vb) and all(same_val(x, y) for x, y in zip(va, vb)), f"{label}[{i}].{k}")
            else:
                check(same_val(va, vb), f"{label}[{i}].{k}: {va!r} vs {vb!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.chdir(C.REPO)
    import sdsl
    dmo, CMo = C.load_active_modules("orig")
    dmp, CMp = C.load_active_modules("opt")
    assert CMo is not CMp
    maps = list(C.MAPS)[:1] if args.quick else list(C.MAPS)
    t_start = time.time()
    for m in maps:
        print(f"== {m}")
        pgm, env, yaml_path, img_path = C.load_env(m)
        mo = CMo(env, yaml_path=yaml_path, image_path=img_path)
        mp = CMp(env, yaml_path=yaml_path, image_path=img_path)

        # 1. static helpers
        check(mo.room_distance_lookup == mp.room_distance_lookup, f"{m}: room_distance_lookup")
        h, w = mo.pixels.shape[:2]
        pts = [(r, c) for r in range(-2, h + 2) for c in range(-2, w + 2)]
        mism = sum(mo.get_room_at_point(p) != mp.get_room_at_point(p) for p in pts)
        check(mism == 0, f"{m}: get_room_at_point mismatches {mism}/{len(pts)}")
        rng = np.random.default_rng(0)
        sample = [(int(rng.integers(-3, h + 3)), int(rng.integers(-3, w + 3))) for _ in range(40 if args.quick else 120)]
        rooms = list(mo.room_colors)
        for p in sample:
            for r in rooms:
                a, b = mo.point_to_room_edge_distance(p, r), mp.point_to_room_edge_distance(p, r)
                check(a == b and type(a[1]) is type(b[1]), f"{m}: edge dist {p} {r}: {a} vs {b}")
        for visited in (set(), set(rooms[:-1]), set(rooms)):
            for mv in (mo, mp):
                mv.reset_visited_flags()
                for r in visited:
                    mv.mark_node_visited(r)
            for p in sample:
                a = mo.min_distance_to_neighbor_rooms(p, mo.room_graph)
                b = mp.min_distance_to_neighbor_rooms(p, mp.room_graph)
                check(a == b, f"{m}: min_dist {p} visited={len(visited)}: {a} vs {b}")
        print(f"  static helpers ok ({len(pts)} pixels, {len(sample) * len(rooms)} edge-dist pairs)")

        # 2. simulate_movements on real localization outputs
        bbox = env.bounding_box()
        poses = C.sample_free_poses(pgm, env, 3 if args.quick else 10, np.random.default_rng(7))
        nsim = 0
        for pi, (x, y, z) in enumerate(poses):
            angles = np.linspace(z, 2 * np.pi + z, 16, endpoint=False)
            dists = np.array([env.measure_distance(sdsl.R3(x, y, float(t))) for t in angles])
            odo = [sdsl.R3(0.0, 0.0, float(t - z)) for t in angles]
            pred = sdsl.Predicate_Fwd2D_Arr(env, odo, list(dists), 0.7, 0.015)
            with C.capture_native_stdout():
                vox = sdsl.cleanup_SE2(sdsl.localize_omp_forkjoin_3d(bbox, pred, 10, timeout=0.0, verbose=False))
            if not vox:
                continue
            vox = vox[:40] if args.quick else vox[:150]
            beliefs_sets = [[1.0 / len(vox)] * len(vox), list(rng.dirichlet(np.ones(len(vox)) * 0.3))]
            for bi, bel in enumerate(beliefs_sets):
                for alpha in (0.05, 0.3, 0.0):
                    for mv in (mo, mp):
                        mv.reset_visited_flags()
                    ro = mo.simulate_movements(angles, dists, vox, bel, alpha=alpha)
                    rp = mp.simulate_movements(angles, dists, vox, bel, alpha=alpha)
                    compare_sims(ro, rp, f"{m}/pose{pi}/bel{bi}/alpha{alpha}")
                    nsim += len(ro)
            for mv in (mo, mp):
                mv.reset_visited_flags()
            ro = mo.simulate_movements(angles, dists, vox, beliefs_sets[0])
            rp = mp.simulate_movements(angles, dists, vox, beliefs_sets[0], compute_metrics=False)
            compare_sims(ro, rp, f"{m}/pose{pi}/no-metrics", metrics=False)
        print(f"  simulate_movements ok ({nsim} accepted moves compared)")

        # 3. lock-step active episode (fixed timeout -> deterministic localization)
        dmo.TIMEOUT = 0.0
        for mv in (mo, mp):
            mv.reset_visited_flags()
        random.seed(3)
        x, y = poses[0][0], poses[0][1]
        z = random.uniform(0, 2 * math.pi)
        state = {"prev_voxels": None, "prev_belief": None, "prev_pos": None, "x": x, "y": y, "z": z}
        steps = 3 if args.quick else 8
        for step in range(steps):
            with C.capture_native_stdout():
                angles, dists, voxels, belief = dmp.localize(env, state["x"], state["y"], state["z"], state)
            if belief:
                cc_orig = sdsl.connectedComponentsWithBeliefs(voxels, True, belief)
                cc_cached = state["cc"]
                check([vox_t(v) for v in cc_orig[0]] == [vox_t(v) for v in cc_cached[0]]
                      and list(cc_orig[1]) == list(cc_cached[1]), f"{m}/ep step{step}: cached CC differs")
            ro = mo.simulate_movements(angles, dists, voxels, belief)
            rp = mp.simulate_movements(angles, dists, voxels, belief)
            compare_sims(ro, rp, f"{m}/ep step{step}")

            def score(s):
                e, a = s.get("entropy"), s.get("avg_neighbor_room_distance")
                return float("inf") if e is None or a is None else e + a
            best = min(rp, key=score, default=None)
            if best is None:
                break
            step_len = float(np.minimum(np.maximum(best["distance"] - 0.15, 0), 0.3))
            state["x"] += step_len * math.cos(best["angle"] + state["z"])
            state["y"] += step_len * math.sin(best["angle"] + state["z"])
            state["z"] = (best["angle"] + state["z"]) % (2 * math.pi)
        print(f"  lock-step episode ok ({step + 1} steps)")
    print(f"\n{'FAILED' if FAILS else 'PASSED'}: {len(FAILS)} mismatches  ({time.time() - t_start:.0f}s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

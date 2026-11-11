"""Benchmark tasks. Each task is ``fn(ctx) -> dict`` (JSON-serialisable)."""
from __future__ import annotations

import cProfile
import io
import json
import math
import os
import pstats
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import common as C

# ---------------------------------------------------------------------------
# Parameters shared by every host (so results are comparable)
# ---------------------------------------------------------------------------
PARAMS = {
    "seed": 12345,
    "n_rays": 16,
    "kk": 0.7,
    "err": 0.015,
    "env_query_poses": 500,
    "loc_poses_per_map": 4,
    "loc_depths": [4, 5, 6, 7, 8, 9, 10],
    "thread_maps": ["symmetry_2", "apt"],
    "thread_poses": 3,
    "thread_depth": 10,
    "belief_ns": [100, 200, 400, 800, 1600, 3200, 6400],
    "sim_ns": [10, 25, 50, 100, 200],
    # The RPi runs a PREFIX of the Mac's episodes (same seeds -> same first repeats/steps);
    # the report compares hosts on that common subset.
    "episodes": {
        "deployed": {"timeout": 1.0, "repeats": 3, "max_steps": 15,
                     "override": {"rpi": {"repeats": 2, "max_steps": 10}}},
        # Fixed work (no timeout, depth 10). On the apartment map one step can take
        # >1 min on the M1 and hours on the RPi (quadratic split), so the RPi skips it.
        "fixed":    {"timeout": 0.0, "repeats": 2, "max_steps": 10, "skip_maps": {"rpi": ["apt"]},
                     "override": {"rpi": {"repeats": 1, "max_steps": 8}}},
    },
    "profile_max_steps": 8,
    "threshold": 0.55,
}

if os.environ.get("BENCH_QUICK"):
    PARAMS.update({
        "env_query_poses": 50, "loc_poses_per_map": 1, "loc_depths": [4, 6, 8], "thread_poses": 1,
        "thread_depth": 7, "thread_maps": ["symmetry_2"], "belief_ns": [100, 200, 400], "sim_ns": [10, 25],
        "episodes": {"deployed": {"timeout": 1.0, "repeats": 1, "max_steps": 2},
                     "fixed": {"timeout": 0.0, "repeats": 1, "max_steps": 2}},
        "profile_max_steps": 2,
    })
    C.MAPS = {k: C.MAPS[k] for k in ("symmetry_2",)}

# Per-host safety budgets (seconds). A single call exceeding the budget stops
# the corresponding sweep from going larger (recorded as "skipped_beyond").
BUDGETS = {
    "mac": {"call": 60, "episode": 400, "kernel": 20},
    "rpi": {"call": 300, "episode": 2400, "kernel": 120},
}


def _odometry_and_measurements(env, x, y, z, n_rays):
    import sdsl
    angles = np.linspace(z, 2 * np.pi + z, n_rays, endpoint=False)
    dists = [env.measure_distance(sdsl.R3(x, y, float(t))) for t in angles]
    odo = [sdsl.R3(0.0, 0.0, float(t - z)) for t in angles]
    return angles, dists, odo


def _in_any_voxel(voxels, q):
    for v in voxels:
        bl, tr = v.bottom_left, v.top_right
        th = q[2] % (2 * math.pi)
        if bl[0] <= q[0] <= tr[0] and bl[1] <= q[1] <= tr[1] and bl[2] <= th <= tr[2]:
            return True
    return False


# ---------------------------------------------------------------------------
# 1. Environment query throughput
# ---------------------------------------------------------------------------

def task_env_query(ctx):
    import sdsl
    out = {}
    maps = list(C.MAPS)
    for mi, m in enumerate(maps):
        pgm, env, _, _ = C.load_env(m)
        rng = np.random.default_rng(PARAMS["seed"])
        poses = C.sample_free_poses(pgm, env, PARAMS["env_query_poses"], rng)
        qs = [sdsl.R3(*p) for p in poses]
        q2 = [sdsl.R3(p[0] + 0.3 * math.cos(p[2]), p[1] + 0.3 * math.sin(p[2]), p[2]) for p in poses]
        vox = [sdsl.Voxel_R3(sdsl.R3(p[0] - 0.05, p[1] - 0.05, 0.0), sdsl.R3(p[0] + 0.05, p[1] + 0.05, 0.1))
               for p in poses]
        n = len(qs)
        res = {}

        def per_call(fn):
            spc, calls = C.timeit_calls(fn, min_time=0.3)
            return spc / n

        res["measure_distance_us"] = 1e6 * per_call(lambda: [env.measure_distance(q) for q in qs])
        res["contains_us"] = 1e6 * per_call(lambda: [env.contains(q) for q in qs])
        res["collision_detection_us"] = 1e6 * per_call(
            lambda: [env.collision_detection(a, b) for a, b in zip(qs, q2)])
        res["intersects_us"] = 1e6 * per_call(lambda: [env.intersects(v) for v in vox])
        res["R3_ctor_us"] = 1e6 * per_call(lambda: [sdsl.R3(p[0], p[1], p[2]) for p in poses])
        res["map_px"] = [pgm.width, pgm.height]
        out[m] = res
        ctx.tracker.sub((mi + 1) / len(maps), m)
        ctx.tracker.result(f"env_query:{m}",
                           f"measure_distance {res['measure_distance_us']:.2f}µs, contains {res['contains_us']:.2f}µs, "
                           f"collision {res['collision_detection_us']:.1f}µs")
    labels = maps
    ctx.tracker.plot_bar("env_query", [[out[m]["measure_distance_us"], out[m]["contains_us"],
                                        out[m]["collision_detection_us"]] for m in labels], labels,
                         "env query cost [µs/call]", legend=["measure_distance", "contains", "collision_detection"])
    return out


# ---------------------------------------------------------------------------
# 2. Localization cost vs recursion depth (no timeout = fixed work)
# ---------------------------------------------------------------------------

def task_localize_depth(ctx):
    import sdsl
    budget = BUDGETS[ctx.host_kind]["call"]
    out = {}
    maps = list(C.MAPS)
    total = len(maps) * PARAMS["loc_poses_per_map"]
    k = 0
    for m in maps:
        pgm, env, _, _ = C.load_env(m)
        bbox = env.bounding_box()
        rng = np.random.default_rng(PARAMS["seed"] + 1)
        poses = C.sample_free_poses(pgm, env, PARAMS["loc_poses_per_map"], rng)
        recs = []
        for p in poses:
            _, dists, odo = _odometry_and_measurements(env, *p, PARAMS["n_rays"])
            pred = sdsl.Predicate_Fwd2D_Arr(env, odo, dists, PARAMS["kk"], PARAMS["err"])
            for depth in PARAMS["loc_depths"]:
                with C.capture_native_stdout() as cap:
                    t0 = time.perf_counter()
                    vox = sdsl.localize_omp_forkjoin_3d(bbox, pred, depth, timeout=0.0, verbose=True)
                    t1 = time.perf_counter()
                    clean = sdsl.cleanup_SE2(vox)
                    t2 = time.perf_counter()
                log = C.parse_localize_log(cap["text"])
                rec = {"pose": p, "depth": depth, "localize_s": t1 - t0, "cleanup_s": t2 - t1,
                       "n_raw": len(vox), "n_clean": len(clean),
                       "iter_counts": log[0]["iter_counts"] if log else None,
                       "threads": log[0]["threads"] if log else None,
                       "gt_covered": _in_any_voxel(vox, p)}
                recs.append(rec)
                if t2 - t0 > budget:
                    recs.append({"pose": p, "skipped_beyond_depth": depth, "reason": f"call > {budget}s"})
                    ctx.tracker.log(f"localize_depth {m}: depth {depth} took {t2 - t0:.1f}s, skipping deeper")
                    break
            k += 1
            ctx.tracker.sub(k / total, f"{m} pose {k}")
        out[m] = recs
        by_d = {}
        for r in recs:
            if "depth" in r:
                by_d.setdefault(r["depth"], []).append(r["localize_s"] + r["cleanup_s"])
        ds = sorted(by_d)
        ctx.tracker.plot_line("localize_depth", ds, [float(np.median(by_d[d])) for d in ds], m,
                              "localize+cleanup median [s] vs depth (timeout=0)", "depth", "s", log_y=True,
                              overview=True)
        d10 = by_d.get(10)
        ctx.tracker.result(f"localize_depth:{m}",
                           f"depth10 median {np.median(d10):.3f}s" if d10 else "depth 10 not reached")
    return out


# ---------------------------------------------------------------------------
# 3. OpenMP thread scaling (subprocess per thread count)
# ---------------------------------------------------------------------------

def worker_threads(args):
    """Runs inside a subprocess with OMP_NUM_THREADS set."""
    import sdsl
    res = []
    for m in PARAMS["thread_maps"]:
        pgm, env, _, _ = C.load_env(m)
        bbox = env.bounding_box()
        rng = np.random.default_rng(PARAMS["seed"] + 2)
        poses = C.sample_free_poses(pgm, env, PARAMS["thread_poses"], rng)
        for p in poses:
            _, dists, odo = _odometry_and_measurements(env, *p, PARAMS["n_rays"])
            pred = sdsl.Predicate_Fwd2D_Arr(env, odo, dists, PARAMS["kk"], PARAMS["err"])
            ts = []
            for rep in range(2):
                with C.capture_native_stdout() as cap:
                    t0 = time.perf_counter()
                    vox = sdsl.localize_omp_forkjoin_3d(bbox, pred, PARAMS["thread_depth"], timeout=0.0, verbose=True)
                    ts.append(time.perf_counter() - t0)
            log = C.parse_localize_log(cap["text"])
            res.append({"map": m, "pose": p, "s": min(ts), "n_raw": len(vox),
                        "threads_reported": log[0]["threads"] if log else None,
                        "iter_counts": log[0]["iter_counts"] if log else None})
    return res


def task_threads(ctx):
    ncpu = os.cpu_count() or 4
    counts = sorted(set([1, 2, 4] + ([3] if ncpu == 4 else []) + ([6, 8] if ncpu >= 8 else [])))
    counts = [c for c in counts if c <= ncpu]
    out = {}
    for i, k in enumerate(counts):
        env = dict(os.environ, OMP_NUM_THREADS=str(k))
        cmd = [sys.executable, str(C.HERE / "run_all.py"), "--worker", "threads"]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(C.HERE))
        if p.returncode != 0:
            out[str(k)] = {"error": p.stderr[-2000:]}
            ctx.tracker.log(f"threads={k} failed: {p.stderr[-300:]}")
            continue
        recs = json.loads(p.stdout.strip().splitlines()[-1])
        out[str(k)] = recs
        tot = sum(r["s"] for r in recs)
        ctx.tracker.sub((i + 1) / len(counts), f"threads={k}")
        base = sum(r["s"] for r in out[str(counts[0])]) if "error" not in out[str(counts[0])] else tot
        ctx.tracker.plot_line("threads", [k], [base / tot], "speedup", "localize depth-10 speedup vs OMP threads",
                              "threads", "speedup ×", overview=True)
        ctx.tracker.result(f"threads={k}", f"sum {tot:.3f}s, speedup {base / tot:.2f}×")
    return {"counts": counts, "runs": out}


# ---------------------------------------------------------------------------
# 4. AlwaysTrue split: demonstrates the O(N^2) reserve pattern in Voxel::split
# ---------------------------------------------------------------------------

def task_split_alwaystrue(ctx):
    import sdsl
    out = []
    v = sdsl.Voxel_R3(sdsl.R3(0, 0, 0), sdsl.R3(1, 1, 1))
    p = sdsl.Predicate_AlwaysTrue_3d()
    for i, d in enumerate([2, 3, 4, 5]):
        ts = []
        for _ in range(3):
            with C.capture_native_stdout():
                t0 = time.perf_counter()
                r = sdsl.localize_omp_forkjoin_3d(v, p, d, timeout=0.0, verbose=False)
                ts.append(time.perf_counter() - t0)
        out.append({"depth": d, "n_out": len(r), "n_split_last": len(r) * 8, "s": min(ts)})
        ctx.tracker.sub((i + 1) / 4)
    out.append({"depth": 6, "skipped": True,
                "reason": "depth 6 (262k voxels generated by the final split) did not finish in >200 s on the M1 "
                          "because Voxel::split's exact-size reserve makes the split loop O(N^2)"})
    ctx.tracker.result("split_alwaystrue", " ".join(f"d{r['depth']}:{r['s'] * 1e3:.1f}ms" for r in out if "s" in r))
    return out


# ---------------------------------------------------------------------------
# 5. Belief-layer kernel scaling (cleanup / connected components / fusion / entropy)
# ---------------------------------------------------------------------------

def _big_voxel_set(target):
    """Collect real pre-cleanup localization voxels until we have >= target."""
    import sdsl
    pool = []
    for m in [m for m in ("apt", "plus", "symmetry_2", "square") if m in C.MAPS]:
        pgm, env, _, _ = C.load_env(m)
        bbox = env.bounding_box()
        rng = np.random.default_rng(PARAMS["seed"] + 3)
        for p in C.sample_free_poses(pgm, env, 6, rng):
            _, dists, odo = _odometry_and_measurements(env, *p, PARAMS["n_rays"])
            pred = sdsl.Predicate_Fwd2D_Arr(env, odo, dists, PARAMS["kk"], PARAMS["err"])
            with C.capture_native_stdout():
                vox = sdsl.localize_omp_forkjoin_3d(bbox, pred, 9, timeout=0.0, verbose=False)
            pool.extend(vox)
            if len(pool) >= target:
                return pool[:target]
    return pool


def task_belief_scaling(ctx):
    import sdsl
    budget = BUDGETS[ctx.host_kind]["kernel"]
    ns = PARAMS["belief_ns"]
    ctx.tracker.sub(0.0, "collecting voxels")
    pool = _big_voxel_set(max(ns))
    ns = [n for n in ns if n <= len(pool)]
    U = sdsl.R3(0.1, 0.02, 0.05)
    kernels = {
        "cleanup_SE2": lambda v, b: sdsl.cleanup_SE2(v),
        "connectedComponents(use3rdDim)": lambda v, b: sdsl.connectedComponentsWithBeliefs(v, True, b),
        "connectedComponents(xy)": lambda v, b: sdsl.connectedComponentsWithBeliefs(v, False, b),
        "fusion_2d(n x n)": lambda v, b: sdsl.fusion_2d(v, b, v, U, 0.015),
        "entropy_SE2": lambda v, b: sdsl.entropy_SE2(v, b),
    }
    out = {"pool_size": len(pool), "kernels": {}}
    stopped = set()
    steps = len(ns) * len(kernels)
    k = 0
    for n in ns:
        v = pool[:n]
        b = [1.0 / n] * n
        for name, fn in kernels.items():
            k += 1
            if name in stopped:
                continue
            reps, ts = 0, []
            t_start = time.perf_counter()
            while reps < 3 and (reps == 0 or time.perf_counter() - t_start < max(1.0, budget / 4)):
                t0 = time.perf_counter()
                fn(v, b)
                ts.append(time.perf_counter() - t0)
                reps += 1
            out["kernels"].setdefault(name, []).append({"n": n, "s": min(ts), "reps": reps})
            ctx.tracker.plot_line("belief_scaling", [n], [min(ts)], name, "belief kernels: time vs n",
                                  "n voxels", "s", log_x=True, log_y=True)
            if min(ts) > budget:
                stopped.add(name)
                out["kernels"][name].append({"n_skipped_beyond": n})
            ctx.tracker.sub(k / steps, f"{name} n={n}")
    for name, recs in out["kernels"].items():
        pts = [(r["n"], r["s"]) for r in recs if "s" in r and r["s"] > 1e-5]
        if len(pts) >= 3:
            x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
            slope = float(np.polyfit(x[-4:], y[-4:], 1)[0])
            out.setdefault("exponents", {})[name] = slope
            ctx.tracker.result(f"exponent:{name}", f"t ∝ n^{slope:.2f}; n={pts[-1][0]}: {pts[-1][1] * 1e3:.1f}ms")
    return out


# ---------------------------------------------------------------------------
# 6. Python-side overheads that dominate ChooseMovement's inner loops
# ---------------------------------------------------------------------------

def task_py_overhead(ctx):
    import sdsl
    v = sdsl.Voxel_R3(sdsl.R3(0, 0, 0), sdsl.R3(1, 1, 1))
    x = 0.3
    res = {}
    tests = {
        "sdsl.R3(x,y,z)": lambda: sdsl.R3(1.0, 2.0, 3.0),
        "Voxel_R3(bl,tr)": lambda: sdsl.Voxel_R3(sdsl.R3(0, 0, 0), sdsl.R3(1, 1, 1)),
        "voxel.midpoint()": lambda: v.midpoint(),
        "voxel.bottom_left[0]": lambda: v.bottom_left[0],
        "np.cos(scalar)": lambda: np.cos(x),
        "math.cos(scalar)": lambda: math.cos(x),
        "np.minimum(np.maximum(s,0),m)": lambda: float(np.minimum(np.maximum(x - 0.15, 0), 0.3)),
        "min(max(s,0),m)": lambda: min(max(x - 0.15, 0.0), 0.3),
    }
    for i, (name, fn) in enumerate(tests.items()):
        spc, n = C.timeit_calls(fn, min_time=0.25)
        res[name] = spc * 1e9
        ctx.tracker.sub((i + 1) / len(tests))
    ctx.tracker.result("py_overhead", ", ".join(f"{k} {v:.0f}ns" for k, v in list(res.items())[:4]))
    # conversion cost of passing voxel lists into C++
    for n in (100, 1000, 10000):
        vs = [v] * n
        b = [1.0 / n] * n
        spc, _ = C.timeit_calls(lambda: sdsl.entropy_SE2(vs, b), min_time=0.25)
        res[f"entropy_SE2 identical voxels n={n} (≈conversion) [ns/voxel]"] = spc * 1e9 / n
    return res


# ---------------------------------------------------------------------------
# 7. ChooseMovement building blocks + simulate_movements vs N
# ---------------------------------------------------------------------------

def task_choose_movement(ctx):
    import sdsl
    dm, ChooseMovement = C.load_active_modules(ctx.variant)
    out = {}
    maps = list(C.MAPS)
    for mi, m in enumerate(maps):
        pgm, env, yaml_path, img_path = C.load_env(m)
        t0 = time.perf_counter()
        mover = ChooseMovement(env, yaml_path=yaml_path, image_path=img_path)
        init_s = time.perf_counter() - t0
        rng = np.random.default_rng(PARAMS["seed"] + 4)
        poses = C.sample_free_poses(pgm, env, 20, rng)
        pix = [mover.world_to_pixel((p[0], p[1])) for p in poses]
        rooms = list(mover.room_colors)
        res = {"init_s": init_s, "n_rooms": len(rooms), "image_px": list(mover.pixels.shape[:2])}

        def pc(fn, n):
            spc, _ = C.timeit_calls(fn, min_time=0.3)
            return spc / n
        res["point_to_room_edge_distance_ms"] = 1e3 * pc(
            lambda: [mover.point_to_room_edge_distance(q, rooms[0]) for q in pix[:5]], 5)
        res["get_room_at_point_us"] = 1e6 * pc(lambda: [mover.get_room_at_point(q) for q in pix], len(pix))
        res["world_to_pixel_us"] = 1e6 * pc(lambda: [mover.world_to_pixel((p[0], p[1])) for p in poses], len(poses))
        # min distance: (a) current room unvisited -> instant, (b) all but one room visited -> BFS
        mover.reset_visited_flags()
        res["min_dist_unvisited_ms"] = 1e3 * pc(lambda: [mover.min_distance_to_neighbor_rooms(q, mover.room_graph)
                                                          for q in pix[:5]], 5)
        for r in rooms[:-1]:
            mover.mark_node_visited(r)
        res["min_dist_bfs_ms"] = 1e3 * pc(lambda: [mover.min_distance_to_neighbor_rooms(q, mover.room_graph)
                                                    for q in pix[:5]], 5)
        mover.reset_visited_flags()

        # simulate_movements vs N, from a real localization (deployed settings).
        # On the apartment map collecting inputs means several depth-10 localizations of
        # highly ambiguous poses (minutes on the M1, ~hours on the RPi) and on the M1 every
        # candidate move was rejected anyway (0 feasible sims), so the RPi skips it.
        p = poses[0]
        bbox = env.bounding_box()
        big = []
        collect_poses = [] if (m == "apt" and ctx.host_kind == "rpi") else poses[:8]
        for p in collect_poses:
            angles, dists, odo = _odometry_and_measurements(env, *p, PARAMS["n_rays"])
            pred = sdsl.Predicate_Fwd2D_Arr(env, odo, dists, PARAMS["kk"], PARAMS["err"])
            with C.capture_native_stdout():
                vox = sdsl.cleanup_SE2(sdsl.localize_omp_forkjoin_3d(bbox, pred, 10, timeout=0.0, verbose=False))
            big.extend(vox)
            if len(big) >= max(PARAMS["sim_ns"]):
                break
        angles, dists, _ = _odometry_and_measurements(env, *poses[0], PARAMS["n_rays"])
        sims = []
        for n in PARAMS["sim_ns"]:
            if n > len(big):
                break
            vs = big[:n]
            b = [1.0 / n] * n
            timer = C.StageTimer()
            mover.reset_visited_flags()
            C.wrap_mover(timer, mover)
            t0 = time.perf_counter()
            r = mover.simulate_movements(angles, np.asarray(dists), vs, b)
            dt = time.perf_counter() - t0
            for name in list(vars(mover)):
                if callable(getattr(mover, name)) and hasattr(getattr(mover, name), "__wrapped__"):
                    delattr(mover, name)
            sims.append({"n": n, "s": dt, "n_sims": len(r), "breakdown": timer.snapshot_and_reset()})
            if dt > BUDGETS[ctx.host_kind]["kernel"]:
                break
        res["simulate_vs_n"] = sims
        out[m] = res
        ctx.tracker.sub((mi + 1) / len(maps), m)
        if sims:
            ctx.tracker.plot_line("simulate_vs_n", [s["n"] for s in sims], [s["s"] for s in sims], m,
                                  f"simulate_movements [s] vs n voxels ({ctx.variant})", "n voxels", "s",
                                  overview=True)
        ctx.tracker.result(f"choose_movement:{m}",
                           f"edge-dist {res['point_to_room_edge_distance_ms']:.2f}ms/call, "
                           f"BFS min-dist {res['min_dist_bfs_ms']:.2f}ms/call, "
                           + (f"simulate n={sims[-1]['n']}: {sims[-1]['s']:.2f}s" if sims else ""))
    return out


# ---------------------------------------------------------------------------
# 8. End-to-end active episodes (faithful to experiment_movement.run_on_map)
# ---------------------------------------------------------------------------

STAGE_ORDER = [
    ("localize", "cast_rays"), ("localize", "Predicate_Fwd2D_Arr"), ("localize", "localize_omp_forkjoin_3d"),
    ("localize", "cleanup_SE2"), ("localize", "fusion_2d"), ("localize", "connectedComponentsWithBeliefs"),
    ("driver", "connectedComponentsWithBeliefs"),
    ("simulate", "mover._move_voxels_and_normalize_beliefs"), ("simulate", "connectedComponentsWithBeliefs"),
    ("simulate", "entropy_SE2"), ("simulate", "mover.min_distance_to_neighbor_rooms"),
    ("simulate", "mover.mark_voxels_visited"),
]


def run_episodes(ctx, mode, maps=None, profile=False, max_steps=None, repeats=None, tag=None):
    import sdsl
    cfg = PARAMS["episodes"][mode]
    skip = cfg.get("skip_maps", {}).get(ctx.host_kind, [])
    maps = [m for m in (maps or list(C.MAPS)) if m not in skip]
    ov = cfg.get("override", {}).get(ctx.host_kind, {})
    repeats = repeats or ov.get("repeats")
    max_steps = max_steps or ov.get("max_steps")
    if skip:
        ctx.tracker.log(f"{mode}: skipping maps {skip} on {ctx.host_kind} (see PARAMS)")
    repeats = repeats or cfg["repeats"]
    max_steps = max_steps or cfg["max_steps"]
    budget = BUDGETS[ctx.host_kind]["episode"]
    dm, ChooseMovement = C.load_active_modules(ctx.variant)
    dm.TIMEOUT = cfg["timeout"]
    timer = C.StageTimer()
    C.install_sdsl_timers(timer)
    orig_cast = dm.cast_rays
    dm.cast_rays = timer.wrap(orig_cast, "cast_rays")
    episodes = []
    total_units = len(maps) * repeats * max_steps
    units = 0
    profiles = {}
    try:
        for m in maps:
            pgm, raw_env, yaml_path, img_path = C.load_env(m)
            env = C.EnvProxy(raw_env) if profile else raw_env
            mover = ChooseMovement(env, yaml_path=yaml_path, image_path=img_path)
            C.wrap_mover(timer, mover)
            random.seed(PARAMS["seed"])
            np.random.seed(PARAMS["seed"])
            xmin, ymin = pgm.origin_x, pgm.origin_y
            xmax, ymax = xmin + pgm.width * pgm.resolution, ymin + pgm.height * pgm.resolution
            prof = cProfile.Profile() if profile else None
            for rep in range(repeats):
                start = None
                for _ in range(300):
                    x0, y0 = random.uniform(xmin, xmax), random.uniform(ymin, ymax)
                    if raw_env.contains(sdsl.R3(x0, y0, 0.0)):
                        start = (x0, y0)
                        break
                if start is None:
                    start = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)
                z = random.uniform(0.0, 2 * math.pi)
                state = {"prev_voxels": None, "prev_belief": None, "prev_pos": None,
                         "x": start[0], "y": start[1], "z": z}
                ep = {"map": m, "rep": rep, "mode": mode, "start": [start[0], start[1], z], "steps": [],
                      "converged": None, "truncated": False, "distance": 0.0}
                t_ep = time.perf_counter()
                timer.snapshot_and_reset()
                for step in range(max_steps):
                    rec = {"step": step, "pose": [state["x"], state["y"], state["z"]]}
                    if prof:
                        prof.enable()
                    with C.capture_native_stdout() as cap:
                        t0 = time.perf_counter()
                        with timer.context("localize"):
                            angles, dists, voxels, belief = dm.localize(env, state["x"], state["y"], state["z"], state)
                        t1 = time.perf_counter()
                        max_conf, inside = 0.0, False
                        if belief:
                            if ctx.variant == "opt" and state.get("cc"):
                                comps, sums = state["cc"]          # optimized driver reuses localize()'s CC
                            else:
                                comps, sums = sdsl.connectedComponentsWithBeliefs(voxels, True, belief)
                            if sums:
                                max_conf = max(sums)
                                if ctx.variant == "opt" and state.get("cc"):
                                    c2, s2 = comps, sums
                                else:
                                    # original state_is_in_dominant_belief_voxel recomputes CC
                                    c2, s2 = sdsl.connectedComponentsWithBeliefs(voxels, True, belief)
                                dv = c2[int(np.argmax(s2))]
                                inside = (dv.bottom_left[0] <= state["x"] <= dv.top_right[0]
                                          and dv.bottom_left[1] <= state["y"] <= dv.top_right[1])
                        t2 = time.perf_counter()
                        sims = None
                        if not (max_conf >= PARAMS["threshold"]):
                            with timer.context("simulate"):
                                sims = mover.simulate_movements(angles, dists, voxels, belief)
                        t3 = time.perf_counter()
                    if prof:
                        prof.disable()
                    log = C.parse_localize_log(cap["text"])
                    rec.update({"t_localize": t1 - t0, "t_driver": t2 - t1, "t_simulate": t3 - t2,
                                "t_total": t3 - t0, "n_vox": len(voxels), "max_conf": max_conf,
                                "inside": inside, "loc_log": log[0] if log else None,
                                "stages": timer.snapshot_and_reset()})
                    if profile:
                        rec["env_calls"] = dict(env.counts)
                        env.counts.clear()
                    if max_conf >= PARAMS["threshold"]:
                        ep["converged"] = bool(inside)
                        ep["steps"].append(rec)
                        units += max_steps - step
                        break
                    rec["n_sims"] = len(sims or [])

                    def score(s):
                        e, a = s.get("entropy"), s.get("avg_neighbor_room_distance")
                        return float("inf") if e is None or a is None else e + a
                    best = min(sims, key=score, default=None)
                    ep["steps"].append(rec)
                    units += 1
                    ctx.tracker.sub(units / total_units, f"{mode} {m} rep {rep} step {step}")
                    ctx.tracker.plot_line(f"ep_{mode}", [len(episodes) + step / max_steps], [rec["t_total"]], m,
                                          f"{mode}: step wall time [s] (x = episode index)", "episode", "s",
                                          log_y=True)
                    if best is None:
                        units += max_steps - step - 1
                        break
                    step_len = float(np.minimum(np.maximum(best["distance"] - 0.15, 0), 0.3))
                    dx = step_len * math.cos(best["angle"] + state["z"])
                    dy = step_len * math.sin(best["angle"] + state["z"])
                    ep["distance"] += math.hypot(dx, dy)
                    state["x"] += dx
                    state["y"] += dy
                    state["z"] = (best["angle"] + state["z"]) % (2 * math.pi)
                    if time.perf_counter() - t_ep > budget:
                        ep["truncated"] = True
                        units += max_steps - step - 1
                        ctx.tracker.log(f"episode {m}/{rep} truncated at step {step} (> {budget}s)")
                        break
                else:
                    pass
                ep["wall_s"] = time.perf_counter() - t_ep
                episodes.append(ep)
                st = [s["t_total"] for s in ep["steps"]]
                ctx.tracker.result(f"{tag or mode}:{m}#{rep}",
                                   f"{len(st)} steps, median step {np.median(st):.2f}s, max {max(st):.2f}s, "
                                   f"converged={ep['converged']}")
            if prof:
                s = io.StringIO()
                ps = pstats.Stats(prof, stream=s)
                pdir = ctx.out_dir / "profiles"
                pdir.mkdir(parents=True, exist_ok=True)
                ps.dump_stats(str(pdir / f"{m}.prof"))
                rows = []
                for (fname, line, func), (cc, nc, tt, ct, callers) in ps.stats.items():
                    rows.append({"func": f"{Path(fname).name}:{line}({func})", "ncalls": nc, "tottime": tt,
                                 "cumtime": ct})
                rows.sort(key=lambda r: -r["tottime"])
                profiles[m] = {"top_tottime": rows[:40],
                               "top_cumtime": sorted(rows, key=lambda r: -r["cumtime"])[:40]}
            for name in list(vars(mover)):
                if hasattr(getattr(mover, name), "__wrapped__"):
                    delattr(mover, name)
    finally:
        dm.cast_rays = orig_cast
        C.uninstall_sdsl_timers()
    _push_breakdown(ctx, episodes, tag or mode)
    return {"params": PARAMS["episodes"][mode], "episodes": episodes, "profiles": profiles}


def _push_breakdown(ctx, episodes, tag):
    maps = sorted({e["map"] for e in episodes})
    rows = []
    for m in maps:
        steps = [s for e in episodes if e["map"] == m for s in e["steps"]]
        row = []
        for c, n in STAGE_ORDER:
            vals = [s["stages"].get(f"{c}/{n}", {}).get("s", 0.0) for s in steps]
            row.append(float(np.mean(vals)) if vals else 0.0)
        rows.append(row)
    if rows:
        ctx.tracker.plot_bar(f"breakdown_{tag}", rows, maps, f"{tag}: mean per-step time by stage [s]",
                             legend=[f"{c}:{n.replace('mover.', '')}" for c, n in STAGE_ORDER], overview=True)


def task_episodes_deployed(ctx):
    return run_episodes(ctx, "deployed")


def task_episodes_fixed(ctx):
    return run_episodes(ctx, "fixed")


def task_profile(ctx):
    steps = PARAMS["profile_max_steps"] if ctx.host_kind == "mac" else min(4, PARAMS["profile_max_steps"])
    return run_episodes(ctx, "deployed", profile=True, repeats=1, max_steps=steps, tag="profile")


def worker_pyspy_episode(args):
    """Target for py-spy: one deployed episode on symmetry_2 and apt."""
    class _T:
        def sub(self, *a, **k): pass
        def plot_line(self, *a, **k): pass
        def plot_bar(self, *a, **k): pass
        def result(self, *a, **k): pass
        def log(self, *a, **k): pass

    class _Ctx:
        tracker = _T()
        host_kind = args.host_kind
        variant = args.variant
        out_dir = Path("/tmp")
    run_episodes(_Ctx(), "deployed", maps=[m for m in ("symmetry_2", "apt") if m in C.MAPS],
                 repeats=1, max_steps=6)
    return {"ok": True}


def task_pyspy(ctx):
    import shutil
    exe = shutil.which("py-spy")
    if not exe:
        return {"skipped": "py-spy not installed"}
    if sys.platform == "darwin" and os.geteuid() != 0:
        return {"skipped": "py-spy needs root on macOS"}
    out = {}
    pdir = ctx.out_dir / "profiles"
    pdir.mkdir(parents=True, exist_ok=True)
    for fmt, ext in (("flamegraph", "svg"), ("speedscope", "speedscope.json")):
        dest = pdir / f"pyspy_{ctx.variant}.{ext}"
        cmd = [exe, "record", "--native", "-r", "50", "-f", fmt, "-o", str(dest), "--",
               sys.executable, str(C.HERE / "run_all.py"), "--worker", "pyspy",
               "--host-kind", ctx.host_kind, "--variant", ctx.variant]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(C.HERE))
        out[fmt] = {"rc": p.returncode, "file": str(dest), "stderr": p.stderr[-1500:]}
        ctx.tracker.sub(0.5 if fmt == "flamegraph" else 1.0, fmt)
        if p.returncode != 0:
            break
    return out


TASKS = {
    "env_query": (task_env_query, 15),
    "py_overhead": (task_py_overhead, 10),
    "split_alwaystrue": (task_split_alwaystrue, 3),
    "localize_depth": (task_localize_depth, 120),
    "threads": (task_threads, 60),
    "belief_scaling": (task_belief_scaling, 90),
    "choose_movement": (task_choose_movement, 120),
    "profile": (task_profile, 120),
    "episodes_deployed": (task_episodes_deployed, 500),
    "episodes_fixed": (task_episodes_fixed, 400),
    "pyspy": (task_pyspy, 120),
}

#!/usr/bin/env python3
"""Run the active-SDSL benchmark suite on this machine.

Examples (from the repo root or this directory):
  python3 benchmarks/active_sdsl/run_all.py --host-kind mac
  python3 benchmarks/active_sdsl/run_all.py --host-kind rpi --only env_query,threads
  python3 benchmarks/active_sdsl/run_all.py --host-kind mac --variant opt --only choose_movement,episodes_deployed

Results go to benchmarks/active_sdsl/results/<host-kind>/<variant>/<task>.json and
progress is streamed to visdom (env sdsl_<host-kind>[_<variant>] and sdsl_overview).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if "--quick" in sys.argv:
    os.environ["BENCH_QUICK"] = "1"   # must be set before tasks.py is imported (also inherited by workers)
import common as C  # noqa: E402
import tasks as T  # noqa: E402

# Initial guess of how much slower each host is than the M1 (ETA only; self-corrects).
HOST_SLOWDOWN = {"mac": 1.0, "rpi": 12.0}


class Ctx:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-kind", choices=["mac", "rpi"], default="mac")
    ap.add_argument("--variant", choices=["orig", "opt"], default="orig")
    ap.add_argument("--only", default="", help="comma-separated task ids")
    ap.add_argument("--skip", default="", help="comma-separated task ids")
    ap.add_argument("--force", action="store_true", help="re-run tasks that already have results")
    ap.add_argument("--visdom-server", default="http://localhost")
    ap.add_argument("--visdom-port", type=int, default=8097)
    ap.add_argument("--no-visdom", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny functional smoke run (results/<host>_quick)")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    os.chdir(C.REPO)
    C.load_active_modules(args.variant)  # also decides whether matplotlib needs a stub

    if args.worker == "threads":
        print(json.dumps(T.worker_threads(args), default=C._json_default))
        return
    if args.worker == "pyspy":
        T.worker_pyspy_episode(args)
        return

    selected = [t for t in T.TASKS if (not args.only or t in args.only.split(","))
                and t not in args.skip.split(",")]
    out_dir = C.RESULTS / (args.host_kind + ("_quick" if args.quick else "")) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in selected if args.force or not (out_dir / f"{t}.json").exists()]
    label = args.host_kind if args.variant == "orig" else f"{args.host_kind}_{args.variant}"
    slow = HOST_SLOWDOWN[args.host_kind]
    tracker = __import__("progress").Tracker(
        label, [(t, T.TASKS[t][1] * slow) for t in todo], server=args.visdom_server,
        port=args.visdom_port, jsonl_path=out_dir / "events.jsonl", enabled=not args.no_visdom)

    info = C.host_info()
    C.save_json(out_dir / "host.json", {"host": info, "params": T.PARAMS, "budgets": T.BUDGETS[args.host_kind]})
    tracker.log(f"host: {info.get('cpu_model')} | {info.get('platform')} | py {info.get('python')} | "
                f"{info.get('pointer_bits')}-bit | sdsl {info.get('sdsl_version')} @ {info.get('git_head')}")
    skipped = [t for t in selected if t not in todo]
    if skipped:
        tracker.log(f"already have results for: {', '.join(skipped)} (use --force to re-run)")

    ctx = Ctx()
    ctx.tracker = tracker
    ctx.host_kind = args.host_kind
    ctx.variant = args.variant
    ctx.out_dir = out_dir
    for t in todo:
        tracker.begin(t)
        fn = T.TASKS[t][0]
        t0 = time.time()
        try:
            res = fn(ctx)
            C.save_json(out_dir / f"{t}.json", {"task": t, "seconds": time.time() - t0,
                                                "thermal_end": C.thermal_sample(), "result": res})
            tracker.end("done")
        except Exception as e:
            err = traceback.format_exc()
            C.save_json(out_dir / f"{t}.failed.json", {"task": t, "error": err})
            tracker.log(f"!! {t} failed: {e!r}\n{err[-1500:]}")
            tracker.end("failed")
    tracker.finish()


if __name__ == "__main__":
    main()

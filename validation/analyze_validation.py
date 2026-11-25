"""Pool the per-run CSVs written by run_validation.sh and compare candidates.

Usage: python validation/analyze_validation.py [validation_dir]
"""
import collections
import csv
import glob
import math
import os
import statistics
import sys

OUTCOMES = ("correct", "incorrect", "timeout", "stuck_no_step", "stuck_chance", "stuck_no_entropy", "collision")
STUCK = ("stuck_no_step", "stuck_chance", "stuck_no_entropy")

d = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
files = sorted(f for f in glob.glob(os.path.join(d, "*_seed*.csv")) if not f.endswith("_summary.csv"))
if not files:
    sys.exit(f"no per-run CSVs in {d}")

# (candidate, map, mode) -> list of run rows
runs = collections.defaultdict(list)
for f in files:
    cand = os.path.basename(f).split("_seed")[0]
    with open(f) as fh:
        for r in csv.DictReader(fh):
            runs[(cand, os.path.basename(r["map"]), r["mode"])].append(r)


def stats(rs):
    n = len(rs)
    c = collections.Counter(r["outcome"] for r in rs)
    dist = [float(r["distance"]) for r in rs if r["outcome"] == "correct"]
    steps = [int(r["steps"]) for r in rs if r["outcome"] == "correct"]
    p = c["correct"] / n if n else float("nan")
    se = math.sqrt(p * (1 - p) / n) if n else float("nan")
    return {
        "n": n, "correct": c["correct"], "p": p, "se": se,
        "incorrect": c["incorrect"], "timeout": c["timeout"],
        "stuck": sum(c[s] for s in STUCK), "collision": c["collision"],
        "dist": statistics.median(dist) if dist else float("nan"),
        "steps": statistics.median(steps) if steps else float("nan"),
    }


def line(label, s):
    return (f"{label:<26} correct {s['correct']:>3}/{s['n']:<3} ({s['p']:5.0%} ±{1.96 * s['se']:4.0%})"
            f"  incorrect {s['incorrect']:>3}  timeout {s['timeout']:>3}  stuck {s['stuck']:>3}"
            f"  collision {s['collision']:>3}  | median dist {s['dist']:5.2f} m, steps {s['steps']:5.1f}")


cands = sorted({k[0] for k in runs})
maps = sorted({k[1] for k in runs})
seeds_seen = sorted({os.path.basename(f).split("_seed")[1].split(".")[0] for f in files})
print(f"files: {len(files)}  candidates: {cands}  seeds: {seeds_seen}\n")

print("=== Overall (all maps pooled) ===")
for cand in cands:
    for mode in ("best", "random"):
        rs = [r for (c, m, md), v in runs.items() if c == cand and md == mode for r in v]
        print(line(f"{cand} {mode}", stats(rs)))
    print()

print("=== Per map ===")
for m in maps:
    print(m)
    for cand in cands:
        for mode in ("best", "random"):
            if (cand, m, mode) in runs:
                print("  " + line(f"{cand} {mode}", stats(runs[(cand, m, mode)])))
    print()
print("± is a 95% interval on the correct rate (normal approximation).")
print("median dist / steps are over correct runs only.")

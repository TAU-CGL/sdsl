#!/usr/bin/env python3
"""Build and run the native C++ kernel benchmark under several compiler-flag variants.

  python3 benchmarks/active_sdsl/run_native.py --host-kind mac
  python3 benchmarks/active_sdsl/run_native.py --host-kind rpi --cross-aarch64

Results: results/<host>/native/bench_<variant>[_t1].json  (_t1 = OMP_NUM_THREADS=1)
Progress is streamed to visdom (env sdsl_<host>_native).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from progress import Tracker  # noqa: E402

CPP = C.HERE / "cpp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-kind", choices=["mac", "rpi"], required=True)
    ap.add_argument("--cross-aarch64", action="store_true", help="try installing/using an aarch64 cross compiler (RPi)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--visdom-server", default="http://localhost")
    ap.add_argument("--visdom-port", type=int, default=8097)
    a = ap.parse_args()
    out = C.RESULTS / a.host_kind / "native"
    out.mkdir(parents=True, exist_ok=True)
    slow = 1.0 if a.host_kind == "mac" else 15.0
    if a.host_kind == "mac":
        variants = ["Os", "O3", "O3native", "O3native_noerrno"]
        runs = [(v, None) for v in variants] + [("Os", 1), ("O3native", 1)]
    else:  # the RPi is ~50-80x slower on these kernels: fewer variants, 1 repetition
        os.environ["SDSL_BENCH_REPS"] = "1"
        runs = [("Os", None), ("O3", None), ("O3native", None), ("Os", 1)]
    tasks = [("dump_inputs", 5 * slow), ("build", 60 * slow)]
    if a.cross_aarch64:
        tasks.insert(1, ("install_cross", 300))
        runs.append(("aarch64_O3native", None))
    tasks += [(f"run_{v}" + (f"_t{t}" if t else ""), (40 if t else 30) * slow * (0.2 if a.quick else 1)) for v, t in runs]
    tr = Tracker(f"{a.host_kind}_native", tasks, server=a.visdom_server, port=a.visdom_port,
                 jsonl_path=out / "events.jsonl")

    tr.begin("dump_inputs")
    subprocess.run([sys.executable, str(CPP / "dump_inputs.py"), str(CPP / "inputs")], check=True, cwd=str(C.REPO))
    tr.end()

    if a.cross_aarch64:
        tr.begin("install_cross")
        if shutil.which("aarch64-linux-gnu-g++"):
            tr.end("done", "already installed")
        else:
            p = subprocess.run("DEBIAN_FRONTEND=noninteractive apt-get install -y -q g++-aarch64-linux-gnu",
                               shell=True, capture_output=True, text=True, timeout=1800)
            tr.end("done" if p.returncode == 0 else "failed", p.stdout[-200:] + p.stderr[-300:])

    tr.begin("build")
    t0 = time.time()
    p = subprocess.run(["bash", str(CPP / "build.sh")], capture_output=True, text=True)
    (out / "build.log").write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        tr.log(p.stderr[-2000:])
        tr.end("failed")
        tr.finish()
        sys.exit(1)
    tr.end("done", f"{time.time() - t0:.0f}s")

    for v, threads in runs:
        tid = f"run_{v}" + (f"_t{threads}" if threads else "")
        tr.begin(tid)
        exe = CPP / "bin" / f"bench_{v}"
        if not exe.exists():
            tr.end("skipped", "binary not built")
            continue
        env = dict(os.environ)
        if threads:
            env["OMP_NUM_THREADS"] = str(threads)
        dest = out / (f"bench_{v}" + (f"_t{threads}" if threads else "") + ".json")
        cmd = [str(exe), str(CPP / "inputs"), str(dest)] + (["quick"] if a.quick else [])
        t0 = time.time()
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        lines = []
        n_expected = (1 if a.quick else 16) + 6
        for line in proc.stderr:
            lines.append(line.rstrip())
            tr.sub(min(len(lines) / n_expected, 0.99), line.strip()[:80])
        proc.wait()
        (out / (dest.stem + ".log")).write_text("\n".join(lines))
        if proc.returncode == 0 and dest.exists():
            d = json.loads(dest.read_text())
            both = [x for x in d["localize"] if x.get("t_orig", -1) > 0]
            tot = sum(x["t_orig"] for x in both)
            fixed = sum(x["t_fixed_all"] for x in both)
            tr.end("done", f"localize sum orig {tot:.2f}s → fixed {fixed:.2f}s ({tot / max(fixed, 1e-9):.1f}×)")
        else:
            tr.end("failed", "\n".join(lines[-5:]))
    tr.finish()


if __name__ == "__main__":
    main()

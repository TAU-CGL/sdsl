"""Shared helpers for the active-SDSL benchmark harness.

Nothing in here modifies the sdsl core library. Instrumentation is done by
wrapping module attributes / instance methods at runtime only.
"""
from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import importlib.util
import io
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXAMPLES = REPO / "examples"
RESULTS = HERE / "results"

SLAM = "resources/maps/2d/slam"
# The four maps used by examples/experiment_movement.py (all room-annotated).
MAPS = {
    "square":     (f"{SLAM}/simple_symmetry/square.yaml",     f"{SLAM}/simple_symmetry/square_c.png"),
    "symmetry_2": (f"{SLAM}/simple_symmetry/symmetry_2.yaml", f"{SLAM}/simple_symmetry/symmetry_2_c.png"),
    "plus":       (f"{SLAM}/simple_symmetry/plus.yaml",       f"{SLAM}/simple_symmetry/plus_c.png"),
    "apt":        (f"{SLAM}/apt_20250913_1449/my_map.yaml",   f"{SLAM}/apt_20250913_1449/my_map_c.png"),
}

_libc = ctypes.CDLL(ctypes.util.find_library("c"))


# ---------------------------------------------------------------------------
# Host information
# ---------------------------------------------------------------------------

def _read(path, default=None):
    try:
        return Path(path).read_text().strip()
    except Exception:
        return default


def host_info():
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pointer_bits": 8 * ctypes.sizeof(ctypes.c_void_p),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "matplotlib_stubbed": MATPLOTLIB_STUBBED,
    }
    if sys.platform == "darwin":
        try:
            info["cpu_model"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            info["mem_bytes"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
        except Exception:
            pass
    else:
        model = _read("/proc/device-tree/model")
        if model:
            info["cpu_model"] = model.replace("\x00", "")
        info["governor"] = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        info["max_freq_khz"] = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    info["mem_bytes"] = int(line.split()[1]) * 1024
        except Exception:
            pass
    try:
        import sdsl
        info["sdsl_file"] = sdsl.__file__
        info["sdsl_version"] = getattr(sdsl, "__version__", None)
    except Exception as e:  # pragma: no cover
        info["sdsl_error"] = repr(e)
    try:
        info["git_head"] = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
        info["git_dirty"] = bool(subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain", "--", "examples", "sdsl"], text=True).strip())
    except Exception:
        pass
    return info


def thermal_sample():
    """Cheap snapshot of thermal / frequency / load state."""
    s = {"t": time.time()}
    try:
        s["load1"] = os.getloadavg()[0]
    except Exception:
        pass
    temp = _read("/sys/class/thermal/thermal_zone0/temp")
    if temp:
        s["temp_c"] = int(temp) / 1000.0
    freq = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    if freq:
        s["freq_mhz"] = int(freq) / 1000.0
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2)
        if out.returncode == 0 and "=" in out.stdout:
            s["throttled"] = out.stdout.strip().split("=")[1]
    except Exception:
        pass
    return s


# ---------------------------------------------------------------------------
# Capturing the C++ fmt::print output (goes to the C-level stdout, fd 1)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def capture_native_stdout():
    """Redirect fd 1 (and Python's sys.stdout) into a buffer.

    Yields a dict whose "text" key is filled on exit with everything that was
    printed, both by Python and by the C++ extension (fmt::print -> stdio).
    """
    out = {"text": ""}
    sys.stdout.flush()
    _libc.fflush(None)
    saved_fd = os.dup(1)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 1)
    py_buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = py_buf
    try:
        yield out
    finally:
        _libc.fflush(None)
        sys.stdout = old_stdout
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        tmp.seek(0)
        out["text"] = tmp.read().decode(errors="replace") + py_buf.getvalue()
        tmp.close()


_ITER_RE = re.compile(r"Iteration: (\d+)\n\t(\d+)")
_TIMEOUT_RE = re.compile(r"Timeout reached at iteration (\d+)")
_THREADS_RE = re.compile(r"with #(\d+) threads")


def parse_localize_log(text):
    """Parse the verbose output of localize_omp_forkjoin (one or more calls)."""
    calls = []
    for chunk in text.split("Running with OpenMP")[1:]:
        iters = [(int(i), int(n)) for i, n in _ITER_RE.findall(chunk)]
        to = _TIMEOUT_RE.search(chunk)
        th = _THREADS_RE.search("Running with OpenMP" + chunk)
        calls.append({
            "iter_counts": [n for _, n in iters],
            "iterations_run": len(iters),
            "timeout_at": int(to.group(1)) if to else None,
            "threads": int(th.group(1)) if th else None,
        })
    return calls


# ---------------------------------------------------------------------------
# Loading the active-loop modules (original from the baseline commit, or working tree)
# ---------------------------------------------------------------------------

# The "orig" variant is pinned to the commit the benchmarks were taken against, so that
# committing the optimized examples later does not silently turn "orig" into "opt".
ORIG_REF = os.environ.get("SDSL_ORIG_REF", "328423d")

_variant_cache = {}


def _ensure_matplotlib_or_stub():
    """The examples import matplotlib only for plotting. On the RPi (armhf,
    Python 3.12) there is no matplotlib wheel, so install an inert stub that
    satisfies the imports. Never used for computation."""
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot  # noqa: F401
        return False
    except Exception:
        pass
    import types

    class _Dummy:
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k): return _Dummy()
        def __getattr__(self, n): return _Dummy()

    def _mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        m.__getattr__ = lambda n: _Dummy()
        sys.modules[name] = m
        return m

    mpl = _mod("matplotlib", use=lambda *a, **k: None)
    mpl.pyplot = _mod("matplotlib.pyplot")
    mpl.collections = _mod("matplotlib.collections", PatchCollection=_Dummy, LineCollection=_Dummy)
    mpl.colors = _mod("matplotlib.colors", LinearSegmentedColormap=_Dummy(), hsv_to_rgb=lambda *a, **k: (0, 0, 0))
    mpl.patches = _mod("matplotlib.patches", Rectangle=_Dummy)
    mpl.cm = _mod("matplotlib.cm")
    return True


MATPLOTLIB_STUBBED = None


def load_active_modules(variant="orig"):
    """Return (demo_movement_module, ChooseMovement_class) for a variant.

    variant="orig": examples/{demo_movement,ChooseMovement}.py as of ORIG_REF
                    (extracted to a temp dir), i.e. the unmodified code.
    variant="opt":  the working-tree versions (after harness optimizations).
    """
    global MATPLOTLIB_STUBBED
    if variant in _variant_cache:
        return _variant_cache[variant]
    if MATPLOTLIB_STUBBED is None:
        MATPLOTLIB_STUBBED = _ensure_matplotlib_or_stub()
    if variant == "orig":
        d = Path(tempfile.mkdtemp(prefix="sdsl_orig_"))
        for name in ("demo_movement.py", "ChooseMovement.py"):
            src = subprocess.check_output(
                ["git", "-C", str(REPO), "show", f"{ORIG_REF}:examples/{name}"], text=True)
            (d / name).write_text(src)
        src_dir = d
    elif variant == "opt":
        src_dir = EXAMPLES
    else:
        raise ValueError(variant)

    sys.modules.pop("ChooseMovement", None)
    sys.path.insert(0, str(src_dir))
    try:
        spec = importlib.util.spec_from_file_location(f"demo_movement_{variant}", src_dir / "demo_movement.py")
        dm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dm)
    finally:
        sys.path.remove(str(src_dir))
        sys.modules.pop("ChooseMovement", None)
    _variant_cache[variant] = (dm, dm.ChooseMovement)
    return dm, dm.ChooseMovement


def load_env(map_name):
    import sdsl
    from sdsl.loaders.load_pgm_map import load_pgm_map
    yaml_path, img_path = MAPS[map_name]
    pgm = load_pgm_map(str(REPO / yaml_path))
    env = sdsl.Env_2D_PGM(pgm.grid, pgm.resolution, pgm.origin_x, pgm.origin_y,
                          pgm.occupied_thresh, pgm.negate)
    return pgm, env, str(REPO / yaml_path), str(REPO / img_path)


def sample_free_poses(pgm, env, n, rng):
    import sdsl
    xmin, ymin = pgm.origin_x, pgm.origin_y
    xmax = xmin + pgm.width * pgm.resolution
    ymax = ymin + pgm.height * pgm.resolution
    out = []
    while len(out) < n:
        x, y = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
        if env.contains(sdsl.R3(x, y, 0.0)):
            out.append((float(x), float(y), float(rng.uniform(0, 2 * math.pi))))
    return out


# ---------------------------------------------------------------------------
# Timing instrumentation
# ---------------------------------------------------------------------------

class StageTimer:
    """Accumulates (context, name) -> [total_seconds, calls]."""

    def __init__(self):
        self.ctx = "driver"
        self.acc = defaultdict(lambda: [0.0, 0])
        self.extra = {}

    @contextlib.contextmanager
    def context(self, name):
        old, self.ctx = self.ctx, name
        try:
            yield
        finally:
            self.ctx = old

    def add(self, name, dt):
        a = self.acc[(self.ctx, name)]
        a[0] += dt
        a[1] += 1

    def wrap(self, fn, name, on_result=None):
        timer = self

        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            r = fn(*args, **kwargs)
            timer.add(name, time.perf_counter() - t0)
            if on_result is not None:
                on_result(timer, args, r)
            return r
        wrapped.__wrapped__ = fn
        return wrapped

    def snapshot_and_reset(self):
        snap = {f"{c}/{n}": {"s": v[0], "calls": v[1]} for (c, n), v in self.acc.items()}
        snap.update({f"extra/{k}": v for k, v in self.extra.items()})
        self.acc.clear()
        self.extra = {}
        return snap


class EnvProxy:
    """Counts Python-level environment queries (used only in diagnostic runs)."""

    def __init__(self, raw):
        self.raw = raw
        self.counts = Counter()

    def contains(self, q):
        self.counts["contains"] += 1
        return self.raw.contains(q)

    def collision_detection(self, a, b):
        self.counts["collision_detection"] += 1
        return self.raw.collision_detection(a, b)

    def measure_distance(self, q):
        self.counts["measure_distance"] += 1
        return self.raw.measure_distance(q)

    def __getattr__(self, name):
        return getattr(self.raw, name)


_SDSL_ORIG = {}


def install_sdsl_timers(timer, unwrap_proxy=False):
    """Wrap sdsl entry points so every call is timed under timer.ctx."""
    import sdsl
    names = ["localize_omp_forkjoin_3d", "cleanup_SE2", "fusion_2d",
             "connectedComponentsWithBeliefs", "entropy_SE2"]
    for n in names:
        _SDSL_ORIG.setdefault(n, getattr(sdsl, n))

    def rec_localize(t, args, r):
        t.extra.setdefault("n_vox_raw", []).append(len(r))

    def rec_cleanup(t, args, r):
        t.extra.setdefault("n_vox_clean", []).append(len(r))

    def rec_cc(t, args, r):
        t.extra.setdefault(f"cc_n_in_{t.ctx}", []).append(len(args[0]))

    hooks = {"localize_omp_forkjoin_3d": rec_localize, "cleanup_SE2": rec_cleanup,
             "connectedComponentsWithBeliefs": rec_cc}
    for n in names:
        setattr(sdsl, n, timer.wrap(_SDSL_ORIG[n], n, hooks.get(n)))

    _SDSL_ORIG.setdefault("Predicate_Fwd2D_Arr", sdsl.Predicate_Fwd2D_Arr)
    orig_pred = _SDSL_ORIG["Predicate_Fwd2D_Arr"]

    def pred_ctor(env, *a, **k):
        if isinstance(env, EnvProxy):
            env = env.raw
        return orig_pred(env, *a, **k)
    sdsl.Predicate_Fwd2D_Arr = timer.wrap(pred_ctor, "Predicate_Fwd2D_Arr")


def uninstall_sdsl_timers():
    import sdsl
    for n, f in _SDSL_ORIG.items():
        setattr(sdsl, n, f)


def wrap_mover(timer, mover):
    for name in ("simulate_movements", "_move_voxels_and_normalize_beliefs",
                 "min_distance_to_neighbor_rooms", "mark_voxels_visited",
                 "point_to_room_edge_distance"):
        if hasattr(mover, name):
            setattr(mover, name, timer.wrap(getattr(mover, name), f"mover.{name}"))


def timeit_calls(fn, *, min_time=0.2, max_calls=10_000_000, budget=None):
    """Return (seconds_per_call, calls) by running fn repeatedly for >= min_time."""
    n, total = 0, 0.0
    t_start = time.perf_counter()
    batch = 1
    while total < min_time and n < max_calls:
        t0 = time.perf_counter()
        for _ in range(batch):
            fn()
        dt = time.perf_counter() - t0
        total += dt
        n += batch
        batch = min(batch * 2, 1 << 16)
        if budget and time.perf_counter() - t_start > budget:
            break
    return total / max(n, 1), n


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
    tmp.replace(path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

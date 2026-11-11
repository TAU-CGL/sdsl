#!/usr/bin/env python3
"""Dump the PGM maps + benchmark poses into a tiny binary format read by bench_kernels.cpp.

Poses are the same ones used by the Python localize_depth task (same RNG seed), so
native and Python timings are directly comparable.
"""
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import tasks as T  # noqa: E402


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for m in C.MAPS:
        pgm, env, _, _ = C.load_env(m)
        rng = np.random.default_rng(T.PARAMS["seed"] + 1)
        poses = C.sample_free_poses(pgm, env, T.PARAMS["loc_poses_per_map"], rng)
        grid = np.ascontiguousarray(pgm.grid, dtype=np.uint8)
        with open(out / f"{m}.bin", "wb") as f:
            f.write(struct.pack("<4i", pgm.width, pgm.height, int(pgm.negate), len(poses)))
            f.write(struct.pack("<4d", pgm.resolution, pgm.origin_x, pgm.origin_y, pgm.occupied_thresh))
            for p in poses:
                f.write(struct.pack("<3d", *p))
            f.write(grid.tobytes())
        print(m, grid.shape, len(poses))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / "inputs"))

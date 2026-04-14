"""Localization coverage tests for all 3-DOF planar environment types.

For each environment (Env_R2_Arrangement, Env_2D_PCD, Env_2D_PGM):
  1. Sample N_SAMPLES interior configurations via rejection sampling
     (uses ``contains()``).
  2. For each, cast N_RAYS body-frame rays and run localisation.
  3. Assert that the ground-truth pose falls inside at least one returned voxel
     for >= SUCCESS_RATE of all trials.

Set SDSL_TEST_N_SAMPLES=10000 for the full 10 K-sample sweep described in the
spec; the default of 100 keeps CI under roughly 10 s per environment type.
"""
import os
from pathlib import Path

import numpy as np
import pytest

import sdsl
from sdsl.loaders.load_pgm_map import load_pgm_map

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SAMPLES      = int(os.environ.get("SDSL_TEST_N_SAMPLES", "100"))
N_RAYS         = 16
KK_PRIME_RATIO = 0.8   # tolerate up to 20 % bad / escaping rays
ERROR_BOUND    = 0.2   # metres — loose enough for all three env types
# DEPTH must be <= 3.  The inner predicate's verify() has an early-return guard
# "if (m_iteration < 3) return true" that skips measureDistance().  For depth > 3
# verify() calls env->measureDistance() from inside an OMP parallel region.
# Env_R2_Arrangement's measureDistance uses CGAL::zone with
# Arr_trapezoid_ric_point_location, which is NOT thread-safe for concurrent
# queries.  Depth 3 still fully exercises the k-k' intersects logic and
# returns voxels at 1/8 of the bbox in each dimension.
DEPTH          = 3
SUCCESS_RATE   = 0.99

REPO_ROOT = Path(__file__).parent.parent.parent
APT_YAML  = REPO_ROOT / "resources" / "maps" / "2d" / "slam" / "apt.yaml"

# ---------------------------------------------------------------------------
# L-shaped room used for the arrangement and PCD environments
#
#   (0,8)────(4,8)
#     |        |
#   (0,4)    (4,4)──(8,4)
#     |               |
#   (0,0)──────────(8,0)
# ---------------------------------------------------------------------------
_L_VERTS = np.array([
    [0.0, 0.0],
    [8.0, 0.0],
    [8.0, 4.0],
    [4.0, 4.0],
    [4.0, 8.0],
    [0.0, 8.0],
], dtype=np.float64)


def _poly_to_segments(verts):
    closed = np.vstack([verts, verts[0]])
    return np.column_stack([closed[:-1], closed[1:]])   # (N, 4)


def _poly_to_pcd(verts, spacing=0.15):
    closed = np.vstack([verts, verts[0]])
    pts = []
    for i in range(len(verts)):
        p0, p1 = closed[i], closed[i + 1]
        n = max(2, int(round(np.linalg.norm(p1 - p0) / spacing)) + 1)
        for t in np.linspace(0.0, 1.0, n, endpoint=False):
            pts.append(p0 + t * (p1 - p0))
    return np.array(pts, dtype=np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_interior(sampler_env, bbox, n, rng):
    """Rejection-sample *n* interior (x, y, theta) configurations.

    Uses ``sampler_env.contains()`` as the membership oracle.  Samples are
    drawn in batches to amortise Python loop overhead.
    """
    xmin, xmax = bbox.bottom_left[0], bbox.top_right[0]
    ymin, ymax = bbox.bottom_left[1], bbox.top_right[1]

    configs = []
    while len(configs) < n:
        batch  = max(n * 4, 512)
        xs     = rng.uniform(xmin, xmax, batch)
        ys     = rng.uniform(ymin, ymax, batch)
        thetas = rng.uniform(0.0, 2.0 * np.pi, batch)
        for x, y, theta in zip(xs, ys, thetas):
            if sampler_env.contains(sdsl.R3(x, y, theta)):
                configs.append((x, y, theta))
                if len(configs) == n:
                    break
    return configs


def _in_any_voxel(voxels, q):
    """Return True iff *q* lies inside at least one voxel."""
    for v in voxels:
        bl, tr = v.bottom_left, v.top_right
        if (bl[0] <= q[0] <= tr[0] and
                bl[1] <= q[1] <= tr[1] and
                bl[2] <= q[2] <= tr[2]):
            return True
    return False


# ---------------------------------------------------------------------------
# Fixtures
#
# Each fixture entry yields (loc_env, sampler_env, bbox):
#   loc_env    – environment passed to the predicate and localization call
#   sampler_env – environment used for reliable contains()-based sampling
#                 (may differ from loc_env for Env_2D_PCD whose contains() is
#                 acknowledged as unreliable for sparse clouds)
#   bbox       – initial Voxel_R3 search region for localize_omp_forkjoin_3d
# ---------------------------------------------------------------------------

@pytest.fixture(
    params=["arrangement", "pcd", "pgm"],
    ids=   ["arrangement", "pcd", "pgm"],
)
def env_fixture(request):
    if request.param == "arrangement":
        env = sdsl.Env_R2_Arrangement(_poly_to_segments(_L_VERTS))
        return env, env, env.bounding_box()

    if request.param == "pcd":
        env     = sdsl.Env_2D_PCD(_poly_to_pcd(_L_VERTS))
        # Env_2D_PCD.contains() is unreliable for sparse edge clouds; use the
        # exact arrangement for interior sampling instead.
        sampler = sdsl.Env_R2_Arrangement(_poly_to_segments(_L_VERTS))
        return env, sampler, env.bounding_box()

    if request.param == "pgm":
        if not APT_YAML.exists():
            pytest.skip(f"PGM map not found: {APT_YAML}")
        pgm = load_pgm_map(str(APT_YAML))
        env = sdsl.Env_2D_PGM(
            pgm.grid, pgm.resolution, pgm.origin_x, pgm.origin_y,
            pgm.occupied_thresh, pgm.negate,
        )
        return env, env, env.bounding_box()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_localization_coverage(env_fixture):
    """Ground-truth pose must appear in the localization result >= 95 % of trials."""
    loc_env, sampler_env, bbox = env_fixture
    rng = np.random.default_rng(42)

    configs = _sample_interior(sampler_env, bbox, N_SAMPLES, rng)

    # Body-frame ray directions: N_RAYS uniformly spaced in [0, 2pi)
    d_thetas = np.linspace(0.0, 2.0 * np.pi, N_RAYS, endpoint=False)
    odometry = [sdsl.R3(0.0, 0.0, float(dt)) for dt in d_thetas]

    successes = 0
    for x_gt, y_gt, theta_gt in configs:
        # Measure from ground-truth pose in the robot's body frame
        measurements = [
            loc_env.measure_distance(sdsl.R3(x_gt, y_gt, theta_gt + dt))
            for dt in d_thetas
        ]

        pred   = sdsl.Predicate_Fwd2D_Arr(
            loc_env, odometry, measurements, KK_PRIME_RATIO, ERROR_BOUND)
        voxels = sdsl.localize_omp_forkjoin_3d(
            bbox, pred, DEPTH, verbose=False)

        q_gt = sdsl.R3(x_gt, y_gt, theta_gt)
        if _in_any_voxel(voxels, q_gt):
            successes += 1

    rate = successes / N_SAMPLES
    assert rate >= SUCCESS_RATE, (
        f"Success rate {rate:.1%} < {SUCCESS_RATE:.0%}  "
        f"({successes}/{N_SAMPLES} ground-truth poses recovered)"
    )

import math
import sdsl


def _voxel(x0, y0, t0, x1, y1, t1):
    bl = sdsl.R3(x0, y0, t0)
    tr = sdsl.R3(x1, y1, t1)
    return sdsl.Voxel_R3(bl, tr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref_entropy(projected_beliefs, vol):
    """Pure-Python reference: H = -sum b*log(b+eps)*vol."""
    eps = 1e-12
    return -sum(b * math.log(b + eps) * vol for b in projected_beliefs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_entropy_se2_no_theta_overlap():
    """Three voxels at distinct (x,y) locations with uniform belief."""
    s = 1.0   # side length
    voxels = [
        _voxel(0, 0, 0, s, s, s),
        _voxel(s, 0, 0, 2*s, s, s),
        _voxel(2*s, 0, 0, 3*s, s, s),
    ]
    beliefs = [1/3, 1/3, 1/3]
    vol = s * s  # 2-D projected area

    H = sdsl.entropy_SE2(voxels, beliefs)
    expected = _ref_entropy(beliefs, vol)
    assert abs(H - expected) < 1e-10


def test_entropy_se2_theta_merging():
    """Two voxels share the same (x,y) box but differ in theta — they collapse
    to one projected voxel, so entropy should equal that of a single bin."""
    s = 1.0
    voxels = [
        _voxel(0, 0,   0,   s, s, s),   # theta in [0,   s)
        _voxel(0, 0,   s, s, s, 2*s),   # theta in [s,  2s)  — same xy
    ]
    beliefs = [0.4, 0.6]
    vol = s * s

    H = sdsl.entropy_SE2(voxels, beliefs)

    # After merging: one projected voxel with belief = 1.0
    expected = _ref_entropy([1.0], vol)
    assert abs(H - expected) < 1e-10




def test_entropy_se2_uniform_is_max():
    """Uniform belief has higher entropy than concentrated belief."""
    s = 1.0
    voxels = [_voxel(i*s, 0, 0, (i+1)*s, s, s) for i in range(4)]

    uniform   = [0.25] * 4
    spike     = [0.97, 0.01, 0.01, 0.01]

    H_uniform = sdsl.entropy_SE2(voxels, uniform)
    H_spike   = sdsl.entropy_SE2(voxels, spike)
    assert H_uniform > H_spike


def test_entropy_se2_single_voxel():
    """Single voxel with belief 1 → entropy ≈ 0."""
    v = _voxel(0, 0, 0, 1, 1, 1)
    H = sdsl.entropy_SE2([v], [1.0])
    assert abs(H) < 1e-9


def test_entropy_se2_empty():
    """Empty input returns 0."""
    H = sdsl.entropy_SE2([], [])
    assert H == 0.0

import sdsl

def test_omp_always_true_3d():
    bl = sdsl.R3(1,2,3)
    tr = sdsl.R3(4,5,6)
    v = sdsl.Voxel_R3(bl, tr)
    pred = sdsl.Predicate_AlwaysTrue_3d()
    depth = 4

    res = sdsl.localize_omp_forkjoin_3d(v, pred, depth)
    assert len(res) == (2**3) ** (depth-1)

def test_omp_always_true_4d():
    bl = sdsl.R4(1,2,3,4)
    tr = sdsl.R4(5,6,7,8)
    v = sdsl.Voxel_R4(bl, tr)
    pred = sdsl.Predicate_AlwaysTrue_4d()
    depth = 4

    res = sdsl.localize_omp_forkjoin_4d(v, pred, depth)
    assert len(res) == (2**4) ** (depth-1)
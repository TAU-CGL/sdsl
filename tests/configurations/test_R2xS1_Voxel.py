import sdsl

def test_R2xS1_Voxel_init():
    v = sdsl.R2xS1_Voxel(
        sdsl.R2xS1(-1, -2, 0),
        sdsl.R2xS1(1, 2, 1.57)
    )

    assert v.bottom_left() == sdsl.R2xS1(-1, -2, 0)
    assert v.top_right() == sdsl.R2xS1(1, 2, 1.57)

    q1 = sdsl.R2xS1(0, 0, 0)
    q2 = sdsl.R2xS1(0, 0, 2)
    q3 = sdsl.R2xS1(1, 3, 0.5)
    q4 = sdsl.R2xS1(2, 4, 2)

    assert v.contains(q1)
    assert not v.contains(q2)
    assert not v.contains(q3)
    assert not v.contains(q4)

def test_R2xS1_Voxel_split():
    v = sdsl.R2xS1_Voxel(
        sdsl.R2xS1(-1, -2, 0),
        sdsl.R2xS1(1, 2, 1.57)
    )
    split_v = v.split()
    assert len(split_v) == 8
    assert sdsl.R2xS1_Voxel.voxels_bounding_box(split_v) == v
    assert not sdsl.R2xS1_Voxel.voxels_bounding_box(split_v[:3]) == v
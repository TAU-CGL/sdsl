import sdsl

def test_R3xS1_Voxel_init():
    v = sdsl.R3xS1_Voxel(
        sdsl.R3xS1(-1, -2, -3, 0),
        sdsl.R3xS1(1, 2, 3, 1.57)
    )

    assert v.bottom_left() == sdsl.R3xS1(-1, -2, -3, 0)
    assert v.top_right() == sdsl.R3xS1(1, 2, 3, 1.57)

    q1 = sdsl.R3xS1(0, 0, 0, 0)
    q2 = sdsl.R3xS1(0, 0, 0, 2)
    q3 = sdsl.R3xS1(1, 3, 0.5, 0.5)
    q4 = sdsl.R3xS1(2, 4, 2, 2)

    assert v.contains(q1)
    assert not v.contains(q2)
    assert not v.contains(q3)
    assert not v.contains(q4)

def test_R3xS1_Voxel_split():
    v = sdsl.R3xS1_Voxel(
        sdsl.R3xS1(-1, -2, -3, 0),
        sdsl.R3xS1(1, 2, 3, 1.57)
    )
    split_v = v.split()
    assert len(split_v) == 16
    assert sdsl.R3xS1_Voxel.voxels_bounding_box(split_v) == v
    assert not sdsl.R3xS1_Voxel.voxels_bounding_box(split_v[:8]) == v
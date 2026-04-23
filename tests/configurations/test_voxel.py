import sdsl

def test_voxel_1d_init():
    v = sdsl.Voxel_R1()
    assert v.bottom_left[0] == 0.0
    assert v.top_right[0] == 0.0


def test_voxel_2d_init():
    bl = sdsl.R2(-1,-2)
    tr = sdsl.R2(1, 2)

    v = sdsl.Voxel_R2(bl, tr)
    assert v.bottom_left[0] == -1.0
    assert v.bottom_left[1] == -2.0
    assert v.top_right[0] == 1.0
    assert v.top_right[1] == 2.0


def test_voxel_3d_init():
    bl = sdsl.R3()
    bl[0] = 0.0
    bl[1] = 0.0
    bl[2] = 0.0
    
    tr = sdsl.R3()
    tr[0] = 10.0
    tr[1] = 10.0
    tr[2] = 10.0
    
    v = sdsl.Voxel_R3(bl, tr)
    assert v.bottom_left[0] == 0.0
    assert v.bottom_left[2] == 0.0
    assert v.top_right[0] == 10.0
    assert v.top_right[2] == 10.0


def test_voxel_2d_midpoint():
    bl = sdsl.R2()
    bl[0] = 0.0
    bl[1] = 0.0
    
    tr = sdsl.R2()
    tr[0] = 4.0
    tr[1] = 6.0
    
    v = sdsl.Voxel_R2(bl, tr)
    mid = v.midpoint()
    
    assert mid[0] == 2.0
    assert mid[1] == 3.0


def test_voxel_3d_midpoint():
    bl = sdsl.R3()
    bl[0] = -5.0
    bl[1] = -10.0
    bl[2] = 0.0
    
    tr = sdsl.R3()
    tr[0] = 5.0
    tr[1] = 10.0
    tr[2] = 20.0
    
    v = sdsl.Voxel_R3(bl, tr)
    mid = v.midpoint()
    
    assert mid[0] == 0.0
    assert mid[1] == 0.0
    assert mid[2] == 10.0


def test_voxel_1d_split_natural():
    bl = sdsl.R1()
    bl[0] = 0.0
    
    tr = sdsl.R1()
    tr[0] = 10.0
    
    v = sdsl.Voxel_R1(bl, tr)
    subvoxels = v.split()
    
    assert len(subvoxels) == 2  # 2^1 = 2
    
    # Check first subvoxel: [0, 5]
    assert subvoxels[0].bottom_left[0] == 0.0
    assert subvoxels[0].top_right[0] == 5.0
    
    # Check second subvoxel: [5, 10]
    assert subvoxels[1].bottom_left[0] == 5.0
    assert subvoxels[1].top_right[0] == 10.0


def test_voxel_2d_split_natural():
    bl = sdsl.R2()
    bl[0] = 0.0
    bl[1] = 0.0
    
    tr = sdsl.R2()
    tr[0] = 4.0
    tr[1] = 6.0
    
    v = sdsl.Voxel_R2(bl, tr)
    subvoxels = v.split()
    
    assert len(subvoxels) == 4  # 2^2 = 4
    
    # Verify all subvoxels are within the original voxel
    for sv in subvoxels:
        assert sv.bottom_left[0] >= bl[0]
        assert sv.bottom_left[1] >= bl[1]
        assert sv.top_right[0] <= tr[0]
        assert sv.top_right[1] <= tr[1]
    
    # Check that midpoint (2, 3) is used for splitting
    # Subvoxel 0: bottom_left=(0,0), top_right=(2,3)
    assert subvoxels[0].bottom_left[0] == 0.0
    assert subvoxels[0].bottom_left[1] == 0.0
    assert subvoxels[0].top_right[0] == 2.0
    assert subvoxels[0].top_right[1] == 3.0


def test_voxel_3d_split_natural():
    bl = sdsl.R3()
    bl[0] = -1.0
    bl[1] = -1.0
    bl[2] = -1.0
    
    tr = sdsl.R3()
    tr[0] = 1.0
    tr[1] = 1.0
    tr[2] = 1.0
    
    v = sdsl.Voxel_R3(bl, tr)
    subvoxels = v.split()
    
    assert len(subvoxels) == 8  # 2^3 = 8
    
    # Verify all subvoxels are within the original voxel
    for sv in subvoxels:
        for i in range(3):
            assert sv.bottom_left[i] >= bl[i]
            assert sv.top_right[i] <= tr[i]


def test_voxel_2d_split_custom_midpoint():
    bl = sdsl.R2()
    bl[0] = 0.0
    bl[1] = 0.0
    
    tr = sdsl.R2()
    tr[0] = 10.0
    tr[1] = 10.0
    
    # Use custom midpoint (not the natural center)
    custom_mid = sdsl.R2()
    custom_mid[0] = 3.0
    custom_mid[1] = 7.0
    
    v = sdsl.Voxel_R2(bl, tr)
    subvoxels = v.split(custom_mid)
    
    assert len(subvoxels) == 4  # 2^2 = 4
    
    # Verify the split uses the custom midpoint
    # Subvoxel 0 should have top_right at custom midpoint
    assert subvoxels[0].top_right[0] == 3.0
    assert subvoxels[0].top_right[1] == 7.0


def test_voxel_3d_split_custom_midpoint():
    bl = sdsl.R3()
    bl[0] = 0.0
    bl[1] = 0.0
    bl[2] = 0.0
    
    tr = sdsl.R3()
    tr[0] = 100.0
    tr[1] = 100.0
    tr[2] = 100.0
    
    # Use custom midpoint
    custom_mid = sdsl.R3()
    custom_mid[0] = 25.0
    custom_mid[1] = 75.0
    custom_mid[2] = 50.0
    
    v = sdsl.Voxel_R3(bl, tr)
    subvoxels = v.split(custom_mid)
    
    assert len(subvoxels) == 8  # 2^3 = 8
    
    # Verify the split uses the custom midpoint
    # Subvoxel 0 should have top_right at custom midpoint
    assert subvoxels[0].top_right[0] == 25.0
    assert subvoxels[0].top_right[1] == 75.0
    assert subvoxels[0].top_right[2] == 50.0


def test_voxel_4d_split_natural():
    bl = sdsl.R4()
    tr = sdsl.R4()
    
    for i in range(4):
        bl[i] = 0.0
        tr[i] = 8.0
    
    v = sdsl.Voxel_R4(bl, tr)
    subvoxels = v.split()
    
    assert len(subvoxels) == 16  # 2^4 = 16
    
    # Verify all subvoxels are within the original voxel
    for sv in subvoxels:
        for i in range(4):
            assert sv.bottom_left[i] >= bl[i]
            assert sv.top_right[i] <= tr[i]

def test_voxel_3d_diameter():
    bl = sdsl.R3()
    bl[0] = 0.0
    bl[1] = 0.0
    bl[2] = 0.0
    
    tr = sdsl.R3()
    tr[0] = 1.0
    tr[1] = 2.0
    tr[2] = 2.0
    
    v = sdsl.Voxel_R3(bl, tr)
    diameter = v.diameter()
    
    expected_diameter = ((1.0 - 0.0)**2 + (2.0 - 0.0)**2 + (2.0 - 0.0)**2) ** 0.5
    assert diameter == expected_diameter

def test_voxel_3d_volume():
    bl = sdsl.R3()
    bl[0] = 0.0
    bl[1] = 0.0
    bl[2] = 0.0
    
    tr = sdsl.R3()
    tr[0] = 2.0
    tr[1] = 3.0
    tr[2] = 4.0
    
    v = sdsl.Voxel_R3(bl, tr)
    volume = v.volume()
    
    expected_volume = (2.0 - 0.0) * (3.0 - 0.0) * (4.0 - 0.0)
    assert volume == expected_volume
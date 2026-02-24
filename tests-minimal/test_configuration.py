import pytest
import sys
sys.path.insert(0, '../')
import sdsl


def test_config_1d_init():
    """Test 1D Configuration initialization and access"""
    c = sdsl.Config_1d()
    assert c[0] == 0.0
    
    c[0] = 5.0
    assert c[0] == 5.0


def test_config_2d_init():
    """Test 2D Configuration initialization and access"""
    c = sdsl.Config_2d()
    assert c[0] == 0.0
    assert c[1] == 0.0
    
    c[0] = 1.5
    c[1] = 2.5
    assert c[0] == 1.5
    assert c[1] == 2.5


def test_config_3d_init():
    """Test 3D Configuration initialization and access"""
    c = sdsl.Config_3d()
    assert c[0] == 0.0
    assert c[1] == 0.0
    assert c[2] == 0.0
    
    c[0] = 1.0
    c[1] = 2.0
    c[2] = 3.0
    assert c[0] == 1.0
    assert c[1] == 2.0
    assert c[2] == 3.0


def test_config_4d_init():
    """Test 4D Configuration initialization and access"""
    c = sdsl.Config_4d()
    for i in range(4):
        assert c[i] == 0.0
    
    for i in range(4):
        c[i] = float(i + 1)
    
    for i in range(4):
        assert c[i] == float(i + 1)


def test_config_2d_indexing():
    """Test indexing operations for 2D Configuration"""
    c = sdsl.Config_2d()
    c[0] = 10.5
    c[1] = 20.5
    
    assert c[0] == 10.5
    assert c[1] == 20.5


def test_config_3d_indexing():
    """Test indexing operations for 3D Configuration"""
    c = sdsl.Config_3d()
    values = [1.1, 2.2, 3.3]
    
    for i, val in enumerate(values):
        c[i] = val
    
    for i, val in enumerate(values):
        assert c[i] == val


def test_config_repr():
    """Test string representation of Configuration"""
    c = sdsl.Config_2d()
    c[0] = 1.0
    c[1] = 2.0
    repr_str = repr(c)
    
    assert "Configuration2" in repr_str
    assert "1.0" in repr_str or "1." in repr_str
    assert "2.0" in repr_str or "2." in repr_str

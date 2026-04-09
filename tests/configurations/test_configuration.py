import math
import sdsl

def test_R3_init():
    q = sdsl.R3(1.0, 2.0, 3.0)
    assert q[0] == 1.0
    assert q[1] == 2.0
    assert q[2] == 3.0

def test_R3_conditionals():
    q1 = sdsl.R3(1,2,3)
    q2 = sdsl.R3(4,5,6)
    q3 = sdsl.R3(1,5,6)

    assert q1 == sdsl.R3(1.0, 2.0, 3.0)
    assert q1 != q2
    assert q1 < q2
    assert q1 <= q2
    assert q2 > q1
    assert q2 >= q1
    assert q1 <= q3
    assert not q1 < q3 # Test also the edge-case where one coord is equal
    assert q2 >= q3
    assert not q2 > q3
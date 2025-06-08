import math

import sdsl

def test_R3xS1_init():
    q = sdsl.R3xS1(1.0, 2.0, 3.0, 4.0)
    assert q.x() == 1.0
    assert q.y() == 2.0
    assert q.z() == 3.0
    assert q.r() == 4.0

def test_R3xS1_operators():
    q1 = sdsl.R3xS1(1.0, 2.0, 3.0, 4.0)
    q2 = sdsl.R3xS1(5.0, 6.0, 7.0, 8.0)
    q3 = sdsl.R3xS1(1.0, 6.0, 7.0, 8.0)

    assert q1 == sdsl.R3xS1(1.0, 2.0, 3.0, 4.0)
    assert q1 != q2
    assert q1 < q2
    assert q1 <= q2
    assert q2 > q1
    assert q2 >= q1
    assert q1 <= q3
    assert not q1 < q3 # Test also the edge case where one coord is equal
    assert q2 >= q3
    assert not q2 > q3

def test_R3xS1_arithmetic_inv():
    q = sdsl.R3xS1(1.0, 2.0, 3.0, 4.0)
    assert q * q.inv() == sdsl.R3xS1(0.0, 0.0, 0.0, 0.0)
    assert q.inv() * q == sdsl.R3xS1(0.0, 0.0, 0.0, 0.0)

def test_R3xS1_arithmetic_power():
    e = sdsl.R3xS1(0, 0, 0, 0)
    q = sdsl.R3xS1(1, 0, 0, math.pi * 0.5)
    assert e * q == q * e == q

    k = 8
    dt = 2.0 * math.pi / k
    g = sdsl.R3xS1(math.cos(dt), math.sin(dt), 0, dt)

    q_ = q
    for _ in range(k):
        q_ = q_ * g
    
    # We should have gone back to initial position, but offset 2pi
    q_ = sdsl.R3xS1(0, 0, 0, -2.0 * math.pi) * q_
    assert q_ == q
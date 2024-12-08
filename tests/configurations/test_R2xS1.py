import math

import sdsl


def test_R2xS1_init():
    q = sdsl.R2xS1(1.0, 2.0, 3.0)
    assert q.x() == 1.0
    assert q.y() == 2.0
    assert q.r() == 3.0

def test_R2xS1_operators():
    q1 = sdsl.R2xS1(1.0, 2.0, 3.0)
    q2 = sdsl.R2xS1(4.0, 5.0, 6.0)
    q3 = sdsl.R2xS1(1.0, 5.0, 6.0)

    assert q1 == sdsl.R2xS1(1.0, 2.0, 3.0)
    assert q1 != q2
    assert q1 < q2
    assert q1 <= q2
    assert q2 > q1
    assert q2 >= q1
    assert q1 <= q3
    assert not q1 < q3 # Test also the edge case where one coord is equal
    assert q2 >= q3
    assert not q2 > q3

def test_R2xS1_arithmetic_inv():
    q = sdsl.R2xS1(1.0, 2.0, 3.0)
    assert q * q.inv() == sdsl.R2xS1(0.0, 0.0, 0.0)
    assert q.inv() * q == sdsl.R2xS1(0.0, 0.0, 0.0)

def test_R2xS1_arithmetic_power():
    e = sdsl.R2xS1(0, 0, 0)
    q = sdsl.R2xS1(1, 0, math.pi * 0.5)
    assert e * q == q * e == q

    k = 8
    dt = 2.0 * math.pi / k
    g = sdsl.R2xS1(math.cos(dt), math.sin(dt), dt)

    q_ = q
    for _ in range(k):
        q_ = q_ * g
    
    # We should have gone back to initial position, but offset 2pi
    q_ = sdsl.R2xS1(0,0, -2.0 * math.pi) * q_
    assert q_ == q
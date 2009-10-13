import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from poly import equal_by_values
from power import PowerBasis
from lagrange import LagrangeBasis, bit_reverse, chebyshev_points_sequence, lagrange_from_roots

from test_poly import check_example, check_standard

def test_bit_reverse():
    n = 64
    rs = set()
    for i in range(n):
        b = bit_reverse(i,n)
        rs.add(b)

    for i in range(n):
        assert i in rs

def test_points():
    pts = [-1,0,1]
    b = LagrangeBasis(pts)
    b.extend_points(4)
    assert abs(b.points[-1])>1e-2

def test_examples():
    for (l, x, y) in [([3],0,3),
                      ([], 1, 0),
                      ([1,0,-1], 0.5, -0.5),
                      ([1,0,1], 3, 9)]:
        b = LagrangeBasis([-1,0,1])
        p = b.polynomial(l)
        yield check_example, p, x, y

def test_interpolating():
    xs = np.linspace(-1,1,5)
    p1 = PowerBasis().polynomial([1,2,3,1])
    p2 = LagrangeBasis(xs).polynomial(p1(xs))
    assert equal_by_values(p1,p2)

def test_standard():
    for t in check_standard(LagrangeBasis(),coefficient_addition=False,division=False):
        yield t
    for t in check_standard(LagrangeBasis([1,3,-2]),coefficient_addition=False,division=False):
        yield t

def test_power():
    b = LagrangeBasis()
    p = b.polynomial([0,1])
    for i in range(16):
        assert equal_by_values(p**i, reduce(lambda x, y: x*y, [p]*i, b.one()))


def test_from_roots():
    r = [1,3,8,12,-7]

    p = lagrange_from_roots(r)
    assert_array_almost_equal(p(r),0)

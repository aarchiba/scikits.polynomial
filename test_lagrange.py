import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from poly import equal_by_values
from power import PowerBasis
from lagrange import LagrangeBasis, bit_reverse, chebyshev_points_sequence, lagrange_from_roots

from test_poly import check_example, \
        check_operation, \
        check_operation, \
        check_coefficient_addition, \
        check_scalar_operation, \
        check_divmod, \
        check_product_rule, \
        check_derivative_linearity

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

def test_operations():
    for (l1, l2) in [([], [1,2,3]),
                     ([1,2], [3,4]),
                     ([1], [1,2,3,4]),
                     ([0,0,0,0,1], [1,2,3,4]),
                     ([], [])]:
        b = LagrangeBasis()
        p1 = b.polynomial(l1)
        p2 = b.polynomial(l2)
        yield check_operation, (lambda x, y:x+y), p1, p2
        yield check_operation, (lambda x, y:x-y), p1, p2
        yield check_operation, (lambda x, y:x*y), p1, p2
        if len(l1)==len(l2):
            yield check_coefficient_addition, b, l1, l2

def test_scalar_operations():
    for (c, l) in [(1, [1,2,3]),
                   (10, [1,2,3]),
                   (0.1, [1,2,3]),
                   (3, [2]),
                   (0, [2,5]),
                   (3, [])]:
        b = LagrangeBasis()
        p = b.polynomial(l)
        yield check_scalar_operation, (lambda c, p: c*p), c, p
        yield check_scalar_operation, (lambda c, p: p*c), c, p
        yield check_scalar_operation, (lambda c, p: p+c), c, p
        yield check_scalar_operation, (lambda c, p: c+p), c, p
        yield check_scalar_operation, (lambda c, p: p-c), c, p
        yield check_scalar_operation, (lambda c, p: c-p), c, p
        if c!=0:
            yield check_scalar_operation, (lambda c, p: p/c), c, p

def test_deriv_product():
    """Test that the product rule holds.

    If an operator is linear and respects the product
    rule, then it is the derivative operator (on polynomials,
    at least).
    """
    for (l1,l2) in [([1,2,3],[4,5,6]),
                    ([1,2,3],[1]),
                    ([1],[1]),
                    ([],[1,2,3]),
                    ([],[])]:
        b = LagrangeBasis()
        p1 = b.polynomial(l1)
        p2 = b.polynomial(l2)
        yield check_derivative_linearity, p1, p2
        yield check_product_rule, p1, p2

def test_power():
    b = LagrangeBasis()
    p = b.polynomial([0,1])
    for i in range(16):
        assert equal_by_values(p**i, reduce(lambda x, y: x*y, [p]*i, b.one()))


def test_from_roots():
    r = [1,3,8,12,-7]

    p = lagrange_from_roots(r)
    assert_array_almost_equal(p(r),0)

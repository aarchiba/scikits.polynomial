import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from poly import Polynomial, equal_by_values
from power import PowerBasis
from cheb import ChebyshevBasis

from test_poly import check_example, \
        check_operation, \
        check_operation, \
        check_coefficient_addition, \
        check_scalar_operation, \
        check_divmod, \
        check_product_rule, \
        check_derivative_linearity

def test_examples():
    for (l, x, y) in [([3],0,3),
                      ([], 1, 0),
                      ([0,1], 0.5, 0.5),
                      ([0,0,1], 0, -1),
                      ([0,0,1], 1, 1),
                      ([0,0,1], -1, 1),
                      ]:
        b = ChebyshevBasis()
        p = Polynomial(b, l)
        yield check_example, p, x, y

def test_operations():
    for (l1, l2) in [([], [1,2,3]),
                     ([1,2], [3,4]),
                     ([1], [1,2,3,4]),
                     ([0,0,0,0,1], [1,2,3,4]),
                     ([], [])]:
        b = ChebyshevBasis()
        p1 = Polynomial(b,l1)
        p2 = Polynomial(b,l2)
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
        b = ChebyshevBasis()
        p = Polynomial(b,l)
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
        b = ChebyshevBasis()
        p1 = Polynomial(b,l1)
        p2 = Polynomial(b,l2)
        yield check_derivative_linearity, p1, p2
        yield check_product_rule, p1, p2

def test_power():
    p = Polynomial(ChebyshevBasis(),[0,1])
    for i in range(16):
        assert equal_by_values(p**i, reduce(lambda x, y: x*y, [p]*i, Polynomial(ChebyshevBasis(), [1])))


def test_from_roots():
    r = [1,3,8,12,-7]

    p = ChebyshevBasis().from_roots(r)
    assert_array_almost_equal(p(r),0)

def test_interval_conversion():
    p = Polynomial(ChebyshevBasis(),[0,0,1])
    q = ChebyshevBasis(interval=(1,10)).convert(p)

    assert equal_by_values(p,q)
def test_interval():
    p = Polynomial(ChebyshevBasis(),[0,1])
    q = Polynomial(ChebyshevBasis(interval=(0,1)),[0.5,0.5])
    assert equal_by_values(p,q)

    assert equal_by_values(p,q)




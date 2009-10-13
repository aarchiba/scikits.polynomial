import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from poly import Polynomial, equal_by_values
from power import PowerBasis
from cheb import ChebyshevBasis

from test_poly import check_example, check_standard

def test_examples():
    for (l, x, y) in [([3],0,3),
                      ([], 1, 0),
                      ([0,1], 0.5, 0.5),
                      ([0,0,1], 0, -1),
                      ([0,0,1], 1, 1),
                      ([0,0,1], -1, 1),
                      ]:
        b = ChebyshevBasis()
        p = b.polynomial(l)
        yield check_example, p, x, y

def test_standard():
    for t in check_standard(ChebyshevBasis()):
        yield t
    for t in check_standard(ChebyshevBasis(interval=(3,-2))):
        yield t

def test_from_roots():
    r = [1,3,8,12,-7]

    p = ChebyshevBasis().from_roots(r)
    assert_array_almost_equal(p(r),0)

def test_interval_conversion():
    p = ChebyshevBasis().polynomial([0,0,1])
    q = ChebyshevBasis(interval=(1,10)).convert(p)

    assert equal_by_values(p,q)
def test_interval():
    p = ChebyshevBasis().polynomial([0,1])
    q = ChebyshevBasis(interval=(0,1)).polynomial([0.5,0.5])
    assert equal_by_values(p,q)

    assert equal_by_values(p,q)




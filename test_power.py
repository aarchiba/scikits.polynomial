import numpy as np
from numpy.testing import assert_almost_equal

from poly import equal_by_values
from power import PowerBasis

from test_poly import check_example, check_standard

def test_standard():
    for t in check_standard(PowerBasis()):
        yield t
    for t in check_standard(PowerBasis(-4)):
        yield t

def test_examples():
    for (l, x, y) in [([3],0,3),
                      ([], 1, 0),
                      ([1,0,-1], -1, 0),
                      ([0,0,0,1], 3, 27)]:
        p = PowerBasis().polynomial(l)
        yield check_example, p, x, y

def test_center():
    p1 = PowerBasis(-1).polynomial([0,0,0,1])
    p2 = PowerBasis().polynomial([1,3,3,1])
    assert equal_by_values(p1,p2)

def test_convert_center():
    p1 = PowerBasis().polynomial([1,2,3,4,5])
    assert equal_by_values(p1, PowerBasis(3).convert(p1))

def test_deriv_sample():
    p = PowerBasis().polynomial([1,0,-1])
    assert_almost_equal(p.derivative()(1),(-2))
    assert_almost_equal(p.derivative()(5),(-10))


import numpy as np
from numpy.testing import assert_almost_equal

from poly import Polynomial, equal_by_values
from power import PowerBasis

def test_examples():
    for (p, x, y) in [([3],0,3),
                      ([], 1, 0),
                      ([1,0,-1], -1, 0),
                      ([0,0,0,1], 3, 27)]:
        yield check_example, p, x, y

def check_example(p,x,y):
    p = Polynomial(PowerBasis(),p)
    assert_almost_equal(p(x),y)

def test_equal_by_values():
    p1 = Polynomial(PowerBasis(),[1,2,3,4])
    p2 = Polynomial(PowerBasis(),[1,2,3])
    assert equal_by_values(p1, p1)
    assert not equal_by_values(p1, p2)

def test_center():
    p1 = Polynomial(PowerBasis(-1),[0,0,0,1])
    p2 = Polynomial(PowerBasis(),[1,3,3,1])
    assert equal_by_values(p1,p2)

def test_convert_center():
    p1 = Polynomial(PowerBasis(),[1,2,3,4,5])
    assert equal_by_values(p1, PowerBasis(3).convert(p1))

def test_operations():
    for (l1, l2) in [([], [1,2,3]),
                     ([1,2], [3,4]),
                     ([1], [1,2,3,4]),
                     ([0,0,0,0,1], [1,2,3,4]),
                     ([], [])]:
        yield check_operation, (lambda x, y:x+y), l1, l2
        yield check_operation, (lambda x, y:x-y), l1, l2
        yield check_operation, (lambda x, y:x*y), l1, l2
        yield check_coefficient_addition, l1, l2

def check_operation(op, l1, l2):
    b = PowerBasis()
    p1 = Polynomial(b,l1)
    p2 = Polynomial(b,l2)
    for x in [-1,0,0.3,1]:
        assert_almost_equal(op(p1(x),p2(x)),op(p1,p2)(x))

def check_coefficient_addition(l1, l2):
    b = PowerBasis()
    p1 = Polynomial(b,l1)
    p2 = Polynomial(b,l2)
    n = max(len(l1),len(l2))
    c = np.zeros(n)
    c[:len(l1)] += l1
    c[:len(l2)] += l2
    p = Polynomial(b,c)
    assert equal_by_values(p1+p2, p)

def test_scalar_operations():
    for (c, l) in [(1, [1,2,3]),
                   (10, [1,2,3]),
                   (0.1, [1,2,3]),
                   (3, [2]),
                   (3, [])]:
        yield check_scalar_operation, (lambda c, p: c*p), c, l
        yield check_scalar_operation, (lambda c, p: p*c), c, l
        if c!=0:
            yield check_scalar_operation, (lambda c, p: p/c), c, l

def check_scalar_operation(op, c, l):
    b = PowerBasis()
    p = Polynomial(b,l)
    for x in [-1,0,0.3,1]:
        assert_almost_equal(op(c,p(x)),op(c,p)(x))

# FIXME: test for division by scalar zero

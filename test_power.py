import numpy as np
from numpy.testing import assert_almost_equal

from poly import Polynomial, equal_by_values
from power import PowerBasis

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
                      ([1,0,-1], -1, 0),
                      ([0,0,0,1], 3, 27)]:
        p = Polynomial(PowerBasis(),l)
        yield check_example, p, x, y

def test_center():
    p1 = Polynomial(PowerBasis(-1),[0,0,0,1])
    p2 = Polynomial(PowerBasis(),[1,3,3,1])
    assert equal_by_values(p1,p2)

def test_convert_center():
    p1 = Polynomial(PowerBasis(),[1,2,3,4,5])
    assert equal_by_values(p1, PowerBasis(3).convert(p1))

def test_operations():
    b = PowerBasis()
    for (l1, l2) in [([], [1,2,3]),
                     ([1,2], [3,4]),
                     ([1], [1,2,3,4]),
                     ([0,0,0,0,1], [1,2,3,4]),
                     ([], [])]:
        p1 = Polynomial(b,l1)
        p2 = Polynomial(b,l2)
        yield check_operation, (lambda x, y:x+y), p1, p2
        yield check_operation, (lambda x, y:x-y), p1, p2
        yield check_operation, (lambda x, y:x*y), p1, p2
        yield check_coefficient_addition, b, l1, l2

def test_scalar_operations():
    b = PowerBasis()
    for (c, l) in [(1, [1,2,3]),
                   (10, [1,2,3]),
                   (0.1, [1,2,3]),
                   (3, [2]),
                   (0, [2,5]),
                   (3, [])]:
        p = Polynomial(b,l)
        yield check_scalar_operation, (lambda c, p: c*p), c, p
        yield check_scalar_operation, (lambda c, p: p*c), c, p
        yield check_scalar_operation, (lambda c, p: p+c), c, p
        yield check_scalar_operation, (lambda c, p: c+p), c, p
        yield check_scalar_operation, (lambda c, p: p-c), c, p
        yield check_scalar_operation, (lambda c, p: c-p), c, p
        if c!=0:
            yield check_scalar_operation, (lambda c, p: p/c), c, p

# FIXME: test for division by scalar zero
# Should division by zero produce a polynomial with NaN/Inf coefficients?
# If so, what about division by a zero polynomial? We want to make sure
# we get the same behaviour from all of p/0., p/0, p/Polynomial(b,[]), 
# and p/Polynomial(b,[0]).

# FIXME: allow automatic basis conversion of polynomials that are
# obviously constant or zero? 

def test_divmod():

    for (l1,l2) in [([1,0,0,1],[1,1]),
                    ([-1,-4,0,0],[1,4]),
                    ([1,1],[1,0,0,1]),
                    ([1,0,0,1],[1,0,0,1]),
                    ([1,0,0,1],[1])]:
        yield check_divmod, Polynomial(PowerBasis(),l1), Polynomial(PowerBasis(),l1)
        
def test_deriv_sample():
    p = Polynomial(PowerBasis(), [1,0,-1])
    assert_almost_equal(p.derivative()(1),(-2))
    assert_almost_equal(p.derivative()(5),(-10))

def test_deriv_product():
    """Test that the product rule holds.

    If an operator is linear and respects the product
    rule, then it is the derivative operator (on polynomials,
    at least).
    """
    b = PowerBasis()
    for (l1,l2) in [([1,2,3],[4,5,6]),
                    ([1,2,3],[1]),
                    ([1],[1]),
                    ([],[1,2,3]),
                    ([],[])]:
        p1 = Polynomial(b,l1)
        p2 = Polynomial(b,l2)
        yield check_product_rule, p1, p2
        yield check_derivative_linearity, p1, p2


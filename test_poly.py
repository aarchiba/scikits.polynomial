import numpy as np

import poly


def test_eval():
    yield check_eval, [], 5, 0
    yield check_eval, [3], 5, 3
    yield check_eval, [3,2], 5, 13
    yield check_eval, [3,2,-1], 5, -12

def check_eval(l,x,r):
    assert np.abs(poly.PowerBasisPolynomial(l)(x)-r)<1e-13

def test_add():
    yield check_add, [1,3], [2,-4,-2], -4
    yield check_add, [], [2,-4,-2], -4
    yield check_add, [2], [2], -4
    yield check_add, [2,1], [-2,-1], -4

def check_add(l1, l2, x):
    p1, p2 = poly.PowerBasisPolynomial(l1), poly.PowerBasisPolynomial(l2)
    assert np.abs(p1(x)+p2(x)-(p1+p2)(x))<1e-13
    assert np.abs(p1(x)+p2(x)-(p2+p1)(x))<1e-13

def test_scalar_mul():
    yield check_scalar_mul, [2,1], 4, -4
    yield check_scalar_mul, [], 4, -4
    yield check_scalar_mul, [2,0,1], 4, -4
    yield check_scalar_mul, [2,1,0], 4, -4

def check_scalar_mul(l, c, x):
    p = poly.PowerBasisPolynomial(l)
    assert np.abs(c*p(x)-(c*p)(x))<1e-13
    assert np.abs(c*p(x)-(p*c)(x))<1e-13


def test_sub():
    yield check_sub, [1,3], [2,-4,-2], -4
    yield check_sub, [], [2,-4,-2], -4
    yield check_sub, [2], [2], -4
    yield check_sub, [2,1], [-2,-1], -4

def check_sub(l1, l2, x):
    p1, p2 = poly.PowerBasisPolynomial(l1), poly.PowerBasisPolynomial(l2)
    assert np.abs(p1(x)-p2(x)-(p1-p2)(x))<1e-13

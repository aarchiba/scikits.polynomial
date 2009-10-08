import numpy as np

from poly import PowerBasis, LagrangeBasis, Polynomial


def test_eval():
    yield check_eval, PowerBasis(), [], 5, 0
    yield check_eval, PowerBasis(), [3], 5, 3
    yield check_eval, PowerBasis(), [3,2], 5, 13
    yield check_eval, PowerBasis(), [3,2,-1], 5, -12

def check_eval(basis,l,x,r):
    assert np.abs(Polynomial(basis,l)(x)-r)<1e-13

def test_add():
    yield check_add, PowerBasis(), [1,3], [2,-4,-2], -4
    yield check_add, PowerBasis(), [], [2,-4,-2], -4
    yield check_add, PowerBasis(), [2], [2], -4
    yield check_add, PowerBasis(), [2,1], [-2,-1], -4

def check_add(basis, l1, l2, x):
    p1, p2 = Polynomial(basis,l1), Polynomial(basis,l2)
    assert np.abs(p1(x)+p2(x)-(p1+p2)(x))<1e-13
    assert np.abs(p1(x)+p2(x)-(p2+p1)(x))<1e-13

def test_scalar_mul():
    yield check_scalar_mul, PowerBasis(), [2,1], 4, -4
    yield check_scalar_mul, PowerBasis(), [], 4, -4
    yield check_scalar_mul, PowerBasis(), [2,0,1], 4, -4
    yield check_scalar_mul, PowerBasis(), [2,1,0], 4, -4

def check_scalar_mul(basis, l, c, x):
    p = Polynomial(basis,l)
    assert np.abs(c*p(x)-(c*p)(x))<1e-13
    assert np.abs(c*p(x)-(p*c)(x))<1e-13


def test_sub():
    yield check_sub, PowerBasis(), [1,3], [2,-4,-2], -4
    yield check_sub, PowerBasis(), [], [2,-4,-2], -4
    yield check_sub, PowerBasis(), [2], [2], -4
    yield check_sub, PowerBasis(), [2,1], [-2,-1], -4

def check_sub(basis, l1, l2, x):
    p1, p2 = Polynomial(basis,l1), Polynomial(basis,l2)
    assert np.abs(p1(x)-p2(x)-(p1-p2)(x))<1e-13


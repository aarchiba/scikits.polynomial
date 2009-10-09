import numpy as np

from poly import Polynomial
from power import PowerBasis
from lagrange import LagrangeBasis
from cheb import ChebyshevBasis


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

def test_mul():
    yield check_add, PowerBasis(), [1,3], [2,-4,-2], -4
    yield check_add, PowerBasis(), [], [2,-4,-2], -4
    yield check_add, PowerBasis(), [2], [2], -4
    yield check_add, PowerBasis(), [2,1], [-2,-1], -4

def check_mul(basis, l1, l2, x):
    p1, p2 = Polynomial(basis,l1), Polynomial(basis,l2)
    assert np.abs(p1(x)*p2(x)-(p1*p2)(x))<1e-13
    assert np.abs(p1(x)*p2(x)-(p2*p1)(x))<1e-13

def test_scalar_mul():
    yield check_scalar_mul, PowerBasis(), [2,1], 4, -4
    yield check_scalar_mul, PowerBasis(), [], 4, -4
    yield check_scalar_mul, PowerBasis(), [2,0,1], 4, -4
    yield check_scalar_mul, PowerBasis(), [2,1,0], 4, -4

def check_scalar_mul(basis, l, c, x):
    p = Polynomial(basis,l)
    assert np.abs(c*p(x)-(c*p)(x))<1e-13
    assert np.abs(c*p(x)-(p*c)(x))<1e-13

def test_deriv_power():
    p = Polynomial(PowerBasis(), [1,0,-1])
    assert np.abs(p.derivative()(1)-(-2))<1e-10
    assert np.abs(p.derivative()(5)-(-10))<1e-10

def test_deriv_lagrange():
    p = Polynomial(LagrangeBasis([-1,0,1]), [1,0,-1])
    assert np.abs(p.derivative()(1)-(-1))<1e-10
    assert np.abs(p.derivative()(5)-(-1))<1e-10

def test_sub():
    yield check_sub, PowerBasis(), [1,3], [2,-4,-2], -4
    yield check_sub, PowerBasis(), [], [2,-4,-2], -4
    yield check_sub, PowerBasis(), [2], [2], -4
    yield check_sub, PowerBasis(), [2,1], [-2,-1], -4

def check_sub(basis, l1, l2, x):
    p1, p2 = Polynomial(basis,l1), Polynomial(basis,l2)
    assert np.abs(p1(x)-p2(x)-(p1-p2)(x))<1e-13

def test_lagrange_basics():
    b = LagrangeBasis([-1,1])
    p1 = [0,1]
    p2 = [1,-1]

    yield check_eval, b, p2, 0, 0
    yield check_add, b, p1, p2, 0.3
    yield check_mul, b, p1, p2, 0.3


def test_convert():
    for l in [[], [8], [3,1], [1,0,0,0,1]]:
        yield check_convert, PowerBasis(), PowerBasis(), l
        yield check_convert, PowerBasis(), LagrangeBasis(), l
        yield check_convert, LagrangeBasis(), PowerBasis(), l
        yield check_convert, PowerBasis(0.2), PowerBasis(5), l
        yield check_convert, LagrangeBasis(), LagrangeBasis([-1,0.3,1]), l
        yield check_convert, LagrangeBasis(), ChebyshevBasis(), l
        yield check_convert, ChebyshevBasis(), ChebyshevBasis((0,1)), l

def check_convert(b1, b2, l):
    p1 = Polynomial(b1,l)
    p2 = b2.convert(p1)
    for x in [-1,-0.3,0,0.7,1,np.pi]:
        assert np.abs(p1(x)-p2(x))<1e-8

def test_vectorized():
    yield check_vectorized, PowerBasis()
    yield check_vectorized, LagrangeBasis()

def check_vectorized(b):
    p = Polynomial(b,[1,0,1])
    for shape in [(), (1,), (10,), (2,3), (2,3,5)]:
        z = np.zeros(shape)
        assert p(z).shape == shape


def test_division_power():

    for (l1,l2) in [([1,0,0,1],[1,1]),
                    ([1,1],[1,0,0,1]),
                    ([1,0,0,1],[1,0,0,1]),
                    ([1,0,0,1],[1])]:
        yield check_divmod, Polynomial(PowerBasis(),l1), Polynomial(PowerBasis(),l1)
        
def check_divmod(p1, p2):
    q,r = divmod(p1,p2)
    for x in np.arange(8):
        assert len(r.coefficients)<len(p2.coefficients)
        assert np.abs(p1(x)-(p2(x)*q(x)+r(x)))<1e-8
        assert p1//p2 == q
        assert p1%p2 == r

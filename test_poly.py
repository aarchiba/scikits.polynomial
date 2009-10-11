import numpy as np

from poly import Polynomial, polyfit, equal_by_values
from power import PowerBasis
from lagrange import LagrangeBasis
from cheb import ChebyshevBasis, _dct




def test_equal_by_values():
    p1 = Polynomial(PowerBasis(),[1,2,3,4])
    p2 = Polynomial(PowerBasis(),[1,2,3])
    assert equal_by_values(p1, p1)
    assert not equal_by_values(p1, p2)



# support tests used for all kinds of polynomials

def check_operation(op, p1, p2):
    for x in [-1,0,0.3,1]:
        assert_almost_equal(op(p1(x),p2(x)),op(p1,p2)(x))

def check_coefficient_addition(b, l1, l2):
    p1 = Polynomial(b,l1)
    p2 = Polynomial(b,l2)
    n = max(len(l1),len(l2))
    c = np.zeros(n)
    c[:len(l1)] += l1
    c[:len(l2)] += l2
    p = Polynomial(b,c)
    assert equal_by_values(p1+p2, p)



def check_scalar_operation(op, c, p):
    for x in [-1,0,0.3,1]:
        assert_almost_equal(op(c,p(x)),op(c,p)(x))


def check_divmod(p1, p2):
    q,r = divmod(p1,p2)
    for x in np.arange(8):
        assert len(r.coefficients)<len(p2.coefficients)
        assert np.abs(p1(x)-(p2(x)*q(x)+r(x)))<1e-8
        assert p1//p2 == q
        assert p1%p2 == r

def check_product_rule(p1, p2):
    assert equal_by_values((p1*p2).derivative(),
            p1*p2.derivative()+p1.derivative()*p2)



# specific tests that don't really belong here

def test_deriv_lagrange():
    p = Polynomial(LagrangeBasis([-1,0,1]), [1,0,-1])
    assert np.abs(p.derivative()(1)-(-1))<1e-10
    assert np.abs(p.derivative()(5)-(-1))<1e-10

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


def test_polyfit_exact():
    x = [-1,0,1]
    y = [5,6,7]
    p = polyfit(x,y,1)
    assert np.abs(p(0.5)-6.5)<1e-8

def test_antiderivative_lagrange():
    b = LagrangeBasis()
    b2 = LagrangeBasis([10,20,30])
    for l in [[], [1], [1,2,3], [1,2,3,4,5,6]]:
        yield check_antiderivative, Polynomial(b,l)
        yield check_antiderivative, Polynomial(b2,l)

def test_antiderivative_power():
    b = PowerBasis()
    b2 = PowerBasis(-13)
    for l in [[], [1], [1,2,3], [1,2,3,4,5,6]]:
        yield check_antiderivative, Polynomial(b,l)
        yield check_antiderivative, Polynomial(b2,l)
def test_antiderivative_cheb():
    b = ChebyshevBasis()
    b2 = ChebyshevBasis((-13,21))
    for l in [[], [1], [1,2,3], [1,2,3,4,5,6]]:
        yield check_antiderivative, Polynomial(b,l)
        yield check_antiderivative, Polynomial(b2,l)

def check_antiderivative(p):
    q = p.antiderivative().derivative()
    assert len(p.coefficients)==len(q.coefficients)
    for x in np.linspace(-1,1,len(p.coefficients)+1):
        assert np.abs(p(x)-q(x))<1e-8

def test_dct():
    x = np.array([1, 2.5, 3, -3.3, 5])
    n = len(x)
    y = _dct(x)
    y2 = np.zeros(n)
    for j in range(n):
        y2[j] = 2*np.sum(x*np.cos(np.pi*j*(np.arange(n)+0.5)/n))
    assert np.allclose(y, y2)

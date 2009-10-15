import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from lagrange import Polynomial, LagrangeBasis, bit_reverse, chebyshev_points_sequence, lagrange_from_roots, equal_by_values

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
        p = Polynomial(l,b)
        yield check_example, p, x, y

def test_interpolating():
    xs = np.linspace(-1,1,5)
    p1 = Polynomial([1,2,3,1],LagrangeBasis())
    p2 = Polynomial(p1(xs),LagrangeBasis(xs))
    assert equal_by_values(p1,p2)

def test_standard():
    for t in check_standard(LagrangeBasis(),coefficient_addition=False,division=False,roots=False):
        yield t
    for t in check_standard(LagrangeBasis([1,3,-2]),coefficient_addition=False,division=False,roots=False):
        yield t

def test_power():
    b = LagrangeBasis()
    p = Polynomial([0,1],b)
    for i in range(16):
        assert equal_by_values(p**i, reduce(lambda x, y: x*y, [p]*i, b.one()))


def test_from_roots():
    r = [1,3,8,12,-7]

    p = lagrange_from_roots(r)
    assert_array_almost_equal(p(r),0)


def check_example(p,x,y):
    assert_almost_equal(p(x),y)

def check_standard(b,division=True,coefficient_addition=True,roots=True):

    for (l1, l2) in [([], [1,2,3]),
                 ([1,2], [3,4]),
                 ([1], [1,2,3,4]),
                 ([0,0,0,0,1], [1,2,3,4]),
                 ([], [])]:
        p1 = Polynomial(l1,b)
        p2 = Polynomial(l2,b)
        yield check_operation, (lambda x, y:x+y), p1, p2
        yield check_operation, (lambda x, y:x-y), p1, p2
        yield check_operation, (lambda x, y:x*y), p1, p2
        if coefficient_addition:
            yield check_coefficient_addition, b, l1, l2

    for (c, l) in [(1, [1,2,3]),
                   (10, [1,2,3]),
                   (0.1, [1,2,3]),
                   (3, [2]),
                   (0, [2,5]),
                   (3, [])]:
        p = Polynomial(l,b)
        yield check_scalar_operation, (lambda c, p: c*p), c, p
        yield check_scalar_operation, (lambda c, p: p*c), c, p
        yield check_scalar_operation, (lambda c, p: p+c), c, p
        yield check_scalar_operation, (lambda c, p: c+p), c, p
        yield check_scalar_operation, (lambda c, p: p-c), c, p
        yield check_scalar_operation, (lambda c, p: c-p), c, p
        if c!=0:
            yield check_scalar_operation, (lambda c, p: p/c), c, p

    if division:
        for (l1,l2) in [([1,0,0,1],[1,1]),
                        ([-1,-4,0,0],[1,4]),
                        ([1,1],[1,0,0,1]),
                        ([1,0,0,1],[1,0,0,1]),
                        ([1,0,0,1],[1])]:
            yield check_divmod, Polynomial(l1,b), Polynomial(l1,b)
     
    for (l1,l2) in [([1,2,3],[4,5,6]),
                ([1,2,3],[1]),
                ([1],[1]),
                ([],[1,2,3]),
                ([],[])]:
        p1 = Polynomial(l1,b)
        p2 = Polynomial(l2,b)
        yield check_product_rule, p1, p2
        yield check_derivative_linearity, p1, p2

    if roots:
        for l in [[-1,0,1], [1], [], [1,1,-1,-1]]:
            yield check_real_roots, b, l

def check_operation(op, p1, p2):
    for x in [-1,0,0.3,1]:
        assert_almost_equal(op(p1(x),p2(x)),op(p1,p2)(x))

def check_coefficient_addition(b, l1, l2):
    p1 = Polynomial(l1,b)
    p2 = Polynomial(l2,b)
    n = max(len(l1),len(l2))
    c = np.zeros(n)
    c[:len(l1)] += l1
    c[:len(l2)] += l2
    p = Polynomial(c,b)
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

def check_derivative_linearity(p1, p2):
    f = 7.3
    assert equal_by_values((f*p1).derivative(),f*p1.derivative())
    assert equal_by_values((p1+p2).derivative(),p1.derivative()+p2.derivative())

def check_product_rule(p1, p2):
    assert equal_by_values((p1*p2).derivative(),
            p1*p2.derivative()+p1.derivative()*p2)

def check_from_roots(b,r):
    p = b.from_roots(r)
    assert_almost_equal(PowerBasis().convert(p).coefficients[-1],1)
    if len(r)!=0:
        xs = np.linspace(np.amin(r)-1,np.amax(r)+1,101)
        scale = np.sqrt(np.mean(p(xs)**2))
        assert_array_almost_equal(p(r)/scale,0)

def check_perfidious(b):
    # raise SkipTest
    X = b.X()
    perfidious = b.from_roots(np.arange(20)+1)
    scale = np.sqrt(np.mean(perfidious(np.linspace(1,20,1001))**2))
    for i in range(20):
        assert np.amax(np.abs(perfidious(np.arange(20)+1)))<1e-8*scale

def check_one(b):
    assert_array_almost_equal(b.one()(np.linspace(-1,1,10)), 1)
def check_X(b):
    xs = np.linspace(-1,1,10)
    assert_array_almost_equal(b.X()(xs), xs)

def check_real_roots(b,l):
    roots = np.copy(l)
    roots.sort()
    r = b.from_roots(l).roots()
    r.sort()
    assert_array_almost_equal(r,roots)


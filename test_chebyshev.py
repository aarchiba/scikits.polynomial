from __future__ import division

import numpy as np
import chebyshev as ch
from numpy.testing import *
from decimal import Decimal as Dec
from exceptions import TypeError, ValueError

def trim(x) :
    return ch.chebtrim(x, tol=1e-6)

T0 = [ 1]
T1 = [ 0,  1]
T2 = [-1,  0,   2]
T3 = [ 0, -3,   0,    4]
T4 = [ 1,  0,  -8,    0,   8]
T5 = [ 0,  5,   0,  -20,   0,   16]
T6 = [-1,  0,  18,    0, -48,    0,   32]
T7 = [ 0, -7,   0,   56,   0, -112,    0,   64]
T8 = [ 1,  0, -32,    0, 160,    0, -256,    0, 128]
T9 = [ 0,  9,   0, -120,   0,  432,    0, -576,   0, 256]

Tlist = [T0, T1, T2, T3, T4, T5, T6, T7, T8, T9]

def test__trim() :
    for i in range(5) :
        tgt = [1]
        res = ch._trim([1] + [0]*5)
        assert_equal(res, tgt)

def test__as_series() :
    # check exceptions
    assert_raises(ValueError, ch._as_series, [[]])
    assert_raises(ValueError, ch._as_series, [[[1,2]]])
    assert_raises(ValueError, ch._as_series, [[1],['a']])
    # check common types
    types = ['i', 'd', 'O']
    for i in range(len(types))  :
        for j in range(i) :
            ci = np.ones(1, types[i])
            cj = np.ones(1, types[j])
            [resi, resj] = ch._as_series([ci, cj]) 
            assert_(resi.dtype.char == resj.dtype.char)
            assert_(resj.dtype.char == types[i])

def test__cseries_to_zseries() :
    for i in range(5) :
        inp = np.array([2] + [1]*i, np.double)
        tgt = np.array([.5]*i + [2] + [.5]*i, np.double)
        res = ch._cseries_to_zseries(inp)
        assert_equal(res, tgt)

def test__zseries_to_cseries() :
    for i in range(5) :
        inp = np.array([.5]*i + [2] + [.5]*i, np.double)
        tgt = np.array([2] + [1]*i, np.double)
        res = ch._zseries_to_cseries(inp)
        assert_equal(res, tgt)

def test_chebadd() :
    for i in range(5) :
        for j in range(5) :
            msg = "At i=%d, j=%d" % (i,j)
            tgt = np.zeros(max(i,j) + 1)
            tgt[i] += 1
            tgt[j] += 1
            res = ch.chebadd([0]*i + [1], [0]*j + [1])
            assert_equal(trim(res), trim(tgt), err_msg=msg)

def test_chebsub() :
    for i in range(5) :
        for j in range(5) :
            msg = "At i=%d, j=%d" % (i,j)
            tgt = np.zeros(max(i,j) + 1)
            tgt[i] += 1
            tgt[j] -= 1
            res = ch.chebsub([0]*i + [1], [0]*j + [1])
            assert_equal(trim(res), trim(tgt), err_msg=msg)

def test_chebmul() :
    for i in range(5) :
        for j in range(5) :
            msg = "At i=%d, j=%d" % (i,j)
            tgt = np.zeros(i + j + 1)
            tgt[i + j] += .5
            tgt[abs(i - j)] += .5
            res = ch.chebmul([0]*i + [1], [0]*j + [1])
            assert_equal(trim(res), trim(tgt), err_msg=msg)

def test_chebdiv() :
    for i in range(5) :
        for j in range(5) :
            msg = "At i=%d, j=%d" % (i,j)
            ci = [0]*i + [1]
            cj = [0]*j + [1]
            tgt = ch.chebadd(ci, cj)
            quo, rem = ch.chebdiv(tgt, ci)
            res = ch.chebadd(ch.chebmul(quo, ci), rem)
            assert_equal(trim(res), trim(tgt), err_msg=msg)

def test_cheb2poly() :
    for i in range(10) :
        assert_equal(ch.cheb2poly([0]*i + [1]), Tlist[i])

def test_poly2cheb() :
    for i in range(10) :
        assert_equal(ch.poly2cheb(Tlist[i]), [0]*i + [1])

def test_chebint() :
    # check exceptions
    assert_raises(ValueError, ch.chebint, [0], -1)
    assert_raises(ValueError, ch.chebint, [0], 1, [0,0])
    # check single integration
    for i in range(5) :
        scl = i + 1
        pol = [0]*i + [1]
        tgt = [i] + [0]*i + [1/scl]
        chebpol = ch.poly2cheb(pol)
        chebint = ch.chebint(chebpol, m=1, k=[i])
        res = ch.cheb2poly(chebint)
        assert_almost_equal(trim(res), trim(tgt))
    # check multiple integrations with default k
    for i in range(5) :
        for j in range(2,5) :
            pol = [0]*i + [1]
            tgt = pol[:]
            for k in range(j) :
                tgt = ch.chebint(tgt, m=1)
            res = ch.chebint(pol, m=j)
            assert_almost_equal(trim(res), trim(tgt))
    # check multiple integrations with defined k
    for i in range(5) :
        for j in range(2,5) :
            pol = [0]*i + [1]
            tgt = pol[:]
            for k in range(j) :
                tgt = ch.chebint(tgt, m=1, k=[k])
            res = ch.chebint(pol, m=j, k=range(j))
            assert_almost_equal(trim(res), trim(tgt))

def test_chebder() :
    # check exceptions
    assert_raises(ValueError, ch.chebder, [0], -1)
    # check that zeroth deriviative does nothing
    for i in range(5) :
        tgt = [1] + [0]*i
        res = ch.chebder(tgt, m=0)
        assert_equal(trim(res), trim(tgt))
    # check that derivation is the inverse of integration
    for i in range(5) :
        for j in range(2,5) :
            tgt = [1] + [0]*i
            res = ch.chebder(ch.chebint(tgt, m=j), m=j)
            assert_almost_equal(trim(res), trim(tgt))

def test_chebval() :
    def f(x) :
        return x*(x**2 - 1)

    #check empty input
    assert_equal(ch.chebval([], 1).size, 0)
    #check normal input)
    for i in range(5) :
        tgt = 1
        res = ch.chebval(1, [0]*i + [1])
        assert_almost_equal(res, tgt)
        tgt = (-1)**i
        res = ch.chebval(-1, [0]*i + [1])
        assert_almost_equal(res, tgt)
        zeros = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
        tgt = 0
        res = ch.chebval(zeros,  [0]*i + [1])
        assert_almost_equal(res, tgt)
    x = np.linspace(-1,1)
    tgt = f(x)
    res = ch.chebval(x, [0, -.25, 0, .25])
    assert_almost_equal(res, tgt)


    #check that shape is preserved
    for i in range(3) :
        dims = [2]*i
        x = np.zeros(dims)
        assert_equal(ch.chebval(x, [1]).shape, dims)
        assert_equal(ch.chebval(x, [1,0]).shape, dims)
        assert_equal(ch.chebval(x, [1,0,0]).shape, dims)

def test_chebfromroots() :
    res = ch.chebfromroots([])
    assert_almost_equal(trim(res), [1])
    for i in range(1,5) :
        roots = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
        tgt = [0]*i + [1]
        res = ch.chebfromroots(roots)*2**(i-1)
        assert_almost_equal(trim(res),trim(tgt))

def test_chebroots() :
    assert_almost_equal(ch.chebroots([1]), [])
    assert_almost_equal(ch.chebroots([1, 2]), [-.5])
    for i in range(2,5) :
        tgt = np.linspace(-1, 1, i)
        res = ch.chebroots(ch.chebfromroots(tgt))
        assert_almost_equal(trim(res), trim(tgt))

def test_chebfit() :
    def f(x) :
        return x*(x - 1)*(x - 2)
    # Test exceptions
    assert_raises(ValueError, ch.chebfit, [1],    [1],     -1)
    assert_raises(TypeError,  ch.chebfit, [[1]],  [1],      0)
    assert_raises(TypeError,  ch.chebfit, [],     [1],      0)
    assert_raises(TypeError,  ch.chebfit, [1],    [[[1]]],  0)
    assert_raises(TypeError,  ch.chebfit, [1, 2], [1],      0)
    assert_raises(TypeError,  ch.chebfit, [1],    [1, 2],   0)
    # Test fit
    x = np.linspace(0,2)
    y = f(x)
    coef, domain = ch.chebfit(x, y, 3)
    assert_equal(domain, [0, 2])
    assert_equal(len(coef), 4)
    assert_almost_equal(ch.chebval(x, coef, domain), y)
    coef, domain = ch.chebfit(x, y, 4, domain=[-5, 5])
    assert_equal(domain, [-5, 5])
    assert_equal(len(coef), 5)
    assert_almost_equal(ch.chebval(x, coef, domain), y)

def test_chebtrim() :
    coef = [2, -1, 1, 0]
    # Test exceptions
    assert_raises(ValueError, ch.chebtrim, coef, -1)
    # Test results
    assert_equal(ch.chebtrim(coef), coef[:-1])
    assert_equal(ch.chebtrim(coef, 1), coef[:-3])
    assert_equal(ch.chebtrim(coef, 2), [0])

def test_chebval_domain():
    p = [1,2,3,4]
    xs = np.linspace(-1,1,10)
    assert_array_almost_equal(ch.chebval(xs,p),ch.chebval(2*xs,p,domain=(-2,2)))
    assert_array_almost_equal(ch.chebval(xs,p),ch.chebval(xs/2.+0.5,p,domain=(0,1)))

if __name__ == "__main__" :
    test__trim()
    test__as_series()
    test__cseries_to_zseries()
    test__zseries_to_cseries()
    test_chebadd()
    test_chebsub()
    test_chebmul()
    test_chebdiv()
    test_cheb2poly()
    test_poly2cheb()
    test_chebint()
    test_chebder()
    test_chebval()
    test_chebfromroots()
    test_chebroots()
    test_chebfit()
    test_chebtrim()

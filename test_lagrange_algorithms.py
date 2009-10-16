import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal, assert_array_equal, assert_equal

from lagrange_algorithms import *

def check_shape(f, in_shape, out_shape):
    assert_array_equal(np.shape(f(np.zeros(in_shape))),out_shape)
    if in_shape==():
        assert_array_equal(np.shape(f(0.)),out_shape)
        
def check_output_dtype(f, x, dtype):
    assert np.asarray(f(x)).dtype == dtype

def check_lists(f, in_):
    assert_array_equal(f(list(in_)),f(np.array(in_)))




def test_weights_shape():
    for l in [0,1,2,10]:
        yield check_shape, weights, (l,), (l,)

def test_weights_list():
    for l in [[], [1], [1,2], [1,2,3]]:
        yield check_lists, weights, l

def test_weights_dtype():
    yield check_output_dtype, weights, [1,2], np.float
    yield check_output_dtype, weights, [1.,2.], np.float
    yield check_output_dtype, weights, [1,2+1j], np.complex
    yield check_output_dtype, weights, [1.,2.+1.j], np.complex

def test_weights_incremental():
    pts = np.arange(10)
    existing_weights = None
    for i in [1,2,3,5,10]:
        existing_weights = weights(pts[:i], existing_weights)
        assert_array_equal(existing_weights, weights(pts[:i]))

def test_weights_null():
    assert_array_equal(weights([]),[])

def test_evaluate_too_many_points():
    points = np.arange(10)
    values = np.arange(4)
    xs = np.linspace(0,1,20)
    assert_array_equal(evaluate(points,values,xs),
                       evaluate(points[:len(values)],values,xs))
def check_evaluate_values(f,points,xs):
    xs = np.asarray(xs)
    points = np.asarray(points)
    assert_array_almost_equal(f(xs), evaluate(points, f(points), xs))

def test_evaluate_values():
    for (f, points, xs) in [
            (lambda x: x**2+1, [0,1,2], np.linspace(0,2,10)),
            (lambda x: x**2+1, [0,1,2,3,4,5], np.linspace(0,2,10)),
            (lambda x: np.ones_like(x), [0], np.linspace(0,2,10)),
            (lambda x: 1+x, [0,1], np.linspace(0,2,10)),
            (lambda x: x**2+1, [0,1,2], np.linspace(0,2.j,10)),
            (lambda x: x**2+1, [0,1.j,2], np.linspace(0,2,10)),
            ]:
        yield check_evaluate_values, f, points, xs

def test_evaluate_null():
    assert_equal(evaluate([],[],0),0)
def test_evaluate_shape():
    for s in [(), (1,), (2,), (10,), (1,2), (2,2), (2,1), (3,5,1,1),
              (0,), (0,2), (3,0,2)]:
        yield check_shape, lambda x: evaluate([1,2,3],[4,5,6],x), s, s

def check_evaluation_matrix(points,values,x):
    assert_array_almost_equal(
            evaluate(points, values, x),
            np.dot(evaluation_matrix(points, x), values))
def test_evaluation_matrix():
    for (p, v) in [([1,2],[1,5]),
                   ([1], [2]),
                   ([], []),
                   ([1,2,3j],[1,2j,1]),
                   ]:
        yield check_evaluation_matrix, p, v, np.linspace(0,1,5)
    
    
    
def test_derivative_matrix():
    for (f,df,points) in [(lambda x: x**2, lambda x: 2*x, [0,1,2]),
                          (lambda x: 3*x**2-2*x+1, lambda x: 6*x-2, [0,1,2]),
                          (lambda x: x**5-2*x+1, lambda x: 5*x**4-2, range(6)),
                          (lambda x: x**5-2*x+1, lambda x: 5*x**4-2, range(10)),
                          ]:
        points = np.asarray(points)
        values = f(points)
        dvalues = df(points)
        D = derivative_matrix(points)
        assert_array_almost_equal(dvalues,np.dot(D,values))

def check_division(p1, p2):
    points = np.arange(20)
    xs = np.linspace(-1,1,20)
    q, r = divide(points, p1, p2)
    assert len(r)<len(p2)
    assert_array_almost_equal(evaluate(points,p1,xs), 
            evaluate(points,p2,xs)*evaluate(points,q,xs)+evaluate(points,r,xs))

def test_division():
    for (p1, p2) in [
            ([1,2,3],[1,2]),
            ([1,2,3,4],[1,2]),
            ([1,2],[1,2]),
            ([1,2],[1,2,3]),
            ([1,2],[1]),
            ]:
        yield check_division, p1, p2

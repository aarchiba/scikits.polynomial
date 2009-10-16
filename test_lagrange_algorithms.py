import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal, assert_array_equal

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

def test_evaluate_shape():
    for s in [(), (1,), (2,), (10,), (1,2), (2,2), (2,1), (3,5,1,1),
              (0,), (0,2), (3,0,2)]:
        yield check_shape, lambda x: evaluate([1,2,3],[4,5,6],x), s, s

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

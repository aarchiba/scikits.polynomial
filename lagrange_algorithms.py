import numpy as np


def _dtype(*arrays):
    for a in arrays:
        if np.iscomplexobj(a):
            return np.complex
    return np.float

def weights(points, existing_weights=None):
    """Compute the weights for barycentric interpolation.

    Given a collection of x values, this function computes the array
    of weights used in barycentric interpolation; if an array of 
    existing weights is provided, it is assumed to provide weights
    for the first points passed in.

    """
    dtype = _dtype(points, existing_weights)
    points = np.asarray(points,dtype=dtype)
    if len(points.shape)!=1:
        raise ValueError("weights() must be supplied a one-dimensional array of points")
    
    if existing_weights is None:
        existing_weights = np.ones(1,dtype)
    else:
        existing_weights = np.asarray(existing_weights)
    if len(existing_weights) >= len(points):
        return existing_weights[:len(points)] 

    w = np.empty(len(points),dtype=dtype)
    w[:len(existing_weights)] = 1./existing_weights
    for j in range(len(existing_weights),len(points)):
        w[:j] *= points[:j]-points[j]
        w[j] = np.prod(points[j]-points[:j])
    w **= -1
    return w

def evaluate(points, values, x, existing_weights=None):
    """Evaluate the interpolating polynomial by barycentric interpolation.

    """
    dtype = _dtype(points, values, x, existing_weights)
    points = np.asarray(points,dtype=dtype)
    if len(points.shape)!=1:
        raise ValueError("evaluate() must be supplied a one-dimensional array of points")
    values = np.asarray(values,dtype=dtype)
    if len(values.shape)<1 or values.shape[0]!=points.shape[0]:
        raise ValueError("values must have the same size along its first dimension as points")
    x = np.asarray(x,dtype=dtype)

    w = weights(points, existing_weights)
    if x.size == 0:
        return np.zeros(x.shape+values.shape[1:],dtype=values.dtype)
    c = x[...,np.newaxis]-points
    z = c==0
    c[z] = 1
    c = w/c
    p = (np.dot(c,values)/
            np.sum(c,axis=-1)[(Ellipsis,)+(np.newaxis,)*(len(values.shape)-1)])
    # Now fix where x==some xi
    r = np.nonzero(z)
    if len(r)==1: # evaluation at a scalar
        if len(r[0])>0: # equals one of the points
            p = values[r[0][0]]
    else:
        p[r[:-1]] = values[r[-1]]
    return p

def derivative_matrix(points, existing_weights=None):
    dtype = _dtype(points, existing_weights)
    points = np.asarray(points,dtype=dtype)

    if len(points.shape)!=1:
        raise ValueError("derivative_matrix() must be supplied a one-dimensional array of points")

    n = len(points)

    w = weights(points, existing_weights)

    D = (w/w[:,np.newaxis])/(points[:,np.newaxis]-points)
    D[np.arange(n),np.arange(n)] = 0
    D[np.arange(n),np.arange(n)] = -np.sum(D,axis=1)
    return D


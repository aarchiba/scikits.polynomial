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
    if len(values.shape)<1 or values.shape[0]>points.shape[0]:
        raise ValueError("Not enough points specified for values")
    points = points[:values.shape[0]]
    x = np.asarray(x,dtype=dtype)

    w = weights(points, existing_weights)
    if x.size == 0 or len(points)==0:
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

def evaluation_matrix(points, x, existing_weights=None):
    """Construct the "evaluation matrix" for a set of points.

    The evaluation matrix maps a collection of values, one for each point,
    to the value of the interpolating polynomial at some other points. 
    Put another way, the evaluation matrix is what you would get by
    successively interpolating (1,0,0,...,0), (0,1,0,...,0), ... (0,0,0,...,1).

    """
    dtype = _dtype(points, x, existing_weights)
    points = np.asarray(points,dtype=dtype)
    if len(points.shape)!=1:
        raise ValueError("evaluate() must be supplied a one-dimensional array of points")
    x = np.asarray(x,dtype=dtype)

    w = weights(points, existing_weights)
    if x.size == 0:
        return np.zeros(x.shape+(len(points),),dtype=values.dtype)
    c = x[...,np.newaxis]-points
    z = c==0
    c[z] = 1
    c = w/c
    p = c/np.sum(c,axis=-1)[...,np.newaxis]
    # Now fix where x==some xi
    r = np.nonzero(z)
    if len(r)==1: # evaluation at a scalar
        if len(r[0])>0: # equals one of the points
            p = np.zeros(len(points),dtype=dtype)
            p[r[0][0]] = 1
    else:
        p[r[:-1]] = 0
        p[r] = 1
    return p

def derivative_matrix(points, existing_weights=None):
    """Construct the derivative matrix.

    The derivative matrix maps a collection of values, one for each point,
    to the derivative of the interpolating polynomial evaluated at each
    point. 

    """
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

def divide(points, p1_values, p2_values, p1_degree=None, p2_degree=None, 
        rcond=-1):
    """Divide two polynomials to give a quotient and remainder.

    This code does the division by a brute-force approach, constructing
    a matrix mapping (q,r) -> q*p2+r and then doing a least-squares fit
    for q and r. This should be numerically stable but slow.
    """
    dtype = _dtype(points, p1_values, p2_values)

    points = np.asarray(points, dtype=dtype)
    p1_values = np.asarray(p1_values, dtype=dtype)
    p2_values = np.asarray(p2_values, dtype=dtype)

    if p1_degree is None:
        p1_degree = len(p1_values)-1
    if p2_degree is None:
        p2_degree = len(p2_values)-1

    p2_values = evaluate(points[:len(p2_values)], p2_values, 
                         points[:len(p1_values)])
    
    if len(points)-1<max(p1_degree,p2_degree):
        raise ValueError("not enough points to specify a polynomial of degree %d" % max(p1_degree,p2_degree))

    q_degree = p1_degree-p2_degree
    q_eval_matrix = evaluation_matrix(points[:q_degree+1],
                                      points[:len(p1_values)])
    r_degree = p2_degree-1
    r_eval_matrix = evaluation_matrix(points[:r_degree+1],
                                      points[:len(p1_values)])
    
    q_plus_r_matrix = np.hstack((q_eval_matrix*p2_values[:,np.newaxis],
                                 r_eval_matrix))
    qr, res, rk, s = np.linalg.lstsq(q_plus_r_matrix,p1_values,rcond=rcond)
    return qr[:q_degree+1], qr[q_degree+1:]

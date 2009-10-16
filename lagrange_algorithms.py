import numpy as np
import scipy.linalg


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
    # FIXME: allow complex values but real variables
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

def extend(points, values, n):
    """Extend the array values to length n by evaluating the interpolating polynomial.

    """
    points = np.asarray(points)
    values = np.asarray(values)
    vs = np.array(values.shape)
    if n>len(points):
        raise ValueError("Not enough values in points to interpolate %d values" % n)
    if vs[0]==n:
        return values
    elif vs[0]>n:
        raise ValueError("extend() cannot truncate values")
    vs[0] = n
    v = np.empty(vs, dtype=_dtype(values))
    v[:values.shape[0]] = values
    v[values.shape[0]:] = evaluate(points, values, points[values.shape[0]:n])

    return v

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

    if p1_degree<p2_degree:
        # division should be trivial
        return np.zeros(0,dtype=dtype), p1_values
    p2_values = extend(points, p2_values, len(p1_values))
    
    if len(points)-1<max(p1_degree,p2_degree):
        raise ValueError("not enough points to specify a polynomial of degree %d" % max(p1_degree,p2_degree))

    q_degree = p1_degree - p2_degree
    qr = np.dot(division_matrix(points[:len(p1_values)], p2_values, p2_degree),
                p1_values)

    return qr[:q_degree+1], qr[q_degree+1:]

def division_matrix(points, p_values, p_degree=None):
    """Compute the matrix for division by a polynomial.
    
    If p1 has values p1_values on points, then this function returns a matrix
    M for which M*p1_values = [q_values, r_values], that is, multiplying
    the values of p1 by M gives a vector of the values of the quotient 
    followed by the values of the remainder, each evaluated on just enough
    points to represent them.
    
    """
    dtype = _dtype(points, p_values)

    points = np.asarray(points, dtype=dtype)
    if p_degree is None:
        p_degree = len(p_values)-1
    p_values = extend(points, p_values, len(points))

    q_degree = len(points)-1-p_degree
    q_eval_matrix = evaluation_matrix(points[:q_degree+1],
                                      points[:len(points)])
    r_degree = p_degree-1
    r_eval_matrix = evaluation_matrix(points[:r_degree+1],
                                      points[:len(points)])
    
    q_plus_r_matrix = np.hstack((q_eval_matrix*p_values[:,np.newaxis],
                                 r_eval_matrix))
    
    return scipy.linalg.pinv(q_plus_r_matrix)

def companion_matrix(points, values):
    """Compute the companion matrix of a polynomial.

    The eigenvalues of the companion matrix are the roots of the polynomial.
    
    """
    dtype = _dtype(points, values)
    points = np.asarray(points, dtype=dtype)
    values = np.asarray(values, dtype=dtype)
    if len(points)<len(values):
        raise ValueError("Must have at least as many points as values")
    points = points[:len(values)]

    # Discard quotient, keep remainder
    M = division_matrix(points, values)[1:] 

    # FIXME: this can probably be combined with M without matrix multiplication
    # mX is multiplication by X
    mX = np.eye(len(values))[:,:len(values)-1]
    mX[-1,:] = evaluation_matrix(points[:len(values)-1],points[len(values)-1])
    mX *= points[:,np.newaxis]

    # M is multiplication by X mod p
    M = np.dot(M,mX)

    assert M.shape[0] == M.shape[1]
    return M



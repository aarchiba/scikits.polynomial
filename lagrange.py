import numpy as np
import numpy.linalg

from poly import Polynomial, Basis

def bit_reverse(n, denom):
    """Return the fraction whose binary expansion is the reverse of n

    If n's binary expansion is b_r b_(r-1) ... b_0, return m and d
    such that d is a power of two and m/d has binary expansion
    . b_0 b_1 ... b_r. This is a subrandom sequence, in the sense that
    any the sequence bit_reverse(1), bit_reverse(2), ..., bit_reverse(k)
    samples the interval (0,1) more uniformly than random numbers would.
    """
    if n>denom:
        raise ValueError("n must be less than denom")
    mask = denom//2
    v = 1
    r = 0
    while mask:
        r += (n & mask) and v
        mask //= 2
        v *= 2
    return r

def chebyshev_points_sequence(n):
    """The extrema of the Chebyshev polynomials, in a subrandom order.

    The order visits all the extrema of a polynomial of one degree in
    bit-reversed order, then approximately doubles the degree.
    """
    # FIXME: Make sure we're using the right points.
    s = 0
    denom = 1
    while s+denom<=n:
        s+=denom
        denom*=2
    
    m = bit_reverse(n-s, denom)
    return np.cos(np.pi*(2*m+1)/float(2*denom))

class LagrangeBasis(Basis):
    """This class represents a polynomial by its values at specified points.

    This is the natural representation when constructing the Lagrange
    interpolating polynomial, for example.

    The points chosen can be specified upon construction of a basis object.
    When further points are needed, for example when multiplying polynomials,
    the points are chosen to be Chebyhev points, which cluster at the edges
    of the given interval in a way that is optimal in terms of numerical 
    stability.

    This class is technically mutable, since it keeps track of which points 
    have been specified, but the modifications are invisible to users. The
    class is not currently thread-safe; it is probably best to have a separate
    copy per thread.
    """

    def __init__(self, initial_points=[], interval=None):
        """Construct a basis.

        A set of initial points to include in the basis can be specified.
        If interval is specified, then polynomials in this basis will 
        be fairly well-behaved on that interval. If it is not specified,
        it will use the range spanned by initial_points, if these are 
        available, or (-1,1) if they are not.
        """
        Basis.__init__(self)
        self.initial_points = tuple(initial_points)
        self.points = np.array(initial_points, dtype=np.float)
        if interval is None:
            if len(initial_points)>1:
                self.interval = np.amax(self.points), np.amin(self.points)
            elif len(initial_points)==1:
                self.interval = (initial_points[0]-1,initial_points[0]+1)
            else:
                self.interval = (-1,1)
        else:
            self.interval = interval
        self.initial_points_set = set(initial_points)
        self.first_chebyshev_point_not_tried = 0

    def __eq__(self, other):
        return isinstance(other, LagrangeBasis) and self.initial_points == other.initial_points
    def __hash__(self):
        return hash(tuple(self.initial_points)) # Do arrays hash?

    def extend_points(self, n):
        """Extend the internal list of points."""
        if n<=len(self.points):
            return
        a, b = self.interval
        new_points = []
        while n>len(self.points)+len(new_points):
            x = chebyshev_points_sequence(self.first_chebyshev_point_not_tried)
            x = x*(b-a)/2. + (a+b)/2.
            if x not in self.initial_points_set:
                new_points.append(x)
            self.first_chebyshev_point_not_tried += 1
        if new_points:
            points = np.zeros(n)
            points[:len(self.points)] = self.points
            points[len(self.points):] = np.array(new_points)
            self.points = points

    def extend(self, coefficients, n):
        """Extend a coefficient array.

        This function does two things: select new sample points if necessary,
        and evaluate the polynomial specified by coefficients at more points.
        """
        self.extend_points(n)
        
        new_coefficients = np.zeros(n)
        new_coefficients[:len(coefficients)] = coefficients
        if n>len(coefficients):
            new_coefficients[len(coefficients):] = self.evaluate(coefficients, self.points[len(coefficients):len(new_coefficients)])
        return new_coefficients

    def weights(self, n=None):
        if n is None:
            n=len(self.points)
        self.extend_points(n)
        # FIXME: might be worth caching the weights
        # FIXME: if we use Chebyshev points, there's an analytic formula
        w = np.zeros(n)
        w[0] = 1
        for j in range(1,n):
            w[:j] *= self.points[:j]-self.points[j]
            w[j] = np.prod(self.points[j]-self.points[:j])
        w = 1./w
        return w


    def evaluate(self, coefficients, x):
        """Evaluate a polynomial at a new point.

        This code uses barycentric interpolation, which is quite stable, 
        numerically. The weights are currently not cached between calls, 
        nor does the code use the analytic expression for the weights in
        the case that the points are Chebyshev points.
        """
        self.extend_points(len(coefficients))
        if not np.isscalar(x):
            x = np.asarray(x)
            r = np.zeros(x.shape)
            for (ix, v) in np.ndenumerate(x):
                r[ix] = self.evaluate(coefficients, v)
            return r


        # special case for when x is one of the given points
        if len(coefficients)==0:
            return 0.
        c = x==self.points[:len(coefficients)]
        if np.any(c):
            i = np.where(c)[0][0]
            return coefficients[i]

        w = self.weights(len(coefficients))
        wx = w/(x-self.points[:len(coefficients)])
        return np.sum(wx*coefficients)/np.sum(wx)

    def multiply(self, coefficients, other_coefficients):
        """Multiply two polynomials.

        This code simply extends the polynomials to a high enough order
        then does pointwise multiplication.
        """
        if len(coefficients)==0 or len(other_coefficients)==0:
            return np.zeros(0)
        l = len(coefficients)+len(other_coefficients)-1
        coefficients = self.extend(coefficients, l)
        other_coefficients = self.extend(other_coefficients, l)
        return coefficients*other_coefficients
    
    def derivative_matrix(self, n):
        self.extend_points(n)
        w = self.weights(n)
        x = self.points[:n]
        D = (w/w[:,np.newaxis])/(x[:,np.newaxis]-x)
        D[np.arange(n),np.arange(n)] = 0
        D[np.arange(n),np.arange(n)] = -np.sum(D,axis=1)
        return D

    def derivative(self, coefficients):
        if len(coefficients)==0:
            return np.zeros(0)

        c = np.dot(self.derivative_matrix(len(coefficients)),coefficients)
        return c[:-1]

    def antiderivative(self, coefficients):
        if len(coefficients)==0:
            return np.zeros(1)

        thresh = 1e-13

        coefficients = self.extend(coefficients,len(coefficients)+1)
        D = self.derivative_matrix(len(coefficients))
        U, s, Vh = numpy.linalg.svd(D)
        ss = 1./s
        ss[s<thresh*s[0]] = 0
        c = reduce(np.dot,(Vh.T,np.diag(ss),U.T,coefficients))
        return c

    def convert(self, polynomial):
        n = len(polynomial.coefficients)
        if n==0:
            return Polynomial(self, [])
        self.extend_points(n)
        return Polynomial(self, polynomial(self.points[:n]))

    def __repr__(self):
        return "<LagrangeBasis initial_points=%s>" % (self.initial_points,)


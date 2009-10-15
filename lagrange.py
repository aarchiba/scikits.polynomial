import numpy as np
import numpy.linalg

class Polynomial(object):
    """Polynomial in the Lagrange basis.

    Polynomials in this basis are represented by their values at
    certain points. Exactly which points are specified in their basis
    object, but normally the user specifies an interval and/or some
    initial points. If more points are needed they are chosen to be
    Chebyshev extrema, as these points are optimally distributed
    from the point of view of numerical stability.

    """
    def __init__(self, coefficients, basis):
        # Catch direct construction

        # FIXME: allow vector values and complex dtype
        self.coefficients = np.array(coefficients, dtype=np.float)
        if len(self.coefficients.shape)!=1:
            raise ValueError("Polynomial coefficients must be one-dimensional arrays; given coefficients of shape %s" % (self.coefficients.shape,))
        self.basis = basis

    def iscompatible(self,other):
        return (isinstance(other, self.__class__) and self.basis==other.basis)
    def __eq__(self,other):
        return (self.iscompatible(other)
                and np.all(self.coefficients==other.coefficients))
    def __call__(self, x):
        """Evaluate a polynomial at a new point.

        This code uses barycentric interpolation, which is quite stable, 
        numerically. The weights are currently not cached between calls, 
        nor does the code use the analytic expression for the weights in
        the case that the points are Chebyshev points.
        """
        if len(self.coefficients)==0:
            return np.zeros(np.shape(x))
        # should always be a no-op.
        self.basis.extend_points(len(self.coefficients))
        w = self.basis.weights(len(self.coefficients))

        def evaluate_scalar(x):
            # special case for when x is one of the given points
            # FIXME: could be more efficient
            c = x==self.basis.points[:len(self.coefficients)]
            if np.any(c):
                i = np.where(c)[0][0]
                return self.coefficients[i]

            wx = w/(x-self.basis.points[:len(self.coefficients)])
            return np.sum(wx*self.coefficients)/np.sum(wx)

        if not np.isscalar(x):
            x = np.asarray(x)
            r = np.zeros(x.shape)
            for (ix, v) in np.ndenumerate(x):
                r[ix] = evaluate_scalar(v)
            return r
        else:
            return evaluate_scalar(x)

    def raise_degree(self, n):
        """Raise the degree to n.

        Lagrange polynomials are normally represented by their values on a
        collection of m points, where m is their degree. This function 
        evaluates them at more points, so that (for example) multiplication
        can simply be done pointwise.

        This function does two things: select new sample points if necessary,
        and evaluate the polynomial specified by coefficients at more points.
        Shortening the array is simply truncation.
        """
        if n<len(self.coefficients):
            raise ValueError("raise_degree cannot reduce degrees.")
        elif n==len(self.coefficients):
            return self
        self.basis.extend_points(n)
        
        new_coefficients = np.zeros(n)
        new_coefficients[:len(self.coefficients)] = self.coefficients
        if n>len(self.coefficients):
            new_coefficients[len(self.coefficients):] = \
                    self(self.basis.points[len(self.coefficients):
                                           len(new_coefficients)])
        return Polynomial(new_coefficients, self.basis)


    def __add__(self, other):
        if isinstance(other,Polynomial):
            if not self.iscompatible(other):
                raise IncompatibleBasesError("Polynomials must be in the same basis to be added")
            l = max(len(self.coefficients),len(other.coefficients))
            return Polynomial(self.raise_degree(l).coefficients + 
                                         other.raise_degree(l).coefficients,
                              self.basis)
        else:
            if len(self.coefficients)==0:
                return Polynomial([other],self.basis)
            else:
                return Polynomial(self.coefficients+other, self.basis)
    def __radd__(self, other):
        return self+other

    def __mul__(self, other):
        """Multiply two polynomials, or a polynomial by a scalar.

        This code simply extends the polynomials to a high enough order
        then does pointwise multiplication.
        """
        if len(self.coefficients)==0:
            return self.basis.zero()
        if isinstance(other,Polynomial):
            if len(other.coefficients)==0:
                return self.basis.zero()
            # Don't need other polynomial to be compatible
            l = len(self.coefficients)+len(other.coefficients)-1
            s = self.raise_degree(l)
            return Polynomial(s.coefficients*other(self.basis.points[:l]), 
                    self.basis)
        else:
            return Polynomial(other*self.coefficients,self.basis)

    def __rmul__(self, other):
        return self*other

    def __pow__(self, power):
        """Raise to an integer power."""
        if power != int(power):
            raise ValueError("Can only raise polynomials to integer powers")
        power = int(power)
        if power<0:
            raise ValueError("Cannot raise polynomials to negative powers")
        if power==0:
            return self.basis.one()
        if len(self.coefficients)==0:
            return self.basis.zero()

        l = power*(len(self.coefficients)-1)+1
        p = self.raise_degree(l)
        return Polynomial(p.coefficients**power, self.basis)

    def __neg__(self):
        return Polynomial(-self.coefficients, self.basis)

    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return other + (-self)

    def derivative(self):
        if len(self.coefficients)==0:
            return self.basis.zero()

        D = self.basis.derivative_matrix(len(self.coefficients))
        return Polynomial(np.dot(D,self.coefficients)[:-1],self.basis)

    def antiderivative(self, coefficients):
        if len(coefficients)==0:
            return np.zeros(0)

        thresh = 1e-13

        coefficients = self.raise_degree(len(self.coefficients)+1).coefficients
        D = self.derivative_matrix(len(coefficients))
        U, s, Vh = numpy.linalg.svd(D)
        ss = 1./s
        ss[s<thresh*s[0]] = 0
        return Polynomial(reduce(np.dot,(Vh.T,np.diag(ss),U.T,coefficients)), self.basis)

    def __repr__(self):
        return "<Polynomial basis=%s coefficients=%s>" % (self.basis, self.coefficients)

    def divide(self, other, tol=None):
        """Polynomial division.

        Given P1 and P2 find Q and R so that P1 = Q*P2 + R and the degree
        of R is strictly less than the degree of P2. 

        In some representations this is an approximate process; for this 
        tol defines the tolerance.
        """
        if not self.iscompatible(other):
            raise IncompatibleBasesError("Polynomials in different bases cannot be divided")
        raise NotImplementedError
        return self.basis.polynomial(q), self.basis.polynomial(r)

    def __divmod__(self, other):
        if isinstance(other, Polynomial):
            return self.divide(other)
        else:
            return (self.basis.polynomial(self.coefficients/other), 
                    self.basis.polynomial([]))
            
    def __floordiv__(self, other):
        q, r = divmod(self, other)
        return q
    def __mod__(self, other):
        q, r = divmod(self, other)
        return r

    def __div__(self, other):
        return self.__truediv__(other)
    def __truediv__(self, other):
        if isinstance(other, Polynomial):
            raise ValueError("Cannot do true division by polynomials")
        return (1./other)*self
    def __rtruediv__(self,other):
        raise ValueError("Cannot do true division by polynomials")



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
    # FXME: Make sure we're using the right points.
    s = 0
    denom = 1
    while s+denom<=n:
        s+=denom
        denom*=2
    
    m = bit_reverse(n-s, denom)
    return np.cos(np.pi*(2*m+1)/float(2*denom))



class LagrangeBasis(object):
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
        # FIXME: do something about repeated points
        # At least raise an exception
        # In principle one could use them to allow Hermite interpolation
        # (i.e. the second occurrence signals a derivative value)
        self.polynomial_class = Polynomial
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
        self.first_chebyshev_point_not_tried = 0

    def zero(self):
        return Polynomial([],self)
    def one(self):
        return Polynomial([1],self)
    def X(self):
        self.extend_points(2)
        return Polynomial([self.points[0],self.points[1]],self)

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

            # Ensure new points aren't too close to old points
            if self.initial_points:
                d = (1./8)*abs(b-a)/float(self.first_chebyshev_point_not_tried+1)**2
                if min(*[abs(p-x) for p in self.initial_points])>d:
                    new_points.append(x)
            else:
                new_points.append(x)

            self.first_chebyshev_point_not_tried += 1
        if new_points:
            points = np.zeros(n)
            points[:len(self.points)] = self.points
            points[len(self.points):] = np.array(new_points)
            self.points = points

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

    def derivative_matrix(self, n):
        self.extend_points(n)
        w = self.weights(n)
        x = self.points[:n]
        D = (w/w[:,np.newaxis])/(x[:,np.newaxis]-x)
        D[np.arange(n),np.arange(n)] = 0
        D[np.arange(n),np.arange(n)] = -np.sum(D,axis=1)
        return D


    def convert(self, polynomial):
        n = len(polynomial.coefficients)
        if n==0:
            return self.zero()
        self.extend_points(n)
        return self.polynomial(polynomial(self.points[:n]))

    def __repr__(self):
        return "<LagrangeBasis initial_points=%s>" % (self.initial_points,)

standard_basis = LagrangeBasis()

def lagrange_from_roots(roots, interval=None):
    # FIXME: this could be much more efficient, constructing the
    # result more-or-less directly.
    ur = np.unique(roots)
    b = LagrangeBasis(np.unique(roots),interval=interval)
    X = b.X()
    r = b.one()
    for rt in roots:
        r *= X-rt
    return r

def polyfit(x, y, deg, basis=None):
    """Least-squares polynomial fit.

    This returns the polynomial p of degree at most deg that
    minimizes sum((p(x)-y)**2). This polynomial is represented in
    the given basis.

    """
    if basis is None:
        import lagrange
        basis = lagrange.LagrangeBasis(x)

    x = np.asarray(x)
    y = np.asarray(y)
    A = np.zeros((len(y),deg+1))
    for i in range(deg+1):
        pc = np.zeros(deg+1)
        pc[i] = 1
        A[:,i] = basis.polynomial(pc)(x)

    c, resids, rank, s = numpy.linalg.lstsq(A, y)
    p = 0
    return basis.polynomial(c)

def equal_by_values(p1, p2, interval=(-1,1), tol=1e-8):
    """Compare two polynomials by evaluating them at many points.

    Given two polynomials whose degrees are known to be less than n, 
    if they agree at n distinct points, then they are equal. This
    function tests polynomials for equality in this way. The points
    are selected to be in interval, and the values are considered
    equal if they differ by less than tol.
    """
    a, b = interval
    n = np.max(len(p1.coefficients), len(p2.coefficients))
    if n==0:
        return True
    xs = np.linspace(a,b,n)
    return np.all(np.abs(p1(xs)-p2(xs))<tol)



import numpy as np

class Polynomial:
    """Class for representing polynomials.

    This class represents a polynomial as a linear combination of
    basis functions. The basis functions 1, x, x**2, ... are perhaps
    the most familiar set of basis functions, but they can lead to
    very severe numerical problems. So this class keeps track of
    the set of basis functions used to represent this polynomial. Most
    calculations are actually delegated to the class representing the 
    basis. Calculations between polynomials represented in different 
    bases require explicit conversion, since conversion between bases
    is often a numerically unstable operation.
    """

    def __init__(self, basis, coefficients):
        self.basis = basis
        self.coefficients = np.array(coefficients, dtype=np.float)

    def __call__(self, x):
        return self.basis.evaluate(self.coefficients, x)

    def __add__(self, other):
        if isinstance(other,Polynomial):
            if self.basis != other.basis:
                raise ValueError("Polynomials must be in the same basis to be added")
            l = max(len(self.coefficients),len(other.coefficients))
            c = (self.basis.extend(self.coefficients, l) 
                    + self.basis.extend(other.coefficients, l))
            return Polynomial(self.basis, c)
        else:
            return self + Polynomial(self.basis, [other])
    def __radd__(self, other):
        return self+other

    def __mul__(self, other):
        if isinstance(other,Polynomial):
            if self.basis != other.basis:
                raise ValueError("Polynomials must be in the same basis to be multiplied")
            c = self.basis.multiply(self.coefficients, other.coefficients)
            return Polynomial(self.basis, c)
        else:
            c = np.copy(self.coefficients)
            c *= other
            return Polynomial(self.basis, c)
    def __rmul__(self, other):
        return self*other

    def __sub__(self, other):
        return self + (-1.)*other


class Basis:
    """Abstract base class for polynomial bases.

    This class exists primarily to make explicit the interface
    that a polynomial basis must support.
    """

    def __init__(self):
        pass

    def extend(self, coefficients, n):
        """Extend a coefficient list to length n.

        A polynomial of degree m is naturally represented by a vector
        of coefficients of length m. But it is often useful to be able
        to extend the vector of coefficients to length n; for the power
        basis, this just amounts to padding with zeros, but for other
        bases this may be a nontrivial operation.
        """
        raise NotImplementedError

    def evaluate(self, coefficients, x):
        """Evaluate a polynomial at x.

        It should be possible for x to be an array of arbitrary shape; the
        result will have the same shape.
        """
        raise NotImplementedError

    def multiply(self, coefficients, other_coefficients):
        """Compute the coefficients of the product."""
        raise NotImplementedError

    def __eq__(self, other):
        """Test for equality."""
        raise NotImplementedError
    def __ne__(self, other):
        return not self==other
    def __hash__(self):
        raise NotImplementedError


class PowerBasis(Basis):
    """The basis 1, x, x**2, ..."""

    def __init__(self,center=0):
        Basis.__init__(self)
        self.center = center

    def extend(self, coefficients, n):
        if n<len(coefficients):
            raise ValueError("extend can only make coefficient arrays longer")
        z = np.zeros(n)
        z[:len(coefficients)] = coefficients
        return z

    def evaluate(self, coefficients, x):
        x = np.asarray(x,dtype=float) - self.center
        r = np.zeros(x.shape)
        if len(coefficients)==0:
            return r
        r += coefficients[-1]
        for c in coefficients[-2::-1]:
            r *= x
            r += c
        return r

    def multiply(self, coefficients, other_coefficients):
        if len(coefficients)==0 or len(other_coefficients)==0:
            return np.zeros(0)
        c = np.zeros(len(coefficients)+len(other_coefficients)-1)
        for i, ci in enumerate(coefficients):
            c[i:i+len(coefficients)] += ci*other_coefficients
        return c

    def __eq__(self, other):
        return isinstance(other, PowerBasis) and self.center == other.center
    def __hash__(self):
        return hash(self.center) # Include self's type?


def bit_reverse(n):
    """Return the fraction whose binary expansion is the reverse of n

    If n's binary expansion is b_r b_(r-1) ... b_0, return m and d
    such that d is a power of two and m/d has binary expansion
    . b_0 b_1 ... b_r. This is a subrandom sequence, in the sense that
    any the sequence bit_reverse(1), bit_reverse(2), ..., bit_reverse(k)
    samples the interval (0,1) more uniformly than random numbers would.
    """
    denom = 1
    while denom<=n:
        denom *= 2
    mask = denom//2
    v = 1
    r = 0
    while mask:
        r += (n & mask) and v
        mask //= 2
        v *= 2
    return r, denom

def chebyshev_points_sequence(n):
    """The extrema of the Chebyshev polynomials, in a subrandom order."""
    # FIXME: Make sure we're using the right points.
    m, denom = bit_reverse(n)
    return np.cos(np.pi*(2*m+1)/float(2*denom))

class LagrangeBasis:
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

    def extend(self, coefficients, n):
        """Extend a coefficient array.

        This function does two things: select new sample points if necessary,
        and evaluate the polynomial specified by coefficients at more points.
        """
        a, b = self.interval
        new_points = []
        while n>len(self.points)+len(new_points):
            x = chebyshev_points_sequence(self.first_chebyshev_point_not_tried)
            x = 2*x/(b-a) + (a+b)/2.
            if x not in self.initial_points_set:
                new_points.append(x)
            self.first_chebyshev_point_not_tried += 1
        if new_points:
            points = np.zeros(n)
            points[:len(self.points)] = self.points
            points[len(self.points):] = np.array(new_points)
            self.points = points
        
        new_coefficients = np.zeros(n)
        new_coefficients[:len(coefficients)] = coefficients
        if n>len(coefficients):
            new_coefficients[len(coefficients):] = self.evaluate(coefficients, self.points[len(coefficients):])
        return new_coefficients

    def evaluate(self, coefficients, x):
        """Evaluate a polynomial at a new point.

        This code uses barycentric interpolation, which is quite stable, 
        numerically. The weights are currently not cached between calls, 
        nor does the code use the analytic expression for the weights in
        the case that the points are Chebyshev points.
        """
        if not np.isscalar(x):
            x = np.asarray(x)
            r = np.zeros(x.shape)
            r.flat = [self.evaluate(coefficients, xi) for xi in x.flat]
        # FIXME: might be worth caching the weights
        # FIXME: if we use Chebyshev points, there's an analytic formula
        if len(coefficients)==0:
            return 0.
        w = np.zeros(len(coefficients))
        w[0] = 1
        for j in range(1,len(coefficients)):
            w[:j] *= self.points[:j]-self.points[j]
            w[j] = np.prod(self.points[j]-self.points[:j])
        w = 1./w

        # FIXME: when x is one of the points
        # FIXME: when x is an array
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
    

import numpy as np
import numpy.linalg

class IncompatibleBasesError(ValueError):
    pass

class Polynomial(object):
    """Base class for representing polynomials.

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
        # Catch direct construction
        if not isinstance(self, basis.polynomial_class):
            raise ValueError("Polynomials in basis %s must be instances of %s and should be created by calling basis.polynomial(coefficients)" % (basis, basis.polynomial_class))

        self.basis = basis
        # FIXME: allow vector values and complex dtype
        self.coefficients = np.array(coefficients, dtype=np.float)
        if len(self.coefficients.shape)!=1:
            raise ValueError("Polynomial coefficients must be one-dimensional arrays; given coefficients of shape %s" % (self.coefficients.shape,))

    def iscompatible(self,other):
        return (isinstance(other, self.__class__) and self.basis==other.basis)
    def __eq__(self,other):
        return (self.iscompatible(other)
                and np.all(self.coefficients==other.coefficients))
    def __call__(self, x):
        return self.basis.evaluate(self.coefficients, x)

    def __add__(self, other):
        if isinstance(other,Polynomial):
            if not self.iscompatible(other):
                raise IncompatibleBasesError("Polynomials must be in the same basis to be added")
            return self.basis.polynomial(self.basis.add(self.coefficients,other.coefficients))
        else:
            return self + self.basis.polynomial([other])
    def __radd__(self, other):
        return self+other

    def __mul__(self, other):
        if isinstance(other,Polynomial):
            if not self.iscompatible(other):
                raise IncompatibleBasesError("Polynomials must be in the same basis to be multiplied")
            return self.basis.polynomial(self.basis.multiply(self.coefficients,other.coefficients))
        else:
            c = np.copy(self.coefficients)
            c *= other
            return self.basis.polynomial(c)

    def __rmul__(self, other):
        return self*other

    def __pow__(self, power):
        """Raise to an integer power."""
        return self.basis.polynomial(self.basis.power(self.coefficients, power))

    def __neg__(self):
        return self.basis.polynomial(-self.coefficients)

    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return other + (-self)

    def derivative(self):
        return self.basis.polynomial(self.basis.derivative(self.coefficients))
    def antiderivative(self):
        return self.basis.polynomial(self.basis.antiderivative(self.coefficients))

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
        q, r = self.basis.divide(self.coefficients, other.coefficients)
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

    def companion_matrix(self):
        """Compute the companion matrix of this polynomial.

        The companion matrix is the linear operator representing
        multiplication by X modulo this polynomial. Its minimal
        polynomial is this polynomial, so that its eigenvalues
        with multiplicity are the roots of this polynomial.

        """
        # FIXME: not tested because all subclasses override this method
        return self.basis.companion_matrix(self.coefficients)

    def roots(self):
        """Compute the roots."""
        return self.basis.roots(self.coefficients)

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

class Basis(object):
    """Abstract base class for polynomial bases.

    This class exists primarily to make explicit the interface
    that a polynomial basis must support.

    """

    def __init__(self):
        """Initialize the class.

        Subclasses should set the attribute polynomial_class to the class
        implementing polynomials in this basis.

        """
        self.polynomial_class = None

    def __eq__(self, other):
        """Test for equality."""
        raise NotImplementedError
    def __ne__(self, other):
        return not self==other
    def __hash__(self):
        raise NotImplementedError

    # Polynomial-producing functions
    def polynomial(self, coefficients):
        """Make a polynomial in this basis with the given coefficients.
        
        This is the primary way to construct polynomial objects.
        
        """
        return self.polynomial_class(self, coefficients)

    def zero(self):
        """Return the zero polynomial.
        
        Normally this is represented by an empty list of coefficients.
        
        """
        return self.polynomial([])
    def one(self):
        """Return the polynomial 1."""
        raise NotImplementedError
    def X(self):
        """Return the polynomial X."""
        raise NotImplementedError

    def from_roots(self, roots):
        """Construct a polynomial in this basis given a list of roots."""
        X = self.X()
        r = self.one()
        for rt in roots:
            r *= X-rt
        return r

    def convert(self, polynomial):
        """Convert the given polynomial to this basis."""
        raise NotImplementedError

    # Coefficient-producing functions
    def evaluate(self, coefficients, x):
        """Evaluate the polynomial at x."""
        raise NotImplementedError

    def extend(self, coefficients, n):
        """Extend a coefficient list to length n.

        A polynomial of degree m is naturally represented by a vector
        of coefficients of length m+1. But it is often useful to be able
        to extend the vector of coefficients to length n; for the power
        basis, this just amounts to padding with zeros, but for other
        bases this may be a nontrivial operation.
        """
        raise NotImplementedError

    def add(self, coefficients, other_coefficients):
        """Add two sets of coefficients."""
        l = max(len(coefficients),len(other_coefficients))
        return (self.extend(coefficients, l) 
                + self.extend(other_coefficients, l))
    def subtract(self, coefficients, other_coefficients):
        return self.add(coefficients, -np.array(other_coefficients))
        
    def multiply(self, coefficients, other_coefficients):
        """Multiply two sets of coefficients."""
        raise NotImplementedError

    def power(self, coefficients, power):
        """Raise coefficients to the (nonnegative integer) power."""
        if power != int(power):
            raise ValueError("Can only raise polynomials to integer powers")
        power = int(power)
        if power<0:
            raise ValueError("Cannot raise polynomials to negative powers")

        r = self.one().coefficients
        m = 1
        while 2*m<=power:
            m*=2

        while m:
            r = self.multiply(r, r)
            if m & power:
                r = self.multiply(r, coefficients)
            m//=2

        return r

    def divide(self, coefficients, other_coefficients, tol=None):
        """Polynomial division.

        Given P1 and P2 find Q and R so that P1 = Q*P2 + R and the degree
        of R is strictly less than the degree of P2. 

        In some representations this is an approximate process; for this 
        tol defines the tolerance.
        """
        raise NotImplementedError

    def derivative(self, coefficients):
        raise NotImplementedError
    def antiderivative(self, coefficients):
        raise NotImplementedError

    def companion_matrix(self, coefficients):
        """Compute the companion matrix of this polynomial.

        The companion matrix is the linear operator representing
        multiplication by X modulo this polynomial. Its minimal
        polynomial is this polynomial, so that its eigenvalues
        with multiplicity are the roots of this polynomial.

        This implementation will work for any polynomial type
        that implements division, but it is probably not nearly
        as efficient or numerically stable as a special-purpose
        implementation.

        """
        # FIXME: not tested because all subclasses override this method
        n = len(coefficients)
        M = np.zeros((n,n))
        for i in range(n):
            c = np.zeros(n)
            c[i] = 1
            ignore, r = self.divide((self.polynomial(c)*self.X()), coefficients)
            if len(r)<n:
                r = self.extend(r,n)
            M[:,i] = r
        return M

    def roots(self, coefficients):
        return numpy.linalg.eigvals(self.companion_matrix(coefficients))

class GradedBasis(Basis):
    """A polynomial basis in which the nth element has degree n."""

    def __init__(self):
        Basis.__init__(self)

    def extend(self, coefficients, n):
        """Extend coefficient array to length n.
        
        For GradedBasis and subclasses, this pads with zeros.

        """
        if n<len(coefficients):
            return coefficients[:n]
        z = np.zeros(n)
        z[:len(coefficients)] = coefficients
        return z

    def basis_polynomial(self, n):
        """The nth basis polynomial."""
        # FIXME: vectorize over n?
        c = np.zeros(n+1)
        c[-1] = 1
        return Polynomial(self, c)
    def __getitem__(self, n):
        """Extract the nth basis polynomial."""
        return self.basis_polynomial(n)



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

def declare_functions(module_globals, prefix, basis_class, **extras):
    # This is deeply, deeply wrong.
    import inspect

    names = {'add': basis_class.add,
             'sub': basis_class.subtract,
             #'ext': basis_class.extend,
             #'val': basis_class.evaluate,
             #'int': basis_class.antiderivative,
             #'der': basis_class.derivative,
             #'mul': basis_class.multiply,
             #'div': basis_class.divide,
             }
    for k in extras:
        names[k] = extras[k]

    b_args, b_varargs, b_varkw, b_defaults = inspect.getargspec(basis_class.__init__)
    if b_defaults is None:
        b_defaults = []
    if b_varargs is not None or b_varkw is not None:
        raise ValueError("Cannot cope with a basis whose initializer has varargs.")
    b_needed_args = b_args[:len(b_args)-len(b_defaults)]
    b_defaulted_args = zip(b_args[len(b_args)-len(b_defaults):],b_defaults)

    for (name,function) in names.items():

        args, varargs, varkw, defaults = inspect.getargspec(function)
        if defaults is None:
            defaults = []
        if varargs is not None or varkw is not None:
            raise ValueError("Cannot cope with function %s because it has varargs." % function)

        if args[0]!='self':
            raise ValueError("%s does not appear to be an unbound method" % function)
        args = args[1:]

        needed_args = args[:len(args)-len(defaults)]
        defaulted_args = zip(args[len(args)-len(defaults):],defaults)

        def func(*args,**kwargs):
            
            kwargs = kwargs.copy()
            funcargs = {}
            basis_args = {}

            n=min(len(args),len(needed_args))
            for i in range(n):
                funcargs[needed_args[i]] = args[i]
            args = args[n:]

            n=min(len(args),len(b_needed_args))
            for i in range(n):
                basis_args[b_needed_args[i]] = args[i]
            args = args[n:]

            n=min(len(args),len(defaulted_args))
            for i in range(n):
                funcargs[defaulted_args[i][0]] = args[i]
            args = args[n:]

            n=min(len(args),len(b_defaulted_args))
            for i in range(n):
                basis_args[b_defaulted_args[i][0]] = args[i]
            args = args[n:]

            for (k,v) in defaulted_args:
                if k in kwargs:
                    funcargs[k] = kwargs[k]
                    del kwargs[k]
                elif k not in funcargs:
                    funcargs[k] = v

            for (k,v) in b_defaulted_args:
                if k in kwargs:
                    basis_args[k] = kwargs[k]
                    del kwargs[k]
                elif k not in basis_args:
                    basis_args[k] = v

            if kwargs:
                for k in kwargs:
                    raise ValueError("Keyword argument '%s' not recognized in either %s or %s" % (k, function, basis_class.__init__))

            basis = basis_class(**basis_args)
            print "Calling %s with basis %s and arguments %s" % (function, basis, funcargs)
            return function(basis,**funcargs)

        func.__name__ = prefix+name

        module_globals[func.__name__] = func
        if '__all__' in module_globals:
            module_globals['__all__'].append(func.__name__)

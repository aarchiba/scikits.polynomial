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
        if len(self.coefficients.shape)!=1:
            raise ValueError("Polynomial coefficients must be one-dimensional arrays; given coefficients of shape %s" % self.coefficients.shape)

    def __eq__(self,other):
        return (isinstance(other, Polynomial) 
                and self.basis==other.basis 
                and np.all(self.coefficients==other.coefficients))
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
            # FIXME: is this correct?
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

    def derivative(self):
        return Polynomial(self.basis,self.basis.derivative(self.coefficients))

    def __repr__(self):
        return "<Polynomial basis=%s coefficients=%s>" % (self.basis, self.coefficients)

    def __divmod__(self, other):
        if isinstance(other,Polynomial):
            if self.basis != other.basis:
                raise ValueError("Polynomials must be in the same basis to be added")
            q, r = self.basis.divide(self.coefficients, other.coefficients)
            return Polynomial(self.basis,q), Polynomial(self.basis,r)
        else:
            return (1./other)*self, Polynomial(self.basis,[])

    def __floordiv__(self, other):
        q, r = divmod(self, other)
        return q
    def __mod__(self, other):
        q, r = divmod(self, other)
        return r
    def __truediv__(self, other):
        if isinstance(other, Polynomial):
            raise ValueError("Cannot do true division by polynomials")
        return (1./other)*self
    def __rtruediv__(self,other):
        raise ValueError("Cannot do true division by polynomials")

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

    def derivative(self, coefficients):
        """Compute the coefficients of the derivative."""
        raise NotImplementedError

    def __eq__(self, other):
        """Test for equality."""
        raise NotImplementedError
    def __ne__(self, other):
        return not self==other
    def __hash__(self):
        raise NotImplementedError

    def convert(self, polynomial):
        """Convert the given polynomial to this basis."""
        raise NotImplementedError

    def divide(self, coefficients, other_coefficients, tol=None):
        """Polynomial division.

        Given P1 and P2 find Q and R so that P1 = Q*P2 + R and the degree
        of R is strictly less than the degree of P2. 

        In some representations this is an approximate process; for this 
        tol defines the tolerance.
        """
        raise NotImplementedError


class GradedBasis(Basis):
    """A polynomial basis in which the nth element has degree n."""

    def __init__(self):
        Basis.__init__(self)

    def extend(self, coefficients, n):
        if n<len(coefficients):
            raise ValueError("extend can only make coefficient arrays longer")
        z = np.zeros(n)
        z[:len(coefficients)] = coefficients
        return z



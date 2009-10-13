import numpy as np

from poly import Polynomial, Basis, GradedBasis, IncompatibleBasesError

class PowerPolynomial(Polynomial):
    """Polynomial in the power basis.

    These polynomials are represented as linear combinations of
    1, (x-c), (x-c)**2, ...

    """

    def __call__(self, x):
        x = np.asarray(x,dtype=float) - self.basis.center
        r = np.zeros(x.shape)
        if len(self.coefficients)==0:
            return r
        r += self.coefficients[-1]
        for c in self.coefficients[-2::-1]:
            r *= x
            r += c
        return r

    def _multiply_polynomial(self, other):
        # FIXME: use FFT-based fast convolution where appropriate
        # (actually probably never; degrees high enough for it to
        # be faster are almost certainly numerically useless).
        return self.basis.polynomial(
                np.convolve(self.coefficients, 
                            other.coefficients, 
                            mode='full'))
        
    def derivative(self):
        if len(self.coefficients)==0:
            return self.basis.zero()
        return self.basis.polynomial(self.coefficients[1:]*
                np.arange(1,len(self.coefficients)))
    def antiderivative(self):
        if len(self.coefficients)==0:
            return self.basis.zero()
        c = np.zeros(len(self.coefficients)+1)
        c[1:] = self.coefficients/np.arange(1,len(self.coefficients)+1)
        return self.basis.polynomial(c)

    def divide(self, other, tol=None):
        """Grade-school long division.
        
        Note that this algorithm assumes that leading zero coefficients are 
        exactly zero, which may not be the case if the polynomials
        were obtained by calculation on polynomials of higher order.
        """
        if not self.iscompatible(other):
            raise IncompatibleBasesError("Cannot divide polynomials in different bases")
        if len(other.coefficients)==0:
            raise ValueError("Polynomial division by zero")

        other_coefficients = other.coefficients
        while other_coefficients[-1]==0 and len(other_coefficients)>0:
            other_coefficients = other_coefficients[:-1]

        if len(other_coefficients)==0:
            raise ValueError("Polynomial division by zero")
        if len(other_coefficients)==1:
            return self/other_coefficients[0], self.basis.zero()
        if len(self.coefficients)<len(other_coefficients):
            return self.basis.zero(), self

        c = self.coefficients.copy()
        q = np.zeros(len(self.coefficients)-len(other_coefficients)+1)

        for i in range(len(q)):
            s = c[-1-i]/other_coefficients[-1]
            c[len(c)-i-len(other_coefficients):len(c)-i]-=s*other_coefficients
            q[-1-i] = s
        return (self.basis.polynomial(q), 
                self.basis.polynomial(c[:len(other_coefficients)-1]))




class PowerBasis(GradedBasis):
    """The basis 1, (x-c), (x-c)**2, ..."""

    def __init__(self,center=0):
        Basis.__init__(self)
        self.center = center
        self.polynomial_class = PowerPolynomial

    def one(self):
        return self.polynomial([1])
    def X(self):
        return self.polynomial([0,1])

    def __eq__(self, other):
        return isinstance(other, PowerBasis) and self.center == other.center
    def __hash__(self):
        return hash(self.center) # Include self's type?

    def convert(self, polynomial):
        n = len(polynomial.coefficients)
        if n==0:
            return self.zero()
        rc = np.zeros(n)
        fact = 1
        for i in range(n):
            rc[i]=polynomial(self.center)/fact
            polynomial = polynomial.derivative()
            fact*=i+1
        return self.polynomial(rc)

    def __repr__(self):
        return "<PowerBasis center=%g>" % self.center


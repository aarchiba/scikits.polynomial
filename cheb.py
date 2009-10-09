import numpy as np

from poly import Polynomial, Basis, GradedBasis
import chebyshev

class ChebyshevBasis(GradedBasis):

    def __init__(self, interval=(-1,1)):
        a, b = interval # type checking
        self.interval = (float(a),float(b))

    def evaluate(self, coefficients, x):
        if len(coefficients)==0:
            return 0.
        return chebyshev.chebval(x, coefficients, domain=self.interval)

    def multiply(self, coefficients, other_coefficients):
        if len(coefficients)==0 or len(other_coefficients)==0:
            return np.zeros(0)
        return chebyshev.chebmul(coefficients, other_coefficients)

    def derivative(self, coefficients):
        if len(coefficients)==0:
            return np.zeros(0)
        return chebyshev.chebder(coefficients)
    def antiderivative(self, coefficients):
        if len(coefficients)==0:
            return np.zeros(0)
        return chebyshev.chebint(coefficients)

    def __eq__(self, other):
        return isinstance(other, ChebyshevBasis) and self.interval==other.interval
    def __hash__(self):
        return hash(self.interval)

    def convert(self, polynomial):
        n = len(polynomial.coefficients)
        if n==0:
            return Polynomial(self, [])

        a, b = self.interval
        xk = np.cos(np.pi*(np.arange(n)+0.5)/n)
        fxk = polynomial(((b-a)/2.) * xk + (b+a)/2.)

        c = np.zeros(n)
        for j in range(n):
            # FIXME: use DCT to evaluate this
            c[j] = np.sum(fxk*np.cos(np.pi*j*(np.arange(n)+0.5)/n))

        c[0]/=2.
        return Polynomial(self,(2./n)*c)

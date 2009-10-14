import numpy as np

from poly import Polynomial, Basis, GradedBasis, IncompatibleBasesError


class PowerBasis(GradedBasis):
    """The basis 1, (x-c), (x-c)**2, ..."""

    def __init__(self,center=0):
        Basis.__init__(self)
        self.center = center
        self.polynomial_class = Polynomial

    def one(self):
        return self.polynomial([1])
    def X(self):
        return self.polynomial([0,1])

    def __eq__(self, other):
        return isinstance(other, PowerBasis) and self.center == other.center
    def __hash__(self):
        return hash(self.center) # Include self's type?

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
        return np.convolve(coefficients, other_coefficients, mode='full')

    def derivative(self, coefficients):
        if len(coefficients)==0:
            return np.zeros(0)
        return coefficients[1:]*np.arange(1,len(coefficients))

    def antiderivative(self,coefficients):
        if len(coefficients)==0:
            return np.zeros(0)
        c = np.zeros(len(coefficients)+1)
        c[1:] = coefficients/np.arange(1,len(coefficients)+1)
        return c

    def divide(self, coefficients, other_coefficients, tol=None):
        """Grade-school long division.
        
        Note that this algorithm assumes that leading zero coefficients are 
        exactly zero, which may not be the case if the polynomials
        were obtained by calculation on polynomials of higher order.
        """
        if len(other_coefficients)==0:
            raise ValueError("Polynomial division by zero")
        
        if tol is None:
            tol = 0
        while (np.abs(other_coefficients[-1])<=tol and 
                len(other_coefficients)>0):
            other_coefficients = other_coefficients[:-1]

        if len(other_coefficients)==0:
            raise ValueError("Polynomial division by zero")
        if len(other_coefficients)==1:
            return coefficients/other_coefficients[0], np.zeros(0)
        if len(coefficients)<len(other_coefficients):
            return np.zeros(0), coefficients

        c = coefficients.copy()
        q = np.zeros(len(coefficients)-len(other_coefficients)+1)

        for i in range(len(q)):
            s = c[-1-i]/other_coefficients[-1]
            c[len(c)-i-len(other_coefficients):len(c)-i]-=s*other_coefficients
            q[-1-i] = s
        return q, c[:len(other_coefficients)-1]

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

    def companion_matrix(self,coefficients):
        n = len(coefficients)-1
        if n<1:
            return np.zeros((0,0))
        M = np.zeros((n,n))
        M[:,-1] = -coefficients[:-1]/coefficients[-1]
        M += np.diag(np.ones(n-1),-1)
        return M

    def __repr__(self):
        return "<PowerBasis center=%g>" % self.center


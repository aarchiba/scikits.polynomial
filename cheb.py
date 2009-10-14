import numpy as np

try:
    from scipy.fftpack import dct
except ImportError:
    dct = None
    
from poly import Polynomial, Basis, GradedBasis
import chebyshev

class ChebyshevBasis(GradedBasis):

    def __init__(self, interval=(-1,1)):
        GradedBasis.__init__(self)
        self.polynomial_class = Polynomial
        a, b = interval # type checking
        self.interval = (float(a),float(b))

    def evaluate(self, coefficients, x):
        if len(coefficients)==0:
            # FIXME: same shape as x.
            return 0.
        return chebyshev.chebval(x, coefficients, 
                domain=self.interval)

    def one(self):
        return self.polynomial([1])
    def X(self):
        a, b = self.interval
        return self.polynomial([(b+a)/2.,(b-a)/2.])


    def __eq__(self, other):
        return isinstance(other, ChebyshevBasis) and self.interval==other.interval
    def __hash__(self):
        return hash(self.interval)

    def convert(self, polynomial):
        n = len(polynomial.coefficients)
        if n==0:
            return self.zero()

        a, b = self.interval
        xk = np.cos(np.pi*(np.arange(n)+0.5)/n)
        fxk = polynomial(((b-a)/2.) * xk + (b+a)/2.)

        c = dct(fxk, type=2)
        c /= 2.
        c[0] /= 2.

        return self.polynomial((2./n)*c)

    def multiply(self, coefficients, other_coefficients):
        if len(coefficients)==0 or len(other_coefficients)==0:
            return np.zeros(0)
        return chebyshev.chebmul(coefficients, other_coefficients)

    def divide(self, coefficients, other_coefficients):
        if len(other_coefficients)==0:
            raise ValueError("Polynomial division by zero")
        
        q, r = chebyshev.chebdiv(coefficients, other_coefficients)
        return q, r

    def roots(self,coefficients):
        if len(coefficients)<2:
            return np.zeros(0)
        a, b = self.interval
        return chebyshev.chebroots(coefficients)*((b-a)/2.)+(a+b)/2.

    def derivative(self, coefficients):
        if len(coefficients)==0:
            return np.zeros(0)
        return chebyshev.chebder(coefficients)
    def antiderivative(self, coefficients):
        if len(coefficients)==0:
            return np.zeros(0)
        return chebyshev.chebint(coefficients)


def _dct(x, type=2, axis=-1):
    """
    Fallback implementation of DCT, in case scipy.fftpack is not available.

    """
    if type != 2:
        raise NotImplementedError()
    x = np.atleast_1d(x)

    if x.ndim > 1:
        tp = range(x.ndim)
        if axis is not None:
            tmp = tp[0]
            tp[0] = axis
            tp[axis] = tmp
        x = x.transpose(tp)

    n = x.shape[0]
    r = np.zeros((4*n,) + x.shape[1:], x.dtype)
    r[1:2*n:2] = x
    x = np.fft.rfft(r, axis=0)
    x = x[:n].real
    x *= 2.

    if x.ndim > 1:
        x = x.transpose(tp)

    return x

if dct is None:
    dct = _dct

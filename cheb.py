import numpy as np

try:
    from scipy.fftpack import dct
except ImportError:
    dct = None
    
from poly import Polynomial, Basis, GradedBasis, generate_basis
import chebyshev


def convert_function(self, polynomial):
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

ChebyshevBasis = generate_basis("ChebyshevBasis",(-1,1),
        X=[0,1], 
        mul=chebyshev.chebmul, 
        val=chebyshev.chebval, 
        der=chebyshev.chebder,
        int=chebyshev.chebint, 
        div=chebyshev.chebdiv, 
        roots=chebyshev.chebroots,
        convert_function=convert_function)

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

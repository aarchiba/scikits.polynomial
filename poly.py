

class Poly:

    def __init__(self,coefficients,basis):
        self.coefficients = coefficients
        self.basis = basis

    def change_basis(self,new_basis):
        pass


class PolynomialBasis:

    def __init__(self):
        pass

    def evaluate(self, n, x):
        raise NotImplementedError

    def evaluate_series(self, coefficients, x):
        r = np.zeros(np.shape(x))
        for (n,c) in enumerate(coefficients):
            r += c*self.evaluate(n,x)
        return r

    def derivative(self, n, der=1):
        if der==0:
            r = np.zeros((n+1,))
            r[-1] = 1
            return r
        elif der==1:
            raise NotImplementedError
        elif der>1:
            return derivative_series(self.derivative(n,der=1),der=der-1)
        else:
            raise ValueError("derivative method requires a nonnegative integer order (got %g)" % der)

    def derivative_series(self, coefficients, der=1):
        if der==0:
            return coefficients
        elif der==1:
            r = None
            for n,c in reverse(enumerate(coefficients)):
                v = c*self.derivative(n)
                if r is None:
                    r = v
                else:
                    r[:n] += v
            return r
        elif der>1 and der==int(der):
            for i in range(der):
                coefficients = self.derivative_series(coefficients, der=1)
        else:
            raise ValueError("derivative_series method requires a nonnegative integer order (got %g)" % der)

    def convert(self, coefficients, current_basis):
        raise NotImplementedError

class PowerBasis:
    def __init__(self, center=0.):
        self.center = center

    def evaluate(self, n, x):
        return (x-self.center)**n

    def derivative(self, n, der=1):
        if der==1:
            r = np.zeros(n-1)
            r[-1] = n
            return r
        else:
            return PowerBasis.derivative(self, n, der)

    def convert(self, coefficients, current_basis):
        r = np.zeros(len(coefficients))
        fac = 1
        for i in range(len(coefficients)):
            r[i] = current_basis.evaluate_series(coefficients,self.center)/fac
            coefficients = current_basis.derivative_series(coefficients)
            fac *= i+1
        return r

def OrthogonalPolynomialBasis(PolynomialBasis):
    
    def __init__(self, an_func, sqrt_bn_func, mu, limits=(-np.inf,np.inf)):
        PolynomialBasis.__init__(self)
        self.an_func = an_func
        self.sqrt_bn_func = sqrt_bn_func
        self.mu = mu

    def roots(self, n):
        x, w = scipy.special.orthogonal.gen_roots_and_weights(n, self.an_func, self.sqrt_bn_func, self.mu)
        return x

    def weights(self, n):
        x, w = scipy.special.orthogonal.gen_roots_and_weights(n, self.an_func, self.sqrt_bn_func, self.mu)
        return w

    def convert(self, coefficients, current_basis):
        pass

def orthogonal_fit(orthogonal_basis, n, func):
    """Find the polynomial of degree n that best fits func

    This fit is computed rapidly using the orthogonality relations in
    the given polynomial basis.
    """
    pass

def ChebyshevBasis(OrthogonalPolynomialBasis):

    def __init__(self):
        OrthogonalPolynomialBasis.__init__(self)

    def evaluate(self, n, x):
        return np.cos(n*np.arccos(x))

    

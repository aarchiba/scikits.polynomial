import numpy as np

class PowerBasisPolynomial:

    def __init__(self, coefficients):
        self.coefficients = np.array(coefficients)

    def __call__(self, x):
        r = np.zeros_like(x)
        if len(self.coefficients)==0:
            return r
        r += self.coefficients[-1]
        for c in self.coefficients[-2::-1]:
            r *= x
            r += c
        return r

    def __add__(self, other):
        if isinstance(other,PowerBasisPolynomial):
            l = max(len(self.coefficients),len(other.coefficients))
            c = np.zeros(l)
            c[:len(self.coefficients)] = self.coefficients
            c[:len(other.coefficients)] += other.coefficients
            return PowerBasisPolynomial(c)
        else:
            c = np.copy(self.coefficients)
            c[0] += other
            return PowerBasisPolynomial(c)

    def __mul__(self, other):
        if isinstance(other,PowerBasisPolynomial):
            c = np.zeros(len(self.coefficients)+len(other.coefficients))
            for i, ci in enumerate(self.coefficients):
                c[i:i+len(other.coefficients)] += ci*other.coefficients
        else:
            c = np.copy(self.coefficients)
            c *= other
            return PowerBasisPolynomial(c)
    def __rmul__(self, other):
        return self*other
    def __sub__(self, other):
        return self + (-1)*other





from poly import polynomial, Polynomial
from power import PowerBasis

class PolynomialPlus(Polynomial):

    def foo(self):
        print "foo"

class ExtraMethod(PowerBasis):

    def polynomial(self, coefficients):
        return PolynomialPlus(self, coefficients)

if __name__=='__main__':
    p = polynomial(ExtraMethod(), [1,2,3])
    p.foo()

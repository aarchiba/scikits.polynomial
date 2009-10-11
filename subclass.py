from poly import Polynomial
from power import PowerBasis

class PolynomialPlus(Polynomial):

    def foo(self):
        print "foo"

class ExtraMethod(PowerBasis):

    def __init__(self):
        PowerBasis.__init__(self)

    def _create(self, obj, coefficients):
        obj = PowerBasis._create(self, obj, coefficients)
        obj.__class__ = PolynomialPlus # urk! is this really right?
        return obj


if __name__=='__main__':
    p = Polynomial(ExtraMethod(), [0,0,1])
    p.foo()


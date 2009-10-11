import numpy as np
import pylab as plt

from poly import Polynomial
from power import PowerBasis
from cheb import ChebyshevBasis
from lagrange import LagrangeBasis

def demo_perfidious(n):
    plt.figure()

    r = (np.arange(n)+1)/float(n+1)

    bases = [(PowerBasis(), "Power"),
             (ChebyshevBasis(interval=(1./(n+1),n/float(n+1))),"Chebyshev"), 
             (LagrangeBasis(interval=(1./(n+1),n/float(n+1))),"Lagrange"), 
             (LagrangeBasis(r),"Specialized Lagrange")]

    xs = np.linspace(0,1,50*n)
    
    for (i,(b,l)) in enumerate(bases):
        p = b.from_roots(r)
        plt.subplot(len(bases),1,i+1)
        plt.semilogy(xs,np.abs(p(xs)),label=l)
        plt.xlim(0,1)
        plt.ylim(min=1)
        
        for j in range(n):
            plt.axvline((j+1)/float(n+1),linestyle=":",color="black")
        plt.legend(loc="best")
    print b.points
    print p.coefficients
    plt.subplot(len(bases),1,1)
    plt.title('The "perfidious polynomial" for n=%d' % n)


if __name__=='__main__':
    demo_perfidious(35)
    demo_perfidious(40)
    demo_perfidious(100) # floating-point overflow
    plt.show()

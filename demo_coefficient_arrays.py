import lagrange

lb = lagrange.LagrangeBasis(interval=(0,1))

zero = lb.zero().coefficients
one = lb.one().coefficients
X = lb.X().coefficients

p = lb.multiply(X-0.5,X-0.5)

print lb.evaluate(p,[0,0.5,1])


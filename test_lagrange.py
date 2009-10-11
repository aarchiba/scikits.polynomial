import numpy as np

from poly import Polynomial
from lagrange import LagrangeBasis, bit_reverse, chebyshev_points_sequence

def test_bit_reverse():
    n = 64
    rs = set()
    for i in range(n):
        b = bit_reverse(i,n)
        rs.add(b)

    for i in range(n):
        assert i in rs



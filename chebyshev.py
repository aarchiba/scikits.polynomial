"""Functions for dealing with Chebyshev series.

This module provide s a number of functions that are useful in dealing with
Chebyshev series as well as a `Chebyshev` class that encapsuletes the usual
arithmetic operations. All the chebyshev series are assumed to be ordered
from low to high, thus `array([1,2,3])` will be treated as the series
`T_0 + 2*T_1 + 3*T_2`

Arithmetic
----------
- chebadd -- add a Chebyshev series to another.
- chebsub -- subtract a Chebyshev series from another.
- chebmul -- multiply a Chebyshev series by another
- chebdiv -- divide one Chebyshev series by another.
- chebval -- evaluate a Chebyshev series at given points.

Calculus
--------
- chebder -- differentiate a Chebyshev series.
- chebint -- integrate a Chebyshev series.

Misc Functions
--------------
- chebfromroots -- create a Chebyshev series with specified roots.
- chebroots -- find the roots of a Chebyshev series.
- cheb2poly -- convert a Chebyshev series to a polynomia.
- poly2cheb -- convert a polynomial to a Chebyshev series.
- chebfit -- least squares fit returning a Chebyshev series.
- chebtrim -- trim leading coefficients from a Chebyshev series.

Classes
-------
- Chebyshev -- Chebyshev series class.

Notes
-----
The implementations of multiplication, division, integration, and
differentiation use the algebraic identities:

.. math ::
    T_n(x) = \\frac{z^n + z^{-n}}{2} \\\\
    z\\frac{dx}{dz} = \\frac{z - z^{-1}}{2}.

where

.. math :: x = \\frac{z + z^{-1}}{2}.

These identities allow a Chebyshev series to be expressed as a finite,
symmetric Laurent series. These sorts of Laurent series are referred to as
z-series in this module.

"""
from __future__ import division

__all__ = ['chebadd', 'chebsub', 'chebmul', 'chebdiv', 'chebder', 'chebint',
           'chebtrim', 'cheb2poly', 'poly2cheb', 'chebfromroots', 'chebfit',
           'chebroots', 'Chebyshev', 'RankWarning', 'ChebyshevError',
           'ChebyshevDomainError']

import warnings, exceptions
import numpy.core as npc
import numpy.lib as npl
import numpy.linalg as la

#
# Warnings and Exceptions
#

class RankWarning(UserWarning) :
    """Issued by chebfit when the design matrix is rank deficient."""
    pass

class ChebyshevError(Exception) :
    """Base class for errors in this module."""
    pass

class ChebyshevDomainError(ChebyshevError) :
    """Issued by the Chebshev class when two domains don't match.

    This is raised when an binary operation is passed Cheb objects with
    different domains.

    """
    pass

#
# Helper functions to convert inputs to 1d arrays
#
def _trim(seq) :
    """Remove small Chebyshev series coefficients.

    Parameters
    ----------
    seq : sequence
        Sequence of Chebyshev series coefficients. This routine fails for
        empty sequences.

    Returns
    -------
    series : sequence
        subsequence with trailing zeros removed. If the resulting sequence
        would be empty, return the first element. The returned sequence may
        or may not be a view.

    Notes
    -----
    Do not lose the type info if the sequence contains unknown objects.

    """
    for i in range(len(seq) - 1, -1, -1) :
        if seq[i] != 0 :
            break
    return seq[:i+1]

def _as_series(alist, trim=True) :
    """Return arguments as a list of 1d arrays.

    The return type will always be an array of double, complex double. or
    object.

    Parameters
    ----------
    [a1, a2,...] : list of array_like.
        The arrays must have no more than one dimension when converted.
    trim : boolean
        When True, trailing zeros are removed from the inputs.
        When False, the inputs are passed through intact.

    Returns
    -------
    [a1, a2,...] : list of 1d-arrays
        A copy of the input data as a 1d-arrays.

    Raises
    ------
    ValueError :
        Raised when an input can not be coverted to 1-d array or the
        resulting array is empty.

    """
    arrays = [npc.array(a, ndmin=1, copy=0) for a in alist]
    if min([a.size for a in arrays]) == 0 :
        raise ValueError("Coefficient array is empty")
    if max([a.ndim for a in arrays]) > 1 :
        raise ValueError("Coefficient array is not 1-d")
    if trim :
        arrays = [_trim(a) for a in arrays]

    if any([a.dtype == npc.dtype(object) for a in arrays]) :
        ret = []
        for a in arrays :
            if a.dtype != npc.dtype(object) :
                tmp = npc.empty(len(a), dtype=npc.dtype(object))
                tmp[:] = a[:]
                ret.append(tmp)
            else :
                ret.append(a.copy())
    else :
        try :
            dtype = npl.common_type(*arrays)
        except :
            raise ValueError("Coefficient arrays have no common type")
        ret = [npc.array(a, copy=1, dtype=dtype) for a in arrays]
    return ret

#
# A collection of functions for manipulating z-series. These are private
# functions and do minimal error checking.
#


def _cseries_to_zseries(cs) :
    """Covert Chebyshev series to z-series.

    Covert a Chebyshev series to the equivalent z-series. The result is
    never an empty array. The dtype of the return is the same as that of
    the input. No checks are run on the arguments as this routine is for
    internal use.

    Parameters
    ----------
    cs : 1-d ndarray
        Chebyshev coefficients, ordered from low to high

    Returns
    -------
    zs : 1-d ndarray
        Odd length symmetric z-series, ordered from  low to high.

    """
    n = cs.size
    zs = npc.zeros(2*n-1, dtype=cs.dtype)
    zs[n-1:] = cs/2
    return zs + zs[::-1]


def _zseries_to_cseries(zs) :
    """Covert z-series to a Chebyshev series.

    Covert a z series to the equivalent Chebyshev series. The result is
    never an empty array. The dtype of the return is the same as that of
    the input. No checks are run on the arguments as this routine is for
    internal use.

    Parameters
    ----------
    zs : 1-d ndarray
        Odd length symmetric z-series, ordered from  low to high.

    Returns
    -------
    cs : 1-d ndarray
        Chebyshev coefficients, ordered from  low to high.

    """
    n = (zs.size + 1)//2
    cs = zs[n-1:].copy()
    cs[1:n] *= 2
    return cs


def _zseries_mul(z1, z2) :
    """Multiply two z-series.

    Multiply two z-series to produce a z-series.

    Parameters
    ----------
    z1, z2 : 1-d ndarray
        The arrays must be 1-d but this is not checked.

    Returns
    -------
    product : 1-d ndarray
        The product z-series.

    Notes
    -----
    This is simply convolution. If symmetic/anti-symmetric z-series are
    denoted by S/A then the following rules apply:

    S*S, A*A -> S
    S*A, A*S -> A

    """
    return npc.convolve(z1, z2)


def _zseries_div(z1, z2) :
    """Divide the first z-series by the second.

    Divide `z1` by `z2` and return the quotient and remainder as z-series.
    Warning: this implementation only applies when both z1 and z2 have the
    same symmetry, which is sufficient for present purposes.

    Parameters
    ----------
    z1, z2 : 1-d ndarray
        The arrays must be 1-d and have the same symmetry, but this is not
        checked.

    Returns
    -------

    (quotient, remainder) : 1-d ndarrays
        Quotient and remainder as z-series.

    Notes
    -----
    This is not the same as polynomial division on account of the desired form
    of the remainder. If symmetic/anti-symmetric z-series are denoted by S/A
    then the following rules apply:

    S/S -> S,S
    A/A -> S,A

    The restriction to types of the same symmetry could be fixed but seems like
    uneeded generality. There is no natural form for the remainder in the case
    where there is no symmetry.

    """
    z1 = z1.copy()
    z2 = z2.copy()
    len1 = len(z1)
    len2 = len(z2)
    if len2 == 1 :
        z1 /= z2
        return z1, z1[:1]*0
    elif len1 < len2 :
        return z1[:1]*0, z1
    else :
        dlen = len1 - len2
        scl = z2[0]
        z2 /= scl
        quo = npc.zeros(dlen + 1, dtype=z1.dtype)
        i = 0
        j = dlen
        while i < j :
            r = z1[i]
            quo[i] = z1[i]
            quo[dlen - i] = r
            tmp = r*z2
            z1[i:i+len2] -= tmp
            z1[j:j+len2] -= tmp
            i += 1
            j -= 1
        r = z1[i]
        quo[i] = r
        tmp = r*z2
        z1[i:i+len2] -= tmp
        quo /= scl
        rem = z1[i+1:i-1+len2].copy()
        return quo, rem


def _zseries_der(zs) :
    """Differentiate a z-series.

    The derivative is with respect to x, not z. This is achieved using the
    chain rule and the value of dx/dz given in the module notes.

    Parameters
    ----------
    zs : z-series
        The z-series to differentiate.

    Returns
    -------
    derivative : z-series
        The derivative

    """
    n = len(zs)//2
    ns = npc.array([-.5, 0, .5], dtype=zs.dtype)
    zs *= npc.arange(-n, n+1)
    d, r = _zseries_div(zs, ns)
    return d


def _zseries_int(zs) :
    """Integrate a z-series.

    The integral is with respect to x, not z. This is achieved by a change
    of variable using dx/dz given in the module notes.

    Parameters
    ----------
    zs : z-series
        The z-series to integrate

    Returns
    -------
    integral : z-series
        The integral

    """
    n = 1 + len(zs)//2
    ns = npc.array([-.5, 0, .5], dtype=zs.dtype)
    zs = _zseries_mul(zs, ns)
    zs /= npc.arange(-n, n+1)
    zs[n] = 2*(zs[n+2::4].sum() - zs[n+4::4].sum());
    return zs

#
# Chebyshev series functions
#


def chebtrim(c, tol=0) :
    """Remove small leading coefficients from a Chebyshev series.

    Parameters
    ----------
    c : array_like
        1-d array. If Chebyshev series, ordered from  low to high.
    tol : number
        Leading elements with absolute value less than tol are removed.

    Returns
    -------
    trimmed : ndarray
        1_d array with leading zeros removed. If the cleaned series
        is all zeros a series containing a singel zero is returned.

    Raises
    ------
    ValueError : if tol < 0

    """
    if tol < 0 :
        raise ValueError("tol must be non-negative")

    [c] = _as_series([c])
    [ind] = npc.where(npc.abs(c) > tol)
    if len(ind) == 0 :
        return c[:1]*0
    else :
        return c[:ind[-1]+1].copy()


def poly2cheb(pol) :
    """Convert a polynomial to a Chebyshev series.

    Convert a series containing polynomial coefficients ordered by degree
    from low to high to an equivalent Chebyshev series ordered from low to
    high.

    Inputs
    ------
    pol : array_like
        1-d array containing the polynomial coeffients

    Returns
    -------
    cseries : ndarray
        1-d array containing the coefficients of the equivalent Chebyshev
        series.

    See Also
    --------
    chebadd, chebsub, chebmul, chebdiv, chebder, chebint, chebval, chebfit

    """
    [pol] = _as_series([pol])
    pol = pol[::-1]
    zs = pol[:1].copy()
    x = npc.array([.5, 0, .5], dtype=pol.dtype)
    for i in range(1, len(pol)) :
        zs = _zseries_mul(zs, x)
        zs[i] += pol[i]
    return _zseries_to_cseries(zs)


def cheb2poly(cs) :
    """Convert a Chebyshev series to a polynomial.

    Covert a series containing Chebyshev series coefficients orderd from
    low to high to an equivalent polynomial ordered from low to
    high by degree.

    Inputs
    ------
    cs : array_like
        1-d array containing the Chebyshev series coeffients ordered from
        low to high.

    Returns
    -------
    pol : ndarray
        1-d array containing the coefficients of the equivalent polynomial
        ordered from low to high by degree.

    See Also
    --------
    chebadd, chebsub, chebmul, chebdiv, chebder, chebint, chebval, chebfit

    """
    [cs] = _as_series([cs])
    pol = npc.zeros(len(cs), dtype=cs.dtype)
    quo = _cseries_to_zseries(cs)
    x = npc.array([.5, 0, .5], dtype=pol.dtype)
    for i in range(0, len(cs) - 1) :
        quo, rem = _zseries_div(quo, x)
        pol[i] = rem[0]
    pol[-1] = quo[0]
    return pol


def chebfromroots(roots) :
    """Generate a Chebyschev series with given roots.

    Generate a Chebyshev series whose roots are given by `roots`. The
    resulting series is the produet `(x - roots[0])*(x - roots[1])*...`

    Inputs
    ------
    roots : array_like
        1-d array containing the roots in sorted order.

    Returns
    -------
    series : ndarray
        1-d array containing the coefficients of the Chebeshev series
        ordered from low to high.

    See Also
    --------
    chebadd, chebsub, chebmul, chebdiv, chebder, chebint, chebval, chebfit

    """
    if len(roots) == 0 :
        return npc.ones(1)
    else :
        [roots] = _as_series([roots])
        if roots.size == 0 :
            raise ValueError("No roots were specified.")
        dt = roots.dtype
        prd = npc.array([1], dtype=dt)
        for r in roots :
            fac = npc.array([.5, -r, .5], dtype=roots.dtype)
            prd = _zseries_mul(fac, prd)
        return _zseries_to_cseries(prd)


def chebadd(c1, c2):
    """Add one Chebyshev series to another.

    Returns the sum of two Chebyshev series `c1` + `c2`. The arguments are
    sequences of coeffients ordered from low to high, i.e., [1,2,3] is the
    series :math: "T_0 + 2*T_1 + 3*T_2".

    Parameters
    ----------
    c1, c2 : array_like
        1d arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Chebyshev series of the sum.

    See Also
    --------
    chebsub, chebmul, chebdiv, chebder, chebint, chebval, chebfit

    """
    [c1, c2] = _as_series([c1, c2])
    if len(c1) > len(c2) :
        c1[:c2.size] += c2
        ret = c1
    else :
        c2[:c1.size] += c1
        ret = c2
    return _trim(ret)


def chebsub(c1, c2):
    """Subtract one Chebyshev series from another.

    Returns the difference of two Chebyshev series `c1` - `c2`. The
    arguments are sequences of coeffients ordered from low to high, i.e.,
    [1,2,3] is the series :math: `T_0 + 2*T_1 + 3*T_2.`

    Parameters
    ----------
    c1, c2 : array_like
        1d arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Chebyshev series of the subtraction.

    See Also
    --------
    chebadd, chebmul, chebdiv, chebder, chebint, chebval, chebfit

    Examples
    --------

    """
    [c1, c2] = _as_series([c1, c2])
    if len(c1) > len(c2) :
        c1[:c2.size] -= c2
        ret = c1
    else :
        c2[:c1.size] -= c1
        ret = -c2
    return _trim(ret)


def chebmul(c1, c2):
    """Multiply one Chebyshev series by another.

    Returns the product of two Chebyshev series `c1` * `c2`. The arguments
    are sequences of coeffients ordered from low to high, i.e., [1,2,3] is
    the series :math: T_2 + 2*T_1 + 3*T_3.

    Parameters
    ----------
    c1, c2 : array_like
        1d arrays of chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Chebyshev series of the product.

    See Also
    --------
    chebadd, chebsub, chebdiv, chebder, chebint, chebval, chebfit

    """
    [c1, c2] = _as_series([c1, c2])
    z1 = _cseries_to_zseries(c1)
    z2 = _cseries_to_zseries(c2)
    prd = _zseries_mul(z1, z2)
    ret = _zseries_to_cseries(prd)
    return _trim(ret)


def chebdiv(c1, c2):
    """Divide one Chebyshev series by another.

    Returns the quotient of two Chebyshev series `c1` / `c2`. The arguments
    are sequences of coeffients ordered from low to high, i.e., [1,2,3] is
    the series :math: T_2 + 2*T_1 + 3*T_3.

    Parameters
    ----------
    c1, c2 : array_like
        1d arrays of chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    [quo, rem] : ndarray
        Chebyshev series of the quotient and remainder.

    See Also
    --------
    chebadd, chebsub, chebmul, chebder, chebint, chebval, chebfit

    Examples
    --------

    """
    [c1, c2] = _as_series([c1, c2])
    if len(c1) < len(c2) :
        quo, rem = c1[0:1]*0, c1
    else :
        z1 = _cseries_to_zseries(c1)
        z2 = _cseries_to_zseries(c2)
        quo, rem = _zseries_div(z1, z2)
        quo = _zseries_to_cseries(quo)
        rem = _trim(_zseries_to_cseries(rem))
    return _trim(quo), _trim(rem)


def chebder(cs, m=1) :
    """Differentiate a Chebyshev series.

    Parameters
    ----------
    cs: array_like
        1d array of chebyshev series coefficients ordered from low to high.
    m : int, optional
        Order of differentiation, must be non-negative. (default: 1)

    Returns
    -------
    der : ndarray
        Chebyshev series of the derivative.

    See Also
    --------
    chebadd, chebsub, chebmul, chebdiv, chebint, chebval, chebfit

    Examples
    --------

    """
    [cs] = _as_series([cs])
    if m < 0 :
        raise ValueError, "The order of derivation must be positive"
    if m == 0 :
        ret = cs
    else :
        zs = _cseries_to_zseries(cs)
        for i in range(m) :
            zs = _zseries_der(zs)
        ret = _zseries_to_cseries(zs)
    return ret


def chebint(cs, m=1, k=[]) :
    """Integrate a Chebyshev series.

    Parameters
    ----------
    cs: array_like
        1d array of chebyshev series coefficients ordered from low to high.
    m : int, optional
        Order of integration, must be positeve. (default: 1)
    k : {[], list, scalar}, optional
        Integration constants. The value of the first integral at zero is
        the first value in the list, the value of the second integral at
        zero is the second value in the list, and so on.

        If ``[]`` (default), all constants are set zero.
        If `m = 1`, a single scalar can be given instead of a list.

    Returns
    -------
    der : ndarray
        Chebyshev series of the derivative.

    Raises
    ------
    ValueError

    See Also
    --------
    chebadd, chebsub, chebmul, chebdiv, chebder, chebval, chebfit

    Examples
    --------

    """
    if npc.isscalar(k) :
        k = [k]
    if m < 1 :
        raise ValueError, "The order of integration must be positive"
    if len(k) > m :
        raise ValueError, "Too many integration constants"

    [cs] = _as_series([cs])
    k = list(k) + [0]*(m - len(k))
    zs = _cseries_to_zseries(cs)
    mid = len(cs)
    for i in range(m) :
        zs = _zseries_int(zs)
        zs[mid] += k[i]
        mid += 1
    ret = _zseries_to_cseries(zs)
    return ret


def chebval(x, cs, domain=None):
    """Evaluate a Chebyshev series.

    If `cs` is of length `n`, this function returns :

        cs[0]*T_{n-1}(x) + cs[1]*T_{n-2}(x) + ... + cs[n-1]*T_0(x)

    If x is a sequence then p(x) will be returned for all elements of x.
    If x is another polynomial then the composite polynomial p(x) will be
    returned.

    Parameters
    ----------
    x : array_like
        Array of numbers or objects that support multiplication and
        addition with themselves and with the elements of `cs`.
    cs : array_like
        1-d array of Chebyshev coefficients ordered from low to high.
    domain : {None, [beg, end]}
        Domain of the Chebyshev series. If not None, then the values in `x`
        are shifted and scaled so that `beg` maps to -1 and `end` maps to
        1.

    Returns
    -------
    values : ndarray
        The return array has the same shape as `x`.

    See Also
    --------
    chebadd, chebsub, chebmul, chebdiv, chebder, chebint, chebfit

    Examples
    --------

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    Examples
    --------

    """
    [cs] = _as_series([cs])
    x = npc.array(x, copy=0)
    if x.size == 0 :
        return x

    if domain is not None :
        [beg, end] = domain
        x = (2*x - (end + beg))/(end - beg)
    if len(cs) == 1 :
        c0 = cs[0]
        c1 = 0
    elif len(cs) == 2 :
        c0 = cs[0]
        c1 = cs[1]
    else :
        x2 = 2*x
        c0 = cs[-3] - cs[-1]
        c1 = cs[-2] + cs[-1]*x2
        for i in range(4, len(cs) + 1) :
            tmp = c0
            c0 = cs[-i] - c1
            c1 = tmp + c1*x2
    val = c0 + c1*x
    return val


def chebfit(x, y, deg, domain=None, rcond=None, full=False):
    """Least squares fit of Chebyshev series to data.

    Fit a Chebyshev series ``p(x) = p[0] * T_{deq}(x) + ... + p[deg] *
    T_{0}(x)`` of degree `deg` to points `(x, y)`. Returns a vector of
    coefficients `p` that minimises the squared error.

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int
        Degree of the fitting polynomial
    domain : {None, [beg, end]}
        The `x` values are shifted and scaled so that `beg` maps to -1 and
        `end` maps to 1.  The default is `[x.min(), xmax()]`.
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is len(x)*eps, where eps is the relative precision of
        the float type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is False (the
        default) just the coefficients are returned, when True diagnostic
        information from the singular value decomposition is also returned.

    Returns
    -------
    coef, [beg, end] : ndarray, shape (M,) or (M, K)
        Chebyshev coefficients ordered from low to high and the interval
        used in scaling. If `y` was 2-D, the coefficients for `k`-th data
        set are in ``p[:,k]``.

    [residuals, rank, singular_values, rcond] : present when `full` = True
        Residuals of the least-squares fit, the effective rank of the
        scaled Vandermonde coefficient matrix, its singular values, and the
        specified value of `rcond`. For more details, see `linalg.lstsq`.

    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient. The warning is only raised if `full` = False.  The
        warnings can be turned off by

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.RankWarning)

    See Also
    --------
    chebval : Computes polynomial values.
    polyfit : least squares fit of a polynomial.
    linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    Fixme
    The solution are the coefficients `c[i]` of the Chebyshev series `T(x)`
    that minimize the squared error

    .. math ::
        E = \\sum_{j=0}^k |T(x_j) - y_j|^2.

    where ::

        T(x) = T_n(x) * c[n] + ... + T_1(x) * c[1] + T_0(x) * c[0]

    This problem is usually set up as the following overdetermined system
    of linear equations ::

        T_n(x[0]) * c[n] + ... + T_1(x[0]) * c[1] + T_0(x[0]) * c[0] = y[0]
        T_n(x[1]) * c[n] + ... + T_1(x[1]) * c[1] + T_0(x[1]) * c[0] = y[1]
        ...
        T_n(x[k]) * c[n] + ... + T_1(x[k]) * c[1] + T_0(x[k]) * c[0] = y[k]

    Whose least squares solution can be obtained using the singular value
    decomposition of the design matrix `T_i(x[j])`. If some of the singular
    values are so small that they are neglected then a `RankWarning` will
    be issued. This means that the coeficient values are likey poorly
    determined due to. Using a lower order fit will usually get rid of the
    warning.  The `rcond` parameter can also be set to a value smaller than
    its default, but the resulting fit may be spurious: including
    contributions from the small singular values can add numerical noise to
    the result.

    Fits using Chebyshev series are usually better   conditioned than fits
    using power series, but much can depend on the distribution of the
    sample points and the smoothness of the data. If the quality of the fit
    is inadequate splines may be a good alternative.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           http://en.wikipedia.org/wiki/Curve_fitting

    Examples
    --------

    """
    order = int(deg) + 1
    x = npc.asarray(x) + 0.0
    y = npc.asarray(y) + 0.0

    # check arguments.
    if deg < 0 :
        raise ValueError, "expected deg >= 0"
    if x.ndim != 1:
        raise TypeError, "expected 1D vector for x"
    if x.size == 0:
        raise TypeError, "expected non-empty vector for x"
    if y.ndim < 1 or y.ndim > 2 :
        raise TypeError, "expected 1D or 2D array for y"
    if x.shape[0] != y.shape[0] :
        raise TypeError, "expected x and y to have same length"

    # set rcond
    if rcond is None :
        rcond = len(x)*npc.finfo(x.dtype).eps

    # scale x to interval [-1, 1]
    if domain == None :
        beg, end = x.min(), x.max()
    else :
        [beg, end] = domain
    x = (2*x - (end + beg))/(end - beg)

    # set up the design matrix and solve the least squares equation
    A = npc.ones((order, len(x)), dtype=x.dtype)
    if order > 1 :
        x2 = 2*x
        A[1] = x
        for i in range(2, order) :
            A[i] = x2*A[i-1] - A[i-2]
    c, resids, rank, s = la.lstsq(A.T, y, rcond)

    # warn on rank reduction
    if rank != order and not full:
        msg = "The fit may be poorly conditioned"
        warnings.warn(msg, RankWarning)

    if full :
        return c, [beg, end], [resids, rank, s, rcond]
    else :
        return c, [beg, end]


def chebroots(cs):
    """Roots of a Chebyshev series.

    The values in the rank-1 array `p` are coefficients of a polynomial.
    If the length of `p` is n+1 then the polynomial is described by
    p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]

    Parameters
    ----------
    cs : array_like of shape(M,)
        1D array of Chebyshev coefficients ordered from low to high.

    Returns
    -------
    out : ndarray
        An array containing the complex roots of the chebyshev series.

    Raises
    ------
    ValueError:

    Examples
    --------

    """
    [cs] = _as_series([cs])
    cs = chebtrim(cs)
    if len(cs) <= 1 :
        return npc.array([], dtype=cs.dtype)
    if len(cs) == 2 :
        return npc.array([-cs[0]/cs[1]])
    n = len(cs) - 1
    cmat = npc.zeros((n,n), dtype=cs.dtype)
    cmat.flat[1::n+1] = .5
    cmat.flat[n::n+1] = .5
    cmat[1, 0] = 1
    cmat[:,-1] -= cs[:-1]*(.5/cs[-1])
    roots = la.eigvals(cmat)
    roots.sort()
    return roots


#
# Chebyshev series class
#


class Chebyshev(object):
    """A Chebyshev series class.

    Parameters
    ----------
    coef : array_like
        Chebyshev coefficients, in decreasing order.  For example, ``(1, 2,
        3)`` implies :math:`T_2 + 2T_1 + 3T_0`.
    domain : {[-1, 1], [beg, end]}
        Domain to use. The interval ``[beg, end]`` is mapped to the
        interval ``[-1, 1]`` by shifting and scaling.

    Attributes
    ----------
    coef : 1D array
        Chebyshev coefficients, in decreasing order.  For example,
    domain : [beg, end]
        Domain that is mapped to [-1, 1].

    Notes
    -----
    It is important to specify the domain for many uses of Chebyshev
    series, for instance in fitting data. This is because the important
    equi-ripple property of Chebyshev polynomials only holds in the
    interval ``[-1,1]`` and thus the data must be mapped into that domain
    in order to benefit.

    Examples
    --------

    """
    # Limit runaway size. T_n^m has degree n*2^m
    maxpower = 16

    def __init__(self, coef, domain=[-1, 1]) :
        [self.coef] = _as_series([coef], trim=False)
        self.deg = len(self.coef) - 1
        self.domain = domain[:]


    def __repr__(self):
        coef = repr(self.coef)
        domain = repr(self.domain)
        return "Chebyshev(%s, %s)" % (coef, domain)


    def __str__(self) :
        format = "Chebyshev(%d)"
        return format % len(self.coef)


# Pickle and copy


    def __getstate__(self) :
        ret = self.__dict__.copy()
        ret['coef'] = self.coef.copy()
        ret['domain'] = self.domain[:]
        return ret

    def __setstate__(self, dict) :
        self.__dict__ = dict


# Call


    def __call__(self, val) :
        return chebval(val, self.coef, self.domain)


# Container properties. Do we want these?
# Think no, just access coef. The access by
# T_n, given n , might be useful though


    def __getitem__(self, ind) :
        """Get coefficients by order"""
        if isinstance(ind, slice) :
            start = ind.start
            stop = ind.stop
            step = ind.step
            return self.coef[::-1][start:stop:step].copy()
        elif 0 <= ind < len(self.coef) :
            return self.coef[::-1][i]
        else :
            raise IndexError("Out of range")


    def __setitem__(self, ind, val) :
        """Set coefficients by order"""
        if isinstance(ind, slice) :
            self.coef[ind.start:ind.stop:ind.step] = val
        elif ind < 0 :
            raise ValueError("Order out of range")
        elif ind < len(self.coef) :
            self.coef[::-1][ind] = val
        else :
            coef = npc.zeros(ind + 1, dtype=self.coef.dtype)
            coef[::-1][:len(self.coef)] = self.coef
            coef[::-1][ind] = val
            self.coef = coef

    def __iter__(self) :
        return iter(self.coef[::-1])


    def __len__(self) :
        return len(self.coef)


# Numeric properties.


    def __neg__(self) :
        return Chebyshev(-self.coef, self.domain)


    def __pos__(self) :
        return self


    def __add__(self, other) :
        """Returns sum"""
        if isinstance(other, Chebyshev) :
            if self.domain == other.domain :
                coef = chebadd(self.coef, other.coef)
            else :
                raise ChebyshevDomainError()
        else :
            try :
                coef = chebadd(self.coef, other)
            except :
                return NotImplemented
        return Chebyshev(coef, self.domain)


    def __sub__(self, other) :
        """Returns difference"""
        if isinstance(other, Chebyshev) :
            if self.domain == other.domain :
                coef = chebsub(self.coef, other.coef)
            else :
                raise ChebyshevDomainError()
        else :
            try :
                coef = chebsub(self.coef, other)
            except :
                return NotImplemented
        return Chebyshev(coef, self.domain)


    def __mul__(self, other) :
        """Returns product"""
        if isinstance(other, Chebyshev) :
            if self.domain == other.domain :
                coef = chebmul(self.coef, other.coef)
            else :
                raise ChebyshevDomainError()
        else :
            try :
                coef = chebmul(self.coef, other)
            except :
                return NotImplemented
        return Chebyshev(coef, self.domain)


    def __div__(self, other):
        # set to __floordiv__ /.
        return self.__floordiv__(other)


    def __truediv__(self, other) :
        # there is no true divide if the rhs is not a scalar, although it
        # could return the first n elements of an infinite series.
        # It is hard to see where n would come from, though.
        print 'hello'
        if isinstance(other, Chebyshev) :
            if len(other.coef) == 1 :
                coef = chebdiv(self.coef, other.coef)
            else :
                return NotImplemented
        elif npc.isscalar(other) :
            coef = self.coef/other
        else :
            return NotImplemented
        return Chebyshev(coef, self.domain)


    def __floordiv__(self, other) :
        """Returns the quotient."""
        print 'floor'
        if isinstance(other, Chebyshev) :
            if self.domain == other.domain :
                quo, rem = chebdiv(self.coef, other.coef)
            else :
                raise ChebyshevDomainError()
        else :
            try :
                quo, rem = chebdiv(self.coef, other)
            except :
                return NotImplemented
        return Chebyshev(quo, self.domain)


    def __mod__(self, other) :
        """Returns the remainder."""
        if isinstance(other, Chebyshev) :
            if self.domain == other.domain :
                quo, rem = chebdiv(self.coef, other.coef)
            else :
                raise ChebyshevDomainError()
        else :
            try :
                quo, rem = chebdiv(self.coef, other)
            except :
                return NotImplemented
        return Chebyshev(rem, self.domain)


    def __divmod__(self, other) :
        """Returns quo, remainder"""
        if isinstance(other, Chebyshev) :
            if self.domain == other.domain :
                quo, rem = chebdiv(self.coef, other.coef)
            else :
                raise ChebyshevDomainError()
        else :
            try :
                quo, rem = chebdiv(self.coef, other)
            except :
                return NotImplemented
        return Chebyshev(rem, self.domain), Chebyshev(rem, self.domain)

    def __pow__(self, other) :
        if not (isscalar(other) and int(other) == other and 0 <= other) :
            raise ValueError("Power must be a positive integer.")
        if other >= maxpower :
            raise ValueError("Power is too big.")
        if other == 0 :
            return Chebyshev(1, self.domain)
        elif other == 1 :
            return Chebyshev(self.coef, self.domain)
        else :
            # This can be made more efficient by using powers of two
            # in the usual way.
            z = _as_zseries(self.coef)
            prd = z
            for i in range(2, other) :
                prd = _zseries_mul(prd, z)
            prd = _zseries_to_cseries(prd)
            return Chebyshev(prd, self.domain)
        return NotImplemented


    def __radd__(self, other) :
        try :
            coef = chebadd(self.coef, other)
        except :
            return NotImplemented
        return Chebyshev(coef, self.domain)


    def __rsub__(self, other):
        try :
            coef = chebsub(self.coef, other)
        except :
            return NotImplemented
        return Chebyshev(coef, self.domain)


    def __rmul__(self, other) :
        try :
            coef = chebmul(self.coef, other)
        except :
            return NotImplemented
        return Chebyshev(coef, self.domain)


    def __rdiv__(self, other):
        # set to __floordiv__ /.
        return self.__rfloordiv__(other)


    def __rtruediv__(self, other) :
        # there is no true divide if the rhs is not a scalar, although it
        # could return the first n elements of an infinite series.
        # It is hard to see where n would come from, though.
        print 'hello'
        if len(self.coef) == 1 :
            try :
                quo, rem = chebdiv(other, self.coef[0])
            except :
                return NotImplemented
        return Chebyshev(quo, self.domain)


    def __rfloordiv__(self, other) :
        try :
            quo, rem = chebdiv(other, self.coef)
        except :
            return NotImplemented
        return Chebyshev(quo, self.domain)


    def __rmod__(self, other) :
        try :
            quo, rem = chebdiv(other, self.coef)
        except :
            return NotImplemented
        return Chebyshev(rem, self.domain)


    def __rdivmod__(self, other) :
        try :
            quo, rem = chebdiv(other, self.coef)
        except :
            return NotImplemented
        return Chebyshev(rem, self.domain), Chebyshev(rem, self.domain)


# Enhance me
# some augmented arithmetic operations could be added here


    def __eq__(self, other) :
        res = isinstance(other, Chebyshev) \
                and len(self.coef) == len(other.coef) \
                and self.domain == other.domain \
                and npc.alltrue(self.coef == other.coef)
        return res


    def __ne__(self, other) :
        return not self.__eq__(other)


    def trim(self, tol=0) :
        """Remove small leading coefficients

        Remove leading coefficients until a coefficient is reached whose
        absolute value greater than `tol` or the end of the series is
        reached. If all the coefficients would be removed the series is set
        to `[0]`. A new Chebyshev instance is returned with the new
        coefficients, The current instance remains unchanged.

        Parameters:
        -----------
        tol : non-negative number.
            All leading coefficients less than `tol` will be removed.

        Returns:
        -------
        new_instance : Chebyshev
            Contains the new set of coefficients.

        """
        return Chebyshev(chebtrim(self.coef,tol), self.domain)


    def trunc(self, size) :
        """Truncate series by discarding leading coefficients.

        Reduce the Chebyshev series to length `size` by removing leading
        coefficients. The value of `size` must be greater than zero.

        Parameters:
        -----------
        size : int
            The series is reduced to length `size` by discarding leading
            coefficients. The value of `size` must be greater than zero.

        Returns:
        -------
        new_instance : Chebyshev
            Contains the new set of coefficients.

        """
        if size < 1 :
            raise ValueError("size must be > 0")
        if size >= len(self.coef) :
            return Chebyshev(self.coef, self.domain)
        else :
            return Chebyshev(self.coef[-size:], self.domain)


    def copy(self) :
        """Return a copy.

        A new instance of Chebyshev is returned that has the same values of
        the coefficients and domain as the current instance.

        Returns:
        --------
        new_instance : Chebyshev
            Contains copies of the coefficients and domain.

        """
        return Chebyshev(self.coef, self.domain)


    def integ(self, m=1, k=[]) :
        """Integrate

        Return the definite integral of the Chebyshev series. Refer to
        `chebint` for full documentation.

        See Also
        --------
        chebint : equivalent function


        """
        scale = ((self.domain[1] - self.domain[0])/2)**m
        return Chebyshev(chebint(self.coef, m, k)/scale, self.domain)


    def deriv(self, m=1):
        """Differentiate.

        Return the derivative of the Chebyshev series.  Refer to `chebder`
        for full documentation.

        See Also
        --------
        chebder : equivalent function

        """
        scale = ((self.domain[1] - self.domain[0])/2)**m
        return Chebyshev(chebder(self.coef, m)*scale, self.domain)

    def roots(self) :
        """Return list of roots.

        Return list of roots. See `chebroots` for full documentation.

        See Also
        --------
        chebder : equivalent function

        """
        return Chebyshev(chebroots(self.coef), domain=self.domain)


    @staticmethod
    def fit(x, y, deg, domain=None, rcond=None, full=False) :
        """Return least squares fit to `y` sampled at points `x`

        Return the least squares Chebyshev fit to the data `y` sampled at
        `x` as a Chebyshev object. Using Chebyshev series instead of a
        polynomial is generally better numerically.  See chebfit for full
        documentation.

        See Also
        --------
        chebfit : equivalent function

        """
        res = chebfit(x, y, deg, domain=none, rcond=none, full=full)
        if full :
            [coef, domain, status] = res
            return Chebyshev(coef, domain=domain), status
        else :
            [coef, domain] = res
            return Chebyshev(coef, domain=domain)

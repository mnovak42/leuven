"""
Functions for bandwidth selection
---------------------------------

Taken form :math:`\\texttt{KDEpy}` and simplified to a minimum that is required to 
estimate kernel bandwidth parameter. The kernel bandwidth in :math:`\\texttt{KDEpy}`
is the standard deviation :math:`\sigma` that corresponds to :math:`t=\sigma^2` 
in :cite:`botev2010kernel` and :math:`h=2\sigma^2` in case of 
:math:`K(x_i, x_j) = \\exp[-\\frac{(x_i-x_j)^2}{h}]` 

Functions for 1D Gaussian kernel bandwidth (h) estimations: `Silverman` and 
`Improved Sheather Jones` :cite:`botev2010kernel`.

References:
-----------

- KDEpy documentation at https://kdepy.readthedocs.io/en/latest/ and 
  https://github.com/tommyod/KDEpy
  
"""
import numpy as np
import scipy
import warnings
from scipy import fftpack
from scipy.optimize import brentq

# Choose the largest available float on the system
try:
    FLOAT = scipy.float128
except AttributeError:
    FLOAT = np.float64


def _fixed_point(t, N, I_sq, a2):
    r"""
    Compute the fixed point as described in the paper by Botev et al.

    .. math:

        t = \xi \gamma^{5}(t)

    Parameters
    ----------
    t : float
        Initial guess.
    N : int
        Number of data points.
    I_sq : array-like
        The numbers [1, 2, 9, 16, ...]
    a2 : array-like
        The DCT of the original data, divided by 2 and squared.

    Examples
    --------
    >>> # From the matlab code
    >>> ans = _fixed_point(0.01, 50, np.arange(1, 51), np.arange(1, 51))
    >>> assert np.allclose(ans, 0.009947962622371)
    >>> # another
    >>> ans = _fixed_point(0.07, 25, np.arange(1, 11), np.arange(1, 11))
    >>> assert np.allclose(ans, 0.069100181315957)

    References
    ----------
     - Implementation by Daniel B. Smith, PhD, found at
       https://github.com/Daniel-B-Smith/KDE-for-SciPy/blob/master/kde.py
    """

    # This is important, as the powers might overflow if not done
    I_sq = np.asfarray(I_sq, dtype=FLOAT)
    a2 = np.asfarray(a2, dtype=FLOAT)

    # ell = 7 corresponds to the 5 steps recommended in the paper
    ell = 7

    # Fast evaluation of |f^l|^2 using the DCT, see Plancherel theorem
    f = 2 * np.pi ** (2 * ell) * np.sum(np.power(I_sq, ell) * a2 * np.exp(-I_sq * np.pi ** 2 * t))

    # Norm of a function, should never be negative
    if f <= 0:
        return -1
    for s in reversed(range(2, ell)):
        # This could also be formulated using the double factorial n!!,
        # but this is faster so and requires an import less

        # Step one: estimate t_s from |f^(s+1)|^2
        odd_numbers_prod = np.product(np.arange(1, 2 * s + 1, 2, dtype=FLOAT))
        K0 = odd_numbers_prod / np.sqrt(2 * np.pi)
        const = (1 + (1 / 2) ** (s + 1 / 2)) / 3
        time = np.power((2 * const * K0 / (N * f)), (2.0 / (3.0 + 2.0 * s)))

        # Step two: estimate |f^s| from t_s
        f = 2 * np.pi ** (2 * s) * np.sum(np.power(I_sq, s) * a2 * np.exp(-I_sq * np.pi ** 2 * time))

    # This is the minimizer of the AMISE
    t_opt = np.power(2 * N * np.sqrt(np.pi) * f, -2.0 / 5)

    # Return the difference between the original t and the optimal value
    return t - t_opt


def _root(function, N, args):
    """
    Root finding algorithm. Based on MATLAB implementation by Botev et al.

    >>> # From the matlab code
    >>> ints = np.arange(1, 51)
    >>> ans = _root(_fixed_point, N=50, args=(50, ints, ints))
    >>> np.allclose(ans, 5.203713947289470e-05)
    True
    """
    # From the implementation by Botev, the original paper author
    # Rule of thumb of obtaining a feasible solution
    N = max(min(1050, N), 50)
    tol = 10e-12 + 0.01 * (N - 50) / 1000
    # While a solution is not found, increase the tolerance and try again
    found = 0
    while found == 0:
        try:
            # Other viable solvers include: [brentq, brenth, ridder, bisect]
            x, res = brentq(function, 0, tol, args=args, full_output=True, disp=False)
            found = 1 if res.converged else 0
        except ValueError:
            x = 0
            tol *= 2.0
            found = 0
        if x <= 0:
            found = 0

        # If the tolerance grows too large, minimize the function
        if tol >= 1:
            raise ValueError("Root finding did not converge. Need more data.")

    if not x > 0:
        raise ValueError("Root finding failed to find positive solution.")
    return x


def linbin_numpy(data, grid_points, weights=None):
    """
    1D Linear binning using NumPy. Assigns weights to grid points from data.
    This function is fast for data sets upto approximately 1-10 million,
    it uses vectorized NumPy functions to perform linear binning. Takes around
    100 ms on 1 million data points, so not nearly as fast as the Cython
    implementation (10 ms).
    
    Taken form KDEpy.binning (https://github.com/tommyod/KDEpy/blob/8ab02e0eada8
    bdd07b7cead0b76d6be0226d8c56/KDEpy/binning.py#L116)
    
    References
    ----------
    - Fan, Jianqing, and James S. Marron.
      “Fast Implementations of Nonparametric Curve Estimators.”
      Journal of Computational and Graphical Statistics 3, no. 1 (March 1, 1994).
      https://doi.org/10.1080/10618600.1994.10474629.
    
    Parameters
    ----------
    data : array-like
        Must be of shape (obs,).
    grid_points : array-like
        Must be of shape (points,).
    weights : array-like
        Must be of shape (obs,).
    Examples
    --------
    >>> data = np.array([2, 2.5, 3, 4])
    >>> ans = linbin_numpy(data, np.arange(6), weights=None)
    >>> np.allclose(ans, np.array([0, 0, 0.375, 0.375, 0.25, 0]))
    True
    >>> ans = linbin_numpy(data, np.arange(6), weights=np.arange(1, 5))
    >>> np.allclose(ans, np.array([0, 0, 0.2, 0.4, 0.4, 0]))
    True
    >>> data = np.array([2, 2.5, 3, 4])
    >>> ans = linbin_numpy(data, np.arange(1, 7), weights=None)
    >>> np.allclose(ans, np.array([0, 0.375, 0.375, 0.25, 0, 0]))
    True
    """
    # Convert the data and grid points
    data = np.asarray_chkfinite(data, dtype=np.float)
    grid_points = np.asarray_chkfinite(grid_points, dtype=np.float)
    assert len(data.shape) == 1
    assert len(grid_points.shape) == 1

    # Verify that the grid is equidistant
    diffs = np.diff(grid_points)
    assert np.allclose(np.ones_like(diffs) * diffs[0], diffs)

    if weights is None:
        weights = np.ones_like(data)

    weights = np.asarray_chkfinite(weights, dtype=np.float)
    weights = weights / np.sum(weights)

    if not len(data) == len(weights):
        raise ValueError("Length of data must match length of weights.")

    # Transform the data
    min_grid = np.min(grid_points)
    max_grid = np.max(grid_points)
    num_intervals = len(grid_points) - 1
    dx = (max_grid - min_grid) / num_intervals
    transformed_data = (data - min_grid) / dx

    # Compute the integral and fractional part of the data
    # The integral part is used for lookups, the fractional part is used
    # to weight the data
    fractional, integral = np.modf(transformed_data)
    integral = integral.astype(np.int)

    # Sort the integral values, and the fractional data and weights by
    # the same key. This lets us use binary search, which is faster
    # than using a mask in the the loop below
    indices_sorted = np.argsort(integral)
    integral = integral[indices_sorted]
    fractional = fractional[indices_sorted]
    weights = weights[indices_sorted]

    # Pre-compute these products, as they are used in the loop many times
    frac_weights = fractional * weights
    neg_frac_weights = weights - frac_weights

    # If the data is not a subset of the grid, the integral values will be
    # outside of the grid. To solve the problem, we filter these values away
    unique_integrals = np.unique(integral)
    unique_integrals = unique_integrals[(unique_integrals >= 0) & (unique_integrals <= len(grid_points))]

    result = np.asfarray(np.zeros(len(grid_points) + 1))
    for grid_point in unique_integrals:

        # Use binary search to find indices for the grid point
        # Then sum the data assigned to that grid point
        low_index = np.searchsorted(integral, grid_point, side="left")
        high_index = np.searchsorted(integral, grid_point, side="right")
        result[grid_point] += neg_frac_weights[low_index:high_index].sum()
        result[grid_point + 1] += frac_weights[low_index:high_index].sum()

    return result[:-1]


def improved_sheather_jones(data):
    """
    The Improved Sheater Jones (ISJ) algorithm from the paper by Botev et al.
    This algorithm computes the optimal bandwidth for a gaussian kernel,
    and works very well for bimodal data (unlike other rules). The
    disadvantage of this algorithm is longer computation time, and the fact
    that this implementation does not always converge if very few data
    points are supplied.

    Understanding this algorithm is difficult, see:
    https://books.google.no/books?id=Trj9HQ7G8TUC&pg=PA328&lpg=PA328&dq=
    sheather+jones+why+use+dct&source=bl&ots=1ETdKd_6EF&sig=jZk4R515GB1xsn-
    VZVnjr-JfjSI&hl=en&sa=X&ved=2ahUKEwi1_czNncTcAhVGhqYKHaPiBtcQ6AEwA3oEC
    AcQAQ#v=onepage&q=sheather%20jones%20why%20use%20dct&f=false
    """
    obs, dims = data.shape
    if not dims == 1:
        raise ValueError("ISJ is only available for 1D data.")

    n = 2 ** 10
    # Setting `percentile` higher decreases the chance of overflow
    #xmesh = autogrid(data, boundary_abs=6, num_points=n, boundary_rel=0.5)
    minDat, maxDat = data.min(axis=0), data.max(axis=0)
    rangeDat = maxDat-maxDat 
    outside_borders = max(0.5*rangeDat, 6)
    xmesh = np.linspace(minDat - outside_borders, maxDat + outside_borders, num=n)
    data  = data.ravel()
    xmesh = xmesh.ravel()
    #
    # Create an equidistant grid
    R = np.max(data) - np.min(data)
    # dx = R / (n - 1)
    #data = data.ravel()
    N = len(np.unique(data))
    # Use linear binning to bin the data on an equidistant grid, this is a
    # prerequisite for using the FFT (evenly spaced samples)
    #initial_data = linear_binning(data.reshape(-1, 1), xmesh)
    #print("===== ", (data.reshape(-1, 1)).shape, " vs ", data.shape)
    initial_data = linbin_numpy(data, xmesh)
    assert np.allclose(initial_data.sum(), 1)
    # Compute the type 2 Discrete Cosine Transform (DCT) of the data
    a = fftpack.dct(initial_data)
    # Compute the bandwidth
    I_sq = np.power(np.arange(1, n, dtype=FLOAT), 2)
    a2 = a[1:] ** 2 / 4
    # Solve for the optimal (in the AMISE sense) t
    t_star = _root(_fixed_point, N, args=(N, I_sq, a2))

    # The remainder of the algorithm computes the actual density
    # estimate, but this function is only used to compute the
    # bandwidth, since the bandwidth may be used for other kernels
    # apart from the Gaussian kernel

    # Smooth the initial data using the computed optimal t
    # Multiplication in frequency domain is convolution
    # integers = np.arange(n, dtype=np.float)
    # a_t = a * np.exp(-integers**2 * np.pi ** 2 * t_star / 2)

    # Diving by 2 done because of the implementation of fftpack.idct
    # density = fftpack.idct(a_t) / (2 * R) 

    # Due to overflow, some values might be smaller than zero, correct it
    # density[density < 0] = 0.
    bandwidth = np.sqrt(t_star) * R
    return bandwidth

'''
#
# We use 1D estimates and since scotts_rule=silvermans_rule if dimensions of data 
# is equal to 1, we don't use this
#
def scotts_rule(data):
    """
    Scotts rule.

    Scott (1992, page 152)
    Scott, D.W. (1992) Multivariate Density Estimation. Theory, Practice and
    Visualization. New York: Wiley.

    Examples
    --------
    >>> data = np.arange(9).reshape(-1, 1)
    >>> ans = scotts_rule(data)
    >>> assert np.allclose(ans, 1.76474568962182)
    """
    if not len(data.shape) == 2:
        raise ValueError("Data must be of shape (obs, dims).")

    obs, dims = data.shape
    if not dims == 1:
        raise ValueError("Scotts rule is only available for 1D data.")
    sigma = np.std(data, ddof=1)
    # scipy.norm.ppf(.75) - scipy.norm.ppf(.25) -> 1.3489795003921634
    IQR = (np.percentile(data, q=75) - np.percentile(data, q=25)) / 1.3489795003921634

    sigma = min(sigma, IQR)
    return sigma * np.power(obs, -1.0 / (dims + 4))
'''

def silvermans_rule(data):
    """
    Returns optimal smoothing (standard deviation) if the data is close to
    normal.

    Examples
    --------
    >>> data = np.arange(9).reshape(-1, 1)
    >>> ans = silvermans_rule(data)
    >>> assert np.allclose(ans, 1.8692607078355594)
    """
    if not len(data.shape) == 2:
        raise ValueError("Data must be of shape (obs, dims).")
    obs, dims = data.shape
    if not dims == 1:
        raise ValueError("Silverman's rule is only available for 1D data.")

    if obs == 1:
        return 1
    if obs < 1:
        raise ValueError("Data must be of length > 0.")

    sigma = np.std(data, ddof=1)
    # scipy.stats.norm.ppf(.75) - scipy.stats.norm.ppf(.25) -> 1.3489795003921634
    IQR = (np.percentile(data, q=75) - np.percentile(data, q=25)) / 1.3489795003921634

    sigma = min(sigma, IQR)

    # The logic below is not related to silverman's rule, but if the data is constant
    # it's nice to return a value instead of getting an error. A warning will be raised.
    if sigma > 0:
        return 0.85*sigma * (obs * 3 / 4.0) ** (-1 / 5)
    else:
        # stats.norm.ppf(.99) - stats.norm.ppf(.01) = 4.6526957480816815
        IQR = (np.percentile(data, q=99) - np.percentile(data, q=1)) / 4.6526957480816815
        if IQR > 0:
            bw = IQR * (obs * 3 / 4.0) ** (-1 / 5)
            warnings.warn(
                "Silverman's rule failed. Too many identical values. \
Setting bw = {}".format(
                    bw
                )
            )
            return bw

        # Here, all values are basically constant
        warnings.warn("Silverman's rule failed. Too many identical values. Setting bw = 1.0")
        return 1.0


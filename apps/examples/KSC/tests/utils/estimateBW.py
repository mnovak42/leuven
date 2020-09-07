"""
Estimate Optimal Kernel BandWidth
=================================

Optimal kernel bandwith parameter :math:`\\gamma` estimation for the RBF (Gaussian) 
kernel in the form of :math:`K(x_i, x_j) = \\exp[-\\frac{(x_i-x_j)^2}{\\gamma}]`. 

Description
-----------

Given a :math:`\{\mathbf{x}_i\}_{i=1}^{N}, \mathbf{x}_i\in \mathbb{R}^{d}`
`independent, identically distributed` sample from an unknown `continuous`
`probability density function` (pdf) :math:`f` , the kernel density estimator
for :math:`d=1` is defined as 

.. math::

     \hat{f}(x;h) = \\frac{1}{Nh} \sum_{i=1}^{N} K \\left[ \\frac{x-x_i}{h} \\right]

which becoms

.. math::

     \hat{f}_G(x;h) = \\frac{1}{N\sqrt{2\pi h^2}} \sum_{i=1}^{N} \exp
                         \\left[ 
                           -\\frac{(x-x_i)^2}{2h^2}
                         \\right]

when using a Gaussian kernel :math:`K(\cdot) = K_G(\cdot)` in the form of (standard 
normal pdf)

.. math:: 

    K_G(\\alpha) = \\frac{1}{\sqrt{2\pi}} \exp \\left[ -\\frac{\\alpha^2}{2} \\right]

with :math:`\\alpha=(x-x_i)/h` and our bandwidth parameter :math:`\\gamma = 2h^2` 
(h is the standard deviation).

When estimating the value of the smoothing parameter :math:`h` , the most common 
optimality criterion is the expected :math:`L_2` risk function or Mean 
Integrated Squered Error (MISE)

.. math::
   
    \\texttt{MISE}[\hat{f}_G(x;h)] = \mathbb{E}_f \\left\{ 
          \int \\left[
             \hat{f}_G(x;h) - f(x)
           \\right]^2 \\right\}

The :math:`\\texttt{MISE}` has a complicated dependence on :math:`f` and :math:`h`
that significantly simplifies when its asymptotic form (:math:`N\\to\infty`), 
the :math:`\\texttt{AMISE}` is considered. The (asymptotically) optimal bandwidth 
is deteremined by finding the minimum :math:`\\texttt{AMISE}` (assuming twice 
continuously differentiable generating density function :math:`f`) :cite:`botev2010kernel` .


Available algorithms
--------------------

The following two algorithms are available (taken from :math:`\\texttt{KDEpy}` 
:cite:`kdepy` ) for estimating the kernel bandwidth 
parameter for RBF kernels along the single dimensions (1D):

- ``Silverman`` - **Silverman's rule of thumb** : if the underlying, unknown generating 
  density is Gaussian and Gaussian kernel is used to approximate it, then the 
  :math:`\\texttt{MISE}` optimal bandwith estimate :cite:`silverman1986density`
    
  .. math::
      
      h = \\left[ \\frac{4\hat{\sigma}^5}{3N} \\right]^{\\frac{1}{5}} 
        \\approx 1.06\hat{\sigma}N^{-\\frac{1}{5}}
        
  which can be make more robut (for long-tailed, skewed and bimodal mixture 
  distributions) one can replace the estimated standard deviation :math:`\hat{\sigma}`
  with the *Interquantile Range* (:math:`IQR = Q_3-Q_1` i.e. the difference between 
  the upper and lower quartiles) in case :math:`\hat{\sigma} > \\texttt{IQR}/1.34`
  A further modification is to multiply the above estimate by 0.85 that yields 
  
  .. math::
  
      h = 0.85 \\left[ \\frac{4 \\text{min}[ \hat{\sigma}, \\texttt{IQR} ]  ^5}{3N} \\right]^{\\frac{1}{5}} 
        \\approx 0.9 \\text{min}[ \hat{\sigma}, \\texttt{IQR} ] N^{-\\frac{1}{5}}  
  
  .. note:: Using the above **Silverman** *rule of thumb* can lead to very 
     inaccurate estimate when the underlying estimated density is not Gaussian.
   
- ``ISJ`` - **Improved Sheather Jones (ISJ)** : the ISJ algorithm is a `plug-in`
  method: an initial pilot estimate of the underlying density is plugged-in to 
  the asymptotic form of the MISE to obtain an estimate then the optimal bandwidth
  is determine by optimising this estimate (see more :cite:`loader1999bandwidth` ). 
  The ISJ algorithm utilises a recursive formula to minimise the asymptotic 
  MISE for determining the optimal bandwidth :cite:`botev2010kernel` . 
  
  .. note:: This algorithm outperforms the simple `Silverman` rule and provides 
     very good results for Gaussian kernel bandwidth estimate even in the case of 
     bimodal data. Unlike the `Silverman` rule, this doesn't assume normality. 
     It might have convergence problem in case of insufficient numebr of data. 
 

.. _example_estimateBW:

Example:
--------

Example for calling the ``EstimateBW`` driver for a ``data.dat`` file estimating 
the optimal Gaussian kernel bandwidth h ::

   $ python estimateBW.py data.dat 'Silverman'


-----

"""

import sys

import numpy as np
import bwest as bw


# estimates kernel bandwidth along each dimensions 
def EstimateBW(data, which):
    """
    Esimates the optimal kernel bandwidth :math:`h=2\sigma^2` for Gaussian 
    kernel.

    Args:     
        fname (str) : input-matrix file (see more at the :ref:`Example <example_estimateBW>`)
        which (str) : type of the optimal bandwith estimation algorithms 
          (``Silverman`` or ``ISJ`` (default))

    Returns:
        A ``numpy::ndarray`` that contains the estimated bandwidths along 
        each dimensions.
       
    The file contains the data set for which the optimal kernel bandwidth 
    parameter needs to be determined. It is assumed, that the file contains the 
    :math:`\{\mathbf{x}_i\}_{i=1}^{N}, \mathbf{x}_i\in \mathbb{R}^{d}` input data  
    as :math:`\mathbb{R}^{(N\\times d)}` matrix i.e. each data point is a row.
           
    Loads the input data file and executes the selected algorithm to estimate
    the optimal kernel bandwidth along each of the :math:`d` dimensions of the 
    input data. 
           
    """

    ddim  = data.ndim
    if ddim==1:
       resBW = np.zeros(1, dtype=float)
       rs = np.asfarray(data).reshape(-1, 1)
       sig = 1
       if which == "ISJ":
           sig = bw.improved_sheather_jones(rs)
       elif which == "Silverman":
           sig = bw.silvermans_rule(rs)
       else :
           print (" *** unknown band width estimation option")    
       #print (' h = ', 2*sig*sig , ' sig = ', sig)
       resBW[0] = 2*sig*sig
    elif ddim==2 : 
       resBW = np.zeros(data.shape[1], dtype=float)
       for d in range(data.shape[1]):
           d1  = data[:,d]    
           rs  = np.asfarray(d1).reshape(-1, 1)
           sig = 1
           if which == "ISJ":
               sig = bw.improved_sheather_jones(rs)
           elif which == "Silverman":
               sig = bw.silvermans_rule(rs)
           else :
               print (" *** unknown band width estimation option")    
           #print (' d = ', d, '  h = ', 2*sig*sig, ' sig = ', sig)
           resBW[d] = 2*sig*sig
    else :  
       print (" *** data must be 1D or 2D (one feature per cols.)")
    return resBW
       
       
def main():
    fname = sys.argv[1]  # file name 
    which = sys.argv[2]  # 'ISJ' or 'Silverman' 
    data  = np.loadtxt(fname, dtype=float)
    res   = EstimateBW(data, which)#.astype(str)
    sres  = ",".join([np.format_float_scientific(h,precision=4) for h in res])
    print (sres)
    
    
if __name__ == "__main__":
    main()

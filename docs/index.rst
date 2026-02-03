EzTaoX
========================================================================================

Active galactic nuclei (AGNs) exhibit continuum stochastic variability on time scales ranging 
from hours to decades, and show correlations across a wide range in wavelengths (X-ray -> UV/optical -> IR). 
The variability in longer wavelengths also exhibits time delays (lags) from the variability in shorter wavelengths. 
Scalable and flexible modeling of AGN light curves (stochastic variability + lags) will not only help 
us better understand the origin of AGN variability, but also enable powerful methods for selecting AGN 
candidates in photometric time-domain surveys.

``EzTaoX`` is designed to perform multi-wavelength/multi-band modeling of AGN light curves using Gaussian Processes (GPs). 
GPs provide particular advantages in dealing with the light curves with non-uniform sampling, typical of observations 
taken by ground-based facilities.

``EzTaoX`` is built on top of ``tinygp`` - a fast and flexible GP modeling framework written using ``JAX``. 
``JAX`` is a high-performance numerical computing library that combines automatic differentiation, just-in-time (JIT) 
compilation, and hardware acceleration to empower scalable machine learning and scientific computing applications.
``tinygp`` leverages these features of ``JAX`` and re-implements the novel ``celerite`` algorithm introduced by 
[DFM17]_ in ``JAX`` to provide a :math:`O(N)` scaling for GP likelihood evaluation 
(as opposed to a :math:`O(N^3)` scaling). 

Motivation
---------------------------------------------------------

Current tools/methods used to characterize AGN UV/optical variability typically operate on single-band 
light curves and lack the ability to incorporate multi-band information. 
This limitation prevents the effective utilization of the multi-band coverage of modern time-domain surveys 
(e.g, Rubin LSST) for improved characterization of AGN variability. 

Tools/methods commonly used to measure inter-band lags generally fall into two categories. 
The first category includes techniques that estimate lags by computing the cross-correlation function (CCF) 
between light curves; the CCF reaches its maximum when one light curve is shifted relative to the other by an
amount corresponding to the intrinsic lag. 
These methods are typically fast and computationally efficient, but are limited to lag estimation and 
do not provide a full characterization of the underlying variability process.
The second category consists of methods that model each observed light curves as the convolution of 
an unknown driving continuum with a time-lag distribution, represented by a transfer function.
While such methods provide a more comprehensive treatment by jointly modeling variability and inter-band lags, 
they are often computationally demanding.

``EzTaoX`` aims to establish a new framework that can utilize the multi-band coverage of modern 
time-domain surveys to simultaneously extract stochastic variability and inter-band time delay information 
from multi-band light curves, and at the same time provide significant improvement in computational speed in 
comparison to existing tools of similar capabilities. 

Since ``tingyp`` is a more general GP modeling package, and can sometimes be overwhelming for AGN astronomers 
who are not familiar with GP modeling, ``EzTaoX`` tries to provide a more straight-forward interface for 
modeling AGN light curve using ``tinygp`` and ``JAX``. 
In addition, ``EzTaoX`` expands on top of ``tinygp`` to provide new functionalities that are useful for 
AGN light curve analysis. For example, ``EzTaoX`` provides functions for generating power spectral density 
(PSD) and structure function (SF) of any (and combination of) ``tinygp`` kernels. 
``EzTaoX`` also adds new kernels that are often used in AGN variability analysis, e.g., a Lorentzian kernel. 

MultiBand Modeling Basics
---------------------------------------------------------------------------

The ``EzTaoX`` multi-band model assumes light curves in different bands are generated from the same 
underlying latent GP, however, with different amplitudes in each band. 
In addition, light curves in different band can posses time delays with respect to each other. 
The corresponding kernel function is written as:
:math:`k(|t_j - t_i|) = S_{1}S_{2}\,k_{\rm latent}(|t_j - t_i - \tau_{\rm lag}|)`
where :math:`S` is a scale factor to transform the amplitude of the underlying latent GP to 
the observed amplitude of a single-band light curve, and 
:math:`\tau_{\rm lag}` is the time delay between corresponding bands/wavelengths.

In the ``EzTaoX`` implementation, the value of :math:`S` for the 'reference' band is set to be 1 by default. 
Thus, only the :math:`S` factors for the non-reference bands are allowed to vary (i.e., free parameters). 
To get the best-fit amplitude of the GP in a given band, one needs to multiply the best-fit amplitude of the 
latent GP (a parameter of the :math:`k_{\rm latent}`) by the corresponding :math:`S` factor of the desired band. 

See the Section II in the notebook :doc:`MultibandFitting <notebooks/01_MultibandFitting>` to learn how to 
set the reference band. 

**Note** that in this context, a reference band carries no physical meanings, but rather to simplify the 
inference process through reducing the number of parameters.
By default, ``EzTaoX`` assumes the :math:`S` factors are independent from each other, however, ``EzTaoX``
allows one to provide an additional amplitude function to correlate the :math:`S` factors.


Code Structure
------------------------------------------------------------

``EzTaoX`` consists of the following modules:

- ``kernels``:

    - ``direct``: Kernels following a :math:`O(N^3)` computational complexity.
    - ``quasisep``: Kernels following a :math:`O(N)` computational complexity. 
      This module consists of kernels from in the ``tinygp.kernels.quasisep`` module, and modified to 
      provide new functionalities. 
- ``models``: Light curve models module; the fundamental interface of `EzTaoX` for performing light curve modeling. 
  See tutorials linked below for usage examples.
- ``fitter``: A module of fitter functions. See tutorials linked below for usage examples.
- ``kernel_stat2``: A module of classes/functions to calculate 2nd order statistic of any (and combinations of) 
  GP kernels included in the ``kernels.quasisep`` module 
- ``simulator``: A module providing classes to simulate GP time series given input kernels. 
- ``ts_utils``: A module of utility functions for time-series/light-curve processing. 

Note: ``direct`` vs ``quasisep``
---------------------------------------------------------------

``tinygp`` provides two solver for GP kernels: ``direct`` and ``quasisep``. 
The ``direct`` solver follows the standard approach to perform linear algebra operations needed to evaluate the 
GP likelihood function. The standard approach can be applied to a broader range of GP kernels, however, 
obeys a :math:`O(N^3)` computational scaling, where N is the number of data points in the provided time series. 
The ``quasisep`` solver takes advantage of certain structures in the relevant GP matrices to achieve a :math:`O(N)` scaling. 
However, the ``quasisep`` solver only works for a specific class of kernels as those included in the 
``tinygp.kernels.quasisep`` (as well as the ``eztaox.kernels.quasisep``) module. 

``EzTaoX`` focuses only developing code that can benefit from the :math:`O(N)` scaling of the ``quasisep`` solver. 
Thus, it is highly recommended to use the kernels provided in the ``eztaox.kernels.quasisep`` module, 
since those are best tested. For completeness and support for a broader class of kernels, we also provide an interface 
to work with kernels that follow a :math:`O(N^3)` scaling. This part of EzTaoX is still **under development**. 

GP (quasisep) Kernels Supported
---------------------------------------------------------------

Below, we list GP kernels that can be evaluated using the ``quasisep`` solver of ``tinygp``. 
You can import them from the ``eztaox.kernels.quasisep`` module. 
Please see `here <https://tinygp.readthedocs.io/en/stable/api/kernels.quasisep.html>`__ for a definition of these kernels. 

- `Exp`: The exponential kernel---equivalent to the damped random walk (DRW) kernel.
- `Cosine`: The cosine kernel.
- `Matern32/Matern52`: The Matern32/Matern52 kernel.
- `SHO`: The noise-driven simple harmonic oscillator kernel.
- `Celerite`: The base kernel of the `celerite` algorithm. This kernel is not recommended to use directly.
- `Lorentzian`: The Lorentzian kernel. The product of an `Exp` and a `Cosine` kernel, and the scales of which are correlated. See the doc string for the definition. 
- `CARMA`: Continuous-time auto-regressive moving-average (CARMA) kernel. In the current version, CARMA process that are more complex than CAMRA(2,1) might raise bugs. 

**Note:** You can build new kernels from combining kernels in the list above, and the new kernel will still work in the `EzTaoX` framework.


Tutorials
---------------------------------------------------------------

A preliminary list of tutorials to help users getting started:

* :doc:`MultibandFitting <notebooks/01_MultibandFitting>`: Demonstrating the workflow of conducting multi-band fitting.
* :doc:`Damped Harmonic Oscillator <notebooks/02_DHO>`: Demonstrating fitting higher-order CARMA models beyond the DRW. 
  In this notebook, we are fitting the CARMA(2,1) process, which is also known as the damped harmonic oscillator (DHO) process.
* :doc:`Damped Random Walk <notebooks/03_DRW>`: Demonstrating fitting to a damped random walk.

.. toctree::
   :hidden:

   Home page <self>
   API Reference <autoapi/index>
   Notebooks <notebooks>
   About <about>

.. [DFM17] Daniel Foreman-Mackey, Eric Agol, Sivaram Ambikasaran, Ruth Angus; 
   *Fast and scalable Gaussian process modeling with applications to astronomical time series*, 
   https://arxiv.org/abs/1703.09710
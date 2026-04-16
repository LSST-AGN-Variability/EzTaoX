EzTaoX
========================================================================================

Active galactic nuclei (AGNs) exhibit continuum stochastic variability on timescales ranging
from hours to decades, and they show correlations across a wide range of wavelengths (X-ray -> UV/optical -> IR).
Variability at longer wavelengths also exhibits time delays (lags) relative to variability at shorter wavelengths.
Scalable and flexible modeling of AGN light curves (stochastic variability + lags) will not only help
us better understand the origin of AGN variability, but also enable powerful methods for selecting AGN
candidates in photometric time-domain surveys.

``EzTaoX`` is designed to perform multi-wavelength/multi-band modeling of AGN light curves using Gaussian Processes (GPs).
GPs provide particular advantages for handling light curves with non-uniform and gappy sampling, as is typical of observations
taken by ground-based facilities.

``EzTaoX`` is built on top of ``tinygp``, a fast and flexible GP modeling framework written using ``JAX``.
``JAX`` is a high-performance numerical computing library that combines automatic differentiation, just-in-time (JIT)
compilation, and hardware acceleration to empower scalable machine learning and scientific computing applications.
``tinygp`` leverages these features of ``JAX`` and re-implements the novel ``celerite`` algorithm introduced by
[DFM17]_ in ``JAX`` to provide :math:`O(N)` scaling for GP likelihood evaluation
(as opposed to :math:`O(N^3)` scaling). 

Motivation
---------------------------------------------------------

Current tools and methods used to characterize AGN UV/optical variability typically operate on single-band
light curves and lack the ability to incorporate multi-band information.
This limitation prevents the effective use of the multi-band coverage provided by modern time-domain surveys
(e.g., Rubin LSST) for improved characterization of AGN variability.

Methods commonly used to measure inter-band lags generally fall into two categories.
The first category includes techniques that estimate lags by computing the cross-correlation function (CCF)
between light curves; the CCF reaches its maximum when one light curve is shifted relative to the other by an
amount corresponding to the intrinsic lag.
These methods are typically fast and computationally efficient, but they are limited to lag estimation and
do not provide a full characterization of the underlying variability process.
The second category consists of methods that model each observed light curve as the convolution of
an unknown driving continuum with a time-lag distribution represented by a transfer function.
While such methods provide a more comprehensive treatment by jointly modeling variability and inter-band lags,
they are often computationally demanding.

``EzTaoX`` aims to establish a new framework that can use the multi-band coverage of modern
time-domain surveys to simultaneously extract stochastic variability and inter-band time-delay information
from multi-band light curves, while also providing a significant improvement in computational speed
compared with existing tools of similar capability.

Since ``tinygp`` is a more general GP modeling package and can sometimes be overwhelming for AGN astronomers
who are not familiar with GP modeling, ``EzTaoX`` aims to provide a more straightforward interface for
modeling AGN light curves using ``tinygp`` and ``JAX``.
In addition, ``EzTaoX`` builds on ``tinygp`` to provide new functionalities that are useful for
AGN light curve analysis. For example, ``EzTaoX`` provides functions for generating the power spectral density
(PSD) and structure function (SF) of any ``tinygp`` kernel or combination of kernels.
``EzTaoX`` also adds new kernels that are often used in AGN variability analysis, such as a Lorentzian kernel. 

MultiBand Modeling Basics
---------------------------------------------------------------------------

The ``EzTaoX`` multi-band model assumes that light curves in different bands are generated from the same
underlying latent GP, but with different amplitudes in each band.
In addition, light curves in different bands can possess time delays with respect to one another.
The corresponding kernel function is written as:
:math:`k(|t_j - t_i|) = S_{1}S_{2}\,k_{\rm latent}(|t_j - t_i - \tau_{\rm lag}|)`
where :math:`S` is a scale factor that transforms the amplitude of the underlying latent GP into
the observed amplitude of a single-band light curve, and
:math:`\tau_{\rm lag}` is the time delay between the corresponding bands/wavelengths.

In the ``EzTaoX`` implementation, the value of :math:`S` for the 'reference' band is set to 1 by default.
Thus, only the :math:`S` factors for the non-reference bands are allowed to vary (i.e., they are free parameters).
To obtain the best-fit amplitude of the GP in a given band, one needs to multiply the best-fit amplitude of the
latent GP (a parameter of :math:`k_{\rm latent}`) by the corresponding :math:`S` factor for the desired band.

See Section II of the notebook :doc:`MultibandFitting <notebooks/03_MultibandFitting>` to learn how to
set the reference band.

**Note** that, in this context, a reference band carries no physical meaning; it is used only to simplify the
inference process by reducing the number of parameters.
By default, ``EzTaoX`` assumes that the :math:`S` factors are independent of one another. However, ``EzTaoX``
allows the user to provide an additional amplitude function to correlate the :math:`S` factors.


Code Structure
------------------------------------------------------------

``EzTaoX`` consists of the following modules:

- ``kernels``: 
    - ``direct``: GP kernels with :math:`O(N^3)` computational complexity.
    - ``quasisep``: GP kernels with :math:`O(N)` computational complexity.
      This module includes kernels from the ``tinygp.kernels.quasisep`` module, modified to
      provide additional functionality.
- ``models``: A light-curve modeling module; the fundamental interface of ``EzTaoX`` for performing light-curve modeling. 
  See the tutorials linked below for usage examples.
- ``fitter``: A module of fitting functions. See the tutorials linked below for usage examples.
- ``kernel_stat2``: A module of classes and functions for calculating second-order statistics of any GP kernel,
  or combination of GP kernels, included in the ``kernels.quasisep`` module.
- ``simulator``: A module providing classes to simulate GP time series given input kernels.
- ``ts_utils``: A module of utility functions for time-series/light-curve processing.

``direct`` vs ``quasisep``
---------------------------------------------------------------

``tinygp`` provides two solvers for GP kernels: ``direct`` and ``quasisep``. 
The ``direct`` solver follows the standard approach for performing the linear algebra operations needed to evaluate the
GP likelihood function. This approach can be applied to a broader range of GP kernels. However,
it obeys :math:`O(N^3)` computational scaling, where N is the number of data points in the provided time series.
The ``quasisep`` solver takes advantage of certain structures in the relevant GP matrices to achieve :math:`O(N)` scaling.
However, the ``quasisep`` solver works only for a specific class of kernels, such as those included in the
``tinygp.kernels.quasisep`` module (as well as the ``eztaox.kernels.quasisep`` module).

``EzTaoX`` focuses on developing code that can benefit from the :math:`O(N)` scaling of the ``quasisep`` solver.
Thus, it is highly recommended to use the kernels provided in the ``eztaox.kernels.quasisep`` module,
since those are the best tested. For completeness and to support a broader class of kernels, we also provide an interface
for working with kernels that follow :math:`O(N^3)` scaling. This part of ``EzTaoX`` is still **under development**. 

GP (quasisep) Kernels Supported
---------------------------------------------------------------

Below, we list the GP kernels that can be evaluated using the ``quasisep`` solver in ``tinygp``.
You can import them from the ``eztaox.kernels.quasisep`` module.
Please see `here <https://tinygp.readthedocs.io/en/stable/api/kernels.quasisep.html>`__ for definitions of these kernels.

- **Exp**: The exponential kernel---equivalent to the damped random walk (DRW) kernel.
- **Cosine**: The cosine kernel.
- **Matern32/Matern52**: The Matern32/Matern52 kernel.
- **SHO**: The noise-driven simple harmonic oscillator kernel.
- **Celerite**: The base kernel of the `celerite` algorithm. This kernel is not recommended for direct use.
- **Lorentzian**: The Lorentzian kernel. It is the product of an `Exp` kernel and a `Cosine` kernel, whose scales are correlated. See the doc string for the definition.
- **CARMA**: The continuous-time auto-regressive moving-average (CARMA) kernel. In the current version, CARMA processes more complex than CARMA(2,1) might cause bugs.
- **MultibandLowRank**: A multiband kernel implementing a low-rank Kronecker covariance structure.
- **LaguerreSeries**: A Laguerre series approximation of a stationary kernel.

**Note:** You can build new kernels by combining kernels from the list above, and the resulting kernel will still work within the ``EzTaoX`` framework.


Tutorials
---------------------------------------------------------------

A preliminary list of tutorials to help users getting started:

* :doc:`Damped Random Walk <notebooks/01_DRW>`: Demonstrating fitting light curves to a damped random walk (DRW) model.
* :doc:`Damped Harmonic Oscillator <notebooks/02_DHO>`: Demonstrating fitting higher-order CARMA models beyond the DRW. 
  In this notebook, we are fitting the CARMA(2,1) process, which is also known as the damped harmonic oscillator (DHO) process.
* :doc:`Multiband Fitting <notebooks/03_MultibandFitting>`: Demonstrating fitting multi-band light curves to a DRW model.  
* :doc:`Simulating Multi-band Light Curves <notebooks/04_MultibandSimulation>`: Demonstrating how to simulate full, randomly sampled, and user-defined multi-band observing cadences from a shared latent GP.


These notebooks are also available for live experimentation in a binder instance.

.. image:: https://mybinder.org/badge_logo.svg
  :target: https://mybinder.org/v2/gh/LSST-AGN-Variability/EzTaoX/HEAD?urlpath=/lab/tree/docs/notebooks

.. toctree::
   :hidden:

   Home page <self>
   Notebooks <notebooks>
   API Reference <autoapi/index>
   About <about>
   Changelog <changelog>
   

.. [DFM17] Daniel Foreman-Mackey, Eric Agol, Sivaram Ambikasaran, Ruth Angus; 
   *Fast and scalable Gaussian process modeling with applications to astronomical time series*, 
   https://arxiv.org/abs/1703.09710

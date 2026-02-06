"""
A module of light curve models, which are the interface for modeling uni/multi-band
light curves using Gaussian Processes (GPs).
"""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpyro
import tinygp.kernels.quasisep as tkq
from numpy.typing import NDArray
from tinygp import GaussianProcess
from tinygp.helpers import JAXArray

from eztaox.kernels import quasisep


class MultiVarModel(eqx.Module):
    """
    An interface for modeling multivariate/mutli-band time series using GPs.

    This interface only takes GP kernels that can be evaluated using the
    scalable method of `DFM+17 <https://arxiv.org/abs/1703.09710>`. This
    interface allows fitting for a parameterized mean function of the time series,
    additional variance to the measurement uncertainty, and time delays between each
    uni-variate/single-band time series.

    Args:
        X (JAXArray|NDArray): Input data containing time and band indices as a tuple.
        y (JAXArray|NDArray): Observed data values.
        yerr (JAXArray|NDArray): Observational uncertainties.
        base_kernel (Quasisep): A GP kernel from the kernels.quasisep module.
        nBand (int): An integer number of bands in the input light curve.
        multiband_kernel(Quasisep, optional): A multiband kernel specifying the
            cross-band covariance, defaults to kernels.quasisep.MultibandLowRank.
        mean_func(Callable, optional): A callable mean function for the GP, defaults to
            None.
        amp_scale_func(Callable, optional): A callable amplitude scaling function,
            defaults to None.
        lag_func(Callable, optional): A callable function for time delays between bands,
            defaults to None.
        **kwargs: Additional keyword arguments.

            - `zero_mean` (bool): If True, assumes zero-mean GP. Defaults to True.
            - `has_jitter` (bool): If True, assumes the input observational erros
              are underestimated. Defaults to False.
            - `has_lag` (bool): If True, assumes time delays between time series in
              each band. Defaults to False.

    Raises:
        TypeError: If base_kernel is not one from the kernels.quasisep module.
    """

    X: tuple[JAXArray, JAXArray]
    y: JAXArray
    diag: JAXArray
    base_kernel_def: Callable
    multiband_kernel: tkq.Wrapper
    nBand: int
    mean_func: Callable | None
    amp_scale_func: Callable | None
    lag_func: Callable | None
    zero_mean: bool
    has_jitter: bool
    has_lag: bool

    def __init__(
        self,
        X: tuple[JAXArray | NDArray, JAXArray | NDArray],
        y: JAXArray | NDArray,
        yerr: JAXArray | NDArray,
        base_kernel: quasisep.Quasisep,
        nBand: int,
        multiband_kernel: tkq.Wrapper | None = quasisep.MultibandLowRank,
        mean_func: Callable | None = None,
        amp_scale_func: Callable | None = None,
        lag_func: Callable | None = None,
        **kwargs,
    ) -> None:
        if not isinstance(base_kernel, quasisep.Quasisep):
            raise TypeError("This model only takes quasiseperable kernels.")

        # format inputs
        t = jnp.asarray(X[0])
        inds = jnp.argsort(t)
        band = jnp.asarray(X[1], dtype=int)
        y = jnp.asarray(y)
        yerr = jnp.asarray(yerr)

        # assign attributes
        self.X = (t[inds], band[inds])
        self.diag = (yerr**2)[inds]
        self.y = y[inds]
        self.base_kernel_def = jax.flatten_util.ravel_pytree(base_kernel)[1]
        self.nBand = nBand

        # assign callables/classes
        self.multiband_kernel = multiband_kernel
        self.mean_func = mean_func
        self.amp_scale_func = amp_scale_func
        self.lag_func = lag_func

        # assign other attributes
        self.zero_mean = kwargs.get("zero_mean", True)
        self.has_jitter = kwargs.get("has_jitter", False)
        self.has_lag = kwargs.get("has_lag", False)

    def get_mean(
        self, zero_mean: bool, params: dict[str, JAXArray], X: JAXArray
    ) -> JAXArray:
        """Return the mean of the GP."""
        if zero_mean is True:
            mean = 0.0
        elif self.mean_func is not None:
            mean = self.mean_func(params, X)
        else:
            mean = self._default_mean_func(params, X)
        return mean

    def get_amp_scale(self, params: dict[str, JAXArray]) -> JAXArray:
        """Return the ampltiude of GP in each individaul band."""
        if self.amp_scale_func is not None:
            return self.amp_scale_func(params)
        return self._default_amp_scale_func(params)

    def lag_transform(
        self, has_lag: bool, params: dict[str, JAXArray], X: JAXArray
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
        """Shift the time axis by the lag in each band."""
        if has_lag is False:
            lags = jnp.zeros(self.nBand)
        elif self.lag_func is not None:
            lags = self.lag_func(params)
        else:
            lags = self._default_lag_func(params)

        t, band = X
        new_t = t - lags[band]
        inds = jnp.argsort(new_t)
        return (new_t, band), inds

    def log_prior(self, params: dict[str, JAXArray]) -> JAXArray:
        """Calculate the log prior of the input parameters.

        Args:
            params (dict[str, JAXArray]): Model parameters.

        Returns:
            JAXArray: Log prior of the input parameters.
        """
        # Assuming a Gaussian prior for demonstration purposes
        log_prior = 0.0
        return log_prior

    @eqx.filter_jit
    def log_prob(self, params: dict[str, JAXArray]) -> JAXArray:
        """Calculate the log probability of the input parameters.

        Args:
            params (dict[str, JAXArray]): Model parameters.

        Returns:
            JAXArray: Log probability of the input parameters.
        """
        gp, inds = self._build_gp(params)
        return gp.log_probability(y=self.y[inds]) + self.log_prior(params)

    def aic(self, params: dict[str, JAXArray]) -> JAXArray:
        """Calculate the Akaike Information Criterion (AIC) for the model.

        Args:
            params (dict[str, JAXArray]): Model parameters.
        Returns:
            JAXArray: AIC value.
        """
        k = len(jax.flatten_util.ravel_pytree(params)[0])
        gp, inds = self._build_gp(params)
        log_likelihood = gp.log_probability(y=self.y[inds])
        return 2 * k - 2 * log_likelihood

    def bic(self, params: dict[str, JAXArray]) -> JAXArray:
        """Calculate the Bayesian Information Criterion (BIC) for the model.

        Args:
            params (dict[str, JAXArray]): Model parameters.
        Returns:
            JAXArray: BIC value.
        """
        n = self.y.size
        k = len(jax.flatten_util.ravel_pytree(params)[0])
        gp, inds = self._build_gp(params)
        log_likelihood = gp.log_probability(y=self.y[inds])
        return jnp.log(n) * k - 2 * log_likelihood

    def sample(self, params: dict[str, JAXArray]) -> None:
        """A convience function for intergrating with numpyro for MCMC sampling.

        Args:
            params (dict[str, JAXArray]): Model parameters.
        """
        gp, inds = self._build_gp(params)
        numpyro.sample("gp", gp.numpyro_dist(), obs=self.y[inds])

    @eqx.filter_jit
    def pred(
        self, params: dict[str, JAXArray], X: JAXArray
    ) -> tuple[JAXArray, JAXArray]:
        """Make conditional GP prediction.

        Args:
            params (dict[str, JAXArray]): A dictionary containing model parameters.
            X (JAXArray): The time and band information for creating the conditional GP
                prediction.

        Returns:
            tuple[JAXArray, JAXArray]: A tuple of the mean GP prediction and its
            uncertainty (square root of the predicted variance).
        """
        # transform time axis
        new_X, _ = self.lag_transform(self.has_lag, params, X)

        # build gp, cond
        gp, inds = self._build_gp(params)
        _, cond = gp.condition(self.y[inds], new_X)

        return cond.loc, jnp.sqrt(cond.variance)

    def _default_mean_func(self, params: dict[str, JAXArray], X: JAXArray) -> JAXArray:
        return jnp.atleast_1d(params["mean"])[X[1]]

    def _default_amp_scale_func(self, params: dict[str, JAXArray]) -> JAXArray:
        return jnp.insert(jnp.atleast_1d(params["log_amp_scale"]), 0, 0.0)

    def _default_lag_func(
        self, params: dict[str, JAXArray]
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
        return jnp.insert(jnp.atleast_1d(params["lag"]), 0, 0.0)

    def _build_gp(
        self, params: dict[str, JAXArray]
    ) -> tuple[GaussianProcess, JAXArray]:
        # log amp + mean
        log_amp_scales = self.get_amp_scale(params)
        means = partial(self.get_mean, self.zero_mean, params)

        # time axis transform: t and band are not sorted,
        # inds gives the sorted indices for the new_t
        X, inds = self.lag_transform(self.has_lag, params, self.X)
        t = X[0]
        band = X[1]

        # add jitter to the diagonal
        diags = self.diag
        if self.has_jitter is True:
            diags = (
                self.diag + (jnp.exp(jnp.atleast_1d(params["log_jitter"])) ** 2)[band]
            )

        # def kernel
        new_params = params.copy()
        new_params["amplitudes"] = jnp.exp(log_amp_scales)
        kernel = self.multiband_kernel(
            params=new_params,
            kernel=self.base_kernel_def(jnp.exp(new_params["log_kernel_param"])),
        )

        return (
            GaussianProcess(
                kernel,
                (t[inds], band[inds]),
                diag=diags[inds],
                mean=means,
                assume_sorted=True,
            ),
            inds,
        )


class UniVarModel(MultiVarModel):
    """
    A subclass of MultiVarModel for modeling univariate/single-band time series data.

    Args:
        t (JAXArray|NDArray): Time stamps of the input light curve.
        y (JAXArray|NDArray): Observed data values at the corresponding time stamps.
        yerr (JAXArray|NDArray): Observational uncertainties.
        kernel (Quasisep): A GP kernel from the eztaox.kernels.quasisep module.
        mean_func(Callable, optional): A callable mean function for the GP, defaults to
            None.
        amp_scale_func(Callable, optional): A callable amplitude scaling function,
            defaults to None.
        **kwargs: Additional keyword arguments.

            - `zero_mean` (bool): If True, assumes zero-mean GP. Defaults to True.
            - `has_jitter` (bool): If True, assumes the input observational erros
              are underestimated. Defaults to False.

    Raises:
        TypeError: If kernel is not one from the kernels.quasisep module.
    """

    def __init__(
        self,
        t: JAXArray | NDArray,
        y: JAXArray | NDArray,
        yerr: JAXArray | NDArray,
        kernel: quasisep.Quasisep,
        mean_func: Callable | None = None,
        amp_scale_func: Callable | None = None,
        **kwargs,
    ) -> None:
        """Initialize the UniVarModel with time, observed data, and kernel."""

        inds = jnp.argsort(jnp.asarray(t))
        X = (jnp.asarray(t)[inds], jnp.zeros_like(t, dtype=int))
        y = jnp.asarray(y)[inds]
        yerr = jnp.asarray(yerr)[inds]
        base_kernel = kernel
        nBand = 1
        has_lag = False
        super().__init__(
            X,
            y,
            yerr,
            base_kernel,
            nBand,
            mean_func=mean_func,
            amp_scale_func=amp_scale_func,
            has_lag=has_lag,
            **kwargs,
        )

    def _default_amp_scale_func(self, params: dict[str, JAXArray]) -> JAXArray:
        return jnp.array([0.0])

    def lag_transform(  # noqa: D102
        self, has_lag, params, X
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
        # TODO: Write docstring.
        return self.X, jnp.arange(self.X[0].size)

    def pred(self, params, t) -> tuple[JAXArray, JAXArray]:
        """Make conditional GP prediction.

        Args:
            params (dict[str, JAXArray]): A dictionary containing model parameters.
            t (JAXArray): The time information for creating the conditional GP
                prediction.

        Returns:
            tuple[JAXArray, JAXArray]: A tuple of the mean GP prediction and its
            uncertainty (square root of the predicted variance).
        """
        # build gp, cond
        gp, inds = self._build_gp(params)
        _, cond = gp.condition(self.y[inds], (t, jnp.zeros_like(t, dtype=int)))

        return cond.loc, jnp.sqrt(cond.variance)

"""Light curve models module."""
from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpyro
import tinygp
from numpy.typing import NDArray
from tinygp import GaussianProcess
from tinygp.helpers import JAXArray

from eztaox.kernels import mb_kernel


class MultiVarModel(eqx.Module):
    X: JAXArray
    y: JAXArray = eqx.field(converter=jnp.asarray)
    diag: JAXArray = eqx.field(converter=jnp.asarray)
    kernel_def: Callable
    zero_mean: bool = True
    has_jitter: bool = False
    has_lag: bool = False

    def __init__(
        self,
        X: JAXArray,
        y: JAXArray | NDArray,
        yerr: JAXArray | NDArray,
        kernel: tinygp.kernels.quasisep.Quasisep,
        **kwargs,
    ) -> None:
        self.X = X
        self.diag = yerr**2
        self.y = y
        self.kernel_def = jax.flatten_util.ravel_pytree(kernel)[1]
        self.zero_mean = kwargs.get("zero_mean", True)
        self.has_jitter = kwargs.get("has_jitter", False)
        self.has_lag = kwargs.get("has_lag", False)

    def lag_transform(
        self, X: JAXArray, has_lag: bool, params: dict[str, JAXArray]
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
        if has_lag is True:
            lags = jnp.insert(jnp.atleast_1d(params["lag"]), 0, 0.0)
        else:
            nBand = params["log_amp_delta"].size + 1
            lags = jnp.zeros(nBand)
        t, band = X
        new_t = t - lags[band]
        inds = jnp.argsort(new_t)
        return (new_t, band), inds

    def amp_transform(self, params: dict[str, JAXArray]) -> JAXArray:
        return jnp.insert(jnp.atleast_1d(params["log_amp_delta"]), 0, 0.0)

    @staticmethod
    def mean_func(
        zero_mean: bool, nBand: int, params: dict[str, JAXArray], X: JAXArray
    ) -> JAXArray:
        if zero_mean is True:
            means = jnp.zeros(nBand)
        else:
            means = jnp.atleast_1d(params["mean"])
        return means[X[1]]

    def _build_gp(
        self, params: dict[str, JAXArray]
    ) -> tuple[GaussianProcess, JAXArray]:
        # log amp + mean
        log_amps = self.amp_transform(params)
        means = partial(
            MultiVarModel.mean_func, self.zero_mean, log_amps.shape[0], params
        )

        # time axis transform: t and band are not sorted,
        # inds gives the sorted indices for the new_t
        X, inds = self.lag_transform(self.X, self.has_lag, params)
        t = X[0]
        band = X[1]

        # add jitter to the diagonal
        if self.has_jitter is True:
            diags = self.diag[inds] + (jnp.exp(params["log_jitter"]) ** 2)[band[inds]]
        else:
            diags = self.diag[inds]

        # def kernel
        kernel = mb_kernel(
            amplitudes=jnp.exp(log_amps),
            kernel=self.kernel_def(jnp.exp(params["log_kernel_param"])),
        )

        return (
            GaussianProcess(
                kernel,
                (t[inds], band[inds]),
                diag=diags,
                mean=means,
                assume_sorted=True,
            ),
            inds,
        )

    @eqx.filter_jit
    def log_prob(self, params: dict[str, JAXArray]) -> JAXArray:
        gp, inds = self._build_gp(params)
        return gp.log_probability(y=self.y[inds])

    def sample(self, params: dict[str, JAXArray]) -> None:
        gp, inds = self._build_gp(params)
        numpyro.sample("gp", gp.numpyro_dist(), obs=self.y[inds])

    @eqx.filter_jit
    def pred(
        self, params: dict[str, JAXArray], X: JAXArray
    ) -> tuple[JAXArray, JAXArray]:
        # transform time axis
        new_X, inds = self.lag_transform(X, self.has_lag, params)

        # build gp, cond
        gp, inds = self._build_gp(params)
        _, cond = gp.condition(self.y[inds], new_X)

        return cond.loc, jnp.sqrt(cond.variance)


class UniVarModel(eqx.Module):
    t: JAXArray = eqx.field(converter=jnp.asarray)
    y: JAXArray = eqx.field(converter=jnp.asarray)
    yerr: JAXArray = eqx.field(converter=jnp.asarray)
    inds: JAXArray = eqx.field(converter=jnp.asarray)
    kernel_def: Callable
    zero_mean: bool = True
    has_jitter: bool = False

    def __init__(
        self,
        t: JAXArray | NDArray,
        y: JAXArray | NDArray,
        yerr: JAXArray | NDArray,
        kernel: tinygp.kernels.quasisep.Quasisep,
        **kwargs,
    ) -> None:
        self.t = t
        self.y = y
        self.yerr = yerr
        self.inds = jnp.argsort(t)
        self.kernel_def = jax.flatten_util.ravel_pytree(kernel)[1]
        self.zero_mean = kwargs.get("zero_mean", True)
        self.has_jitter = kwargs.get("has_jitter", False)

    @staticmethod
    def mean_func(zero_mean, params: dict[str, JAXArray], X: JAXArray) -> JAXArray:
        if zero_mean is True:
            mean = jnp.zeros(())
        else:
            mean = params["mean"]
        return mean

    def _build_gp(self, params: dict[str, JAXArray]) -> GaussianProcess:
        mean = partial(UniVarModel.mean_func, self.zero_mean, params)

        # add jitter to the diagonal
        if self.has_jitter is True:
            diags = self.yerr**2 + jnp.exp(params["log_jitter"]) ** 2
        else:
            diags = self.yerr**2

        # re-create kernel
        kernel = self.kernel_def(jnp.exp(params["log_kernel_param"]))
        return GaussianProcess(
            kernel,
            self.t[self.inds],
            diag=diags[self.inds],
            mean=mean,
            assume_sorted=True,
        )

    @eqx.filter_jit
    def log_prob(self, params: dict[str, JAXArray]) -> JAXArray:
        gp = self._build_gp(params)
        return gp.log_probability(self.y[self.inds])

    def sample(self, params: dict[str, JAXArray]) -> None:
        gp = self._build_gp(params)
        numpyro.sample("gp", gp.numpyro_dist(), obs=self.y[self.inds])

    @eqx.filter_jit
    def pred(
        self, params: dict[str, JAXArray], t: JAXArray | NDArray
    ) -> tuple[JAXArray, JAXArray]:
        _, cond = self._build_gp(params).condition(self.y[self.inds], t)
        return cond.loc, jnp.sqrt(cond.variance)

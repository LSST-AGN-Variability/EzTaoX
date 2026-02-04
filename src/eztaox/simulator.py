"""Simulator module for multi/uni-variate Gaussian Processes."""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import tinygp.kernels.quasisep as tkq
from numpy.typing import NDArray
from tinygp import GaussianProcess
from tinygp.helpers import JAXArray

from eztaox.kernels import quasisep
from eztaox.ts_utils import _get_nearest_idx


class MultiVarSim(eqx.Module):
    """An interface for simulating multivariate/mutli-band time series using GPs.

    This interface only takes GP kernels that can be evaluated using the
    scalable method of `DFM+17 <https://arxiv.org/abs/1703.09710>`_. This interface
    allows specifying a parameterized mean function of the time series, cross-band
    covariance, and time delays between each uni-variate/single-band time series.

    Args:
        base_kernel (Quasisep): A GP kernel from the kernels.quasisep module.
        min_dt (float): Minimum time step for the simulation.
        max_dt (float): Maximum time step (temporal baseline) for the simulation.
        nBand (int): An integer number of bands in the input light curve.
        init_params (dict[str, JAXArray]): Initial parameters for the GP.
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
            - `has_lag` (bool): If True, assumes time delays between time series in
              each band. Defaults to False.
    """

    base_kernel_def: Callable
    multiband_kernel: tkq.Wrapper
    X: tuple[JAXArray, JAXArray]
    init_params: dict[str, JAXArray]
    nBand: int
    mean_func: Callable | None
    amp_scale_func: Callable | None
    lag_func: Callable | None
    zero_mean: bool
    has_lag: bool

    def __init__(
        self,
        base_kernel: quasisep.Quasisep,
        min_dt: float,
        max_dt: float,
        nBand: int,
        init_params: dict[str, JAXArray],
        multiband_kernel: tkq.Wrapper | None = quasisep.MultibandLowRank,
        mean_func: Callable | None = None,
        amp_scale_func: Callable | None = None,
        lag_func: Callable | None = None,
        **kwargs,
    ) -> None:
        # make sim X
        simN = int(max_dt / min_dt) + 1
        ts, bands = [], []
        for i in range(nBand):
            ts.append(jnp.linspace(0, max_dt, simN))
            bands.append(jnp.full_like(ts[i], i, dtype=int))
        t = jnp.concat(ts)
        band = jnp.concat(bands)

        # assign fixed values
        inds = jnp.argsort(t)
        self.X = (t[inds], band[inds])
        self.nBand = nBand
        self.init_params = init_params

        # assign callables/classes
        self.base_kernel_def = jax.flatten_util.ravel_pytree(base_kernel)[1]
        self.multiband_kernel = multiband_kernel
        self.mean_func = mean_func
        self.amp_scale_func = amp_scale_func
        self.lag_func = lag_func

        # assign other attributes
        self.zero_mean = kwargs.get("zero_mean", True)
        self.has_lag = kwargs.get("has_lag", False)

        # compile funcs
        self._build_gp(self.X, self.init_params)

    def full(
        self, key: jax.random.PRNGKey, params: dict[str, JAXArray] | None = None
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
        """Simulate a multivariace GP time series with unifrom time sampling.

        Args:
            key (jax.random.PRNGKey): Random number generator key.
            params (dict[str, JAXArray] | None, optional): Light curve model parames.
                Defaults to None. If None, uses the initial parameters.

        Returns:
            tuple[tuple[JAXArray, JAXArray], JAXArray]: Simulated time series in the
            form of (time, band) and the simulated light curve values.
        """
        params = params if params is not None else self.init_params
        gp, inds = self._build_gp(self.X, params)

        return (self.X[0][inds], self.X[1][inds]), gp.sample(key)

    def random(
        self,
        nRand: int,
        lc_key: jax.random.PRNGKey,
        random_key: jax.random.PRNGKey,
        params: dict[str, JAXArray] | None = None,
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray, JAXArray]:
        """Simulate a multivariace GP time series with random time sampling.

        Args:
            nRand (int): Number of data points in the simulated time series.
            lc_key (jax.random.PRNGKey): Random number generator key for simulating a
                full light curve with uniform time sampling.
            random_key (jax.random.PRNGKey): Random number generator key for selecting
                random data points from the full light curve.
            params (dict[str, JAXArray] | None, optional): Light curve model parames.
                Defaults to None. If None, uses the initial parameters.

        Returns:
            tuple[tuple[JAXArray, JAXArray], JAXArray]: Simulated time series in the
            form of (time, band) and the simulated light curve values.
        """

        # get full light curve
        params = params if params is not None else self.init_params
        full_X, full_y = self.full(lc_key, params)

        # select randomly & return
        rand_inds = jnp.sort(jax.random.permutation(random_key, full_y.size)[:nRand])
        return (
            (full_X[0][rand_inds], full_X[1][rand_inds]),
            full_y[rand_inds],
        )

    def fixed_input(
        self,
        sim_X: tuple[JAXArray | NDArray, JAXArray | NDArray],
        lc_key: jax.random.PRNGKey,
        params: dict[str, JAXArray] | None = None,
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray, JAXArray]:
        """Simulate a multivariace GP time series with fixed input time and band labels.

        Args:
            sim_X (tuple[JAXArray|NDArray, JAXArray|NDArray]): Input time and band.
            lc_key (jax.random.PRNGKey): Random number generator key for simulating a
                full light curve with uniform time sampling.
            params (dict[str, JAXArray] | None, optional): Light curve model parames.
                Defaults to None. If None, uses the initial parameters.

        Returns:
            tuple[tuple[JAXArray, JAXArray], JAXArray]: Simulated time series in the
            form of (time, band) and the simulated light curve values.
        """
        # convert sim_X to JAXArray and ensure band is int
        sim_X = (jnp.asarray(sim_X[0]), jnp.asarray(sim_X[1]).astype(int))

        # get full light curve
        params = params if params is not None else self.init_params
        full_X, full_y = self.full(lc_key, params)

        # get indices for the input sim_X
        ts, bands, ys = [], [], []
        for i in range(self.nBand):
            full_band_mask = full_X[1] == i
            input_band_mask = sim_X[1] == i
            inds = jax.vmap(_get_nearest_idx, in_axes=(None, 0))(
                full_X[0][full_band_mask], sim_X[0][input_band_mask]
            )
            ts.append(full_X[0][full_band_mask][inds])
            bands.append(jnp.full_like(ts[i], i, dtype=int))
            ys.append(full_y[full_band_mask][inds])

        return ((jnp.concat(ts), jnp.concat(bands)), jnp.concat(ys))

    def fixed_input_fast(
        self,
        sim_X: tuple[JAXArray | NDArray, JAXArray | NDArray],
        lc_key: jax.random.PRNGKey,
        params: dict[str, JAXArray] | None = None,
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
        """Simulate a multivariace GP time series with fixed input time and band labels.

        This method is faster than `fixed_input` since it only simulates the GP at the
        input times, rather than simulating a full light curve and selecting points that
        match the input times.

        Args:
            sim_X (tuple[JAXArray|NDArray, JAXArray|NDArray]): Input time and band.
            lc_key (jax.random.PRNGKey): Random number generator key for simulating a
                full light curve with uniform time sampling.
            params (dict[str, JAXArray] | None, optional): Light curve model parames.
                Defaults to None. If None, uses the initial parameters.

        Returns:
            tuple[tuple[JAXArray, JAXArray], JAXArray]: Simulated time series in the
            form of (time, band) and the simulated light curve values.
        """
        # convert sim_X to JAXArray and ensure band is int
        sim_X = (jnp.asarray(sim_X[0]), jnp.asarray(sim_X[1]).astype(int))

        # build gp
        params = params if params is not None else self.init_params
        gp, inds = self._build_gp(sim_X, params)

        return (sim_X[0][inds], sim_X[1][inds]), gp.sample(lc_key)

    @eqx.filter_jit
    def _build_gp(
        self,
        X: tuple[JAXArray, JAXArray],
        params: dict[str, JAXArray],
    ) -> tuple[GaussianProcess, JAXArray]:
        # log amp + mean
        log_amp_scales = self.get_amp_scale(params)
        means = partial(self.get_mean, self.zero_mean, params)

        # time axis transform: new_inds gives the sorted indices for t, band,
        # after the lag transform
        if self.has_lag is False:
            inds = jnp.arange(X[0].size)
            new_t = X[0]
        else:
            new_X, inds = self.lag_transform(self.has_lag, params, X)
            new_t = new_X[0]

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
                (new_t[inds], X[1][inds]),
                mean=means,
                assume_sorted=True,
            ),
            inds,
        )

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

    def _default_mean_func(self, params: dict[str, JAXArray], X: JAXArray) -> JAXArray:
        return jnp.atleast_1d(params["mean"])[X[1]]

    def _default_amp_scale_func(self, params: dict[str, JAXArray]) -> JAXArray:
        return jnp.insert(jnp.atleast_1d(params["log_amp_scale"]), 0, 0.0)

    def _default_lag_func(
        self, params: dict[str, JAXArray]
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
        return jnp.insert(jnp.atleast_1d(params["lag"]), 0, 0.0)


class UniVarSim(MultiVarSim):
    """An interface for simulating univariate/single-band GP time series.

    Args:
        base_kernel (Quasisep): A GP kernel from the kernels.quasisep module.
        min_dt (float): Minimum time step for the simulation.
        max_dt (float): Maximum time step (temporal baseline) for the simulation.
        init_params (dict[str, JAXArray]): Initial parameters for the GP.
        mean_func(Callable, optional): A callable mean function for the GP, defaults to
            None.
        amp_scale_func(Callable, optional): A callable amplitude scaling function,
            defaults to None.
        **kwargs: Additional keyword arguments.

            * `zero_mean` (bool): If True, assumes zero-mean GP. Defaults to True.
    """

    def __init__(
        self,
        base_kernel: quasisep.Quasisep,
        min_dt: float,
        max_dt: float,
        init_params: dict[str, JAXArray],
        mean_func: Callable | None = None,
        amp_scale_func: Callable | None = None,
        **kwargs,
    ) -> None:
        """Initialize the UniVarSim with time, observed data, and kernel."""

        # univar specific attributes
        nBand = 1
        has_lag = False

        # call super
        super().__init__(
            base_kernel,
            min_dt,
            max_dt,
            nBand,
            init_params,
            mean_func=mean_func,
            amp_scale_func=amp_scale_func,
            has_lag=has_lag,
            **kwargs,
        )

    def _default_amp_scale_func(self, params: dict[str, JAXArray]) -> JAXArray:
        return jnp.array([0.0])

    def full(
        self, key: jax.random.PRNGKey, params: dict[str, JAXArray] | None = None
    ) -> tuple[JAXArray, JAXArray]:
        """Simulate a univariate GP time series with unifrom time sampling.

        Args:
            key (jax.random.PRNGKey): Random number generator key.
            params (dict[str, JAXArray] | None, optional): Light curve model parames.
                Defaults to None. If None, uses the initial parameters.

        Returns:
            tuple[JAXArray, JAXArray]: Simulated time series in the form of (time,
            light curve values).
        """
        params = params if params is not None else self.init_params
        mb_X, mb_y = super().full(key, params)
        return mb_X[0], mb_y

    def random(
        self,
        nRand: int,
        lc_key: jax.random.PRNGKey,
        random_key: jax.random.PRNGKey,
        params: dict[str, JAXArray] | None = None,
    ) -> tuple[JAXArray, JAXArray, JAXArray]:
        """Simulate a univariate GP time series with random time sampling.

        Args:
            nRand (int): Number of data points in the simulated time series.
            lc_key (jax.random.PRNGKey): Random number generator key for simulating a
                full light curve with uniform time sampling.
            random_key (jax.random.PRNGKey): Random number generator key for selecting
                random data points from the full light curve.
            params (dict[str, JAXArray] | None, optional): Light curve model parames.
                Defaults to None. If None, uses the initial parameters.

        Returns:
            tuple[JAXArray, JAXArray]: Simulated time series in the form of (time,
            light curve values).
        """

        # get full light curve
        params = params if params is not None else self.init_params
        full_t, full_y = self.full(lc_key, params)

        # select randomly & return
        rand_inds = jnp.sort(jax.random.permutation(random_key, full_y.size)[:nRand])
        return full_t[rand_inds], full_y[rand_inds]

    def fixed_input(
        self,
        sim_t: JAXArray | NDArray,
        lc_key: jax.random.PRNGKey,
        params: dict[str, JAXArray] | None = None,
    ) -> tuple[JAXArray, JAXArray]:
        """Simulate a univariate GP time series with fixed input time.

        Args:
            sim_t (JAXArray | NDArray): Input time for the simulation.
            lc_key (jax.random.PRNGKey): Random number generator key for simulating a
                full light curve with uniform time sampling.
            params (dict[str, JAXArray] | None, optional): Light curve model parames.
                Defaults to None. If None, uses the initial parameters.

        Returns:
            tuple[JAXArray, JAXArray]: Simulated time series in the form of (time,
            light curve values).
        """
        params = params if params is not None else self.init_params
        full_t, full_y = self.full(lc_key, params)

        # get indices for the input sim_t
        inds = jax.vmap(_get_nearest_idx, in_axes=(None, 0))(full_t, jnp.asarray(sim_t))
        return full_t[inds], full_y[inds]

    def fixed_input_fast(
        self,
        sim_t: JAXArray | NDArray,
        lc_key: jax.random.PRNGKey,
        params: dict[str, JAXArray] | None = None,
    ) -> tuple[JAXArray, JAXArray]:
        """Simulate a univariate GP time series with fixed input time.

        This method is faster than `fixed_input` since it only simulates the GP at the
        input times, rather than simulating a full light curve and selecting points that
        match the input times.

        Args:
            sim_t (JAXArray | NDArray): Input time for the simulation.
            lc_key (jax.random.PRNGKey): Random number generator key for simulating a
                full light curve with uniform time sampling.
            params (dict[str, JAXArray] | None, optional): Light curve model parames.
                Defaults to None. If None, uses the initial parameters.

        Returns:
            tuple[JAXArray, JAXArray]: Simulated time series in the form of (time,
            light curve values).
        """
        params = params if params is not None else self.init_params
        sim_X = (jnp.asarray(sim_t), jnp.zeros_like(sim_t))
        mb_X, mb_y = super().fixed_input_fast(sim_X, lc_key, params)
        return mb_X[0], mb_y

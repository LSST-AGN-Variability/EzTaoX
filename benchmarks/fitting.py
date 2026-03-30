"""Benchmarks for EzTaoX kernel fitting"""

import jax
import jax.numpy as jnp
import jax.random as jr
from tinygp.helpers import JAXArray

from eztaox.kernels import quasisep as ekq
from eztaox.models import MultiVarModel, UniVarModel
from eztaox.simulator import UniVarSim
from eztaox.ts_utils import add_noise

DRW_PARAMS = {"tau": 100.0, "sigma": 0.1}
NBANDS = 2


class KernelUniVarSuite:
    """Timing benchmarks for various Univariate kernels"""

    # Size of lightcurve `n`
    params = [50, 200, 500, 2_000]
    repeat = 5
    sample_time = 0.1

    def setup(self, n) -> None:
        t, y, yerr = generate_drw_univar(n)

        # Exp kernel
        exp_kernel = ekq.Exp(scale=10.0, sigma=0.1)
        self.exp_params = {
            "log_kernel_param": jnp.log(jax.flatten_util.ravel_pytree(exp_kernel)[0]),
            "mean": 0.0,
        }
        self.exp_model = UniVarModel(t, y, yerr, exp_kernel, zeromean=False)

        # Matern32 kernel
        m32_kernel = ekq.Matern32(scale=10.0, sigma=0.1)
        self.m32_params = self.exp_params.copy()
        self.m32_model = UniVarModel(t, y, yerr, m32_kernel, zeromean=False)

        # Matern52 kernel
        m52_kernel = ekq.Matern52(scale=10.0, sigma=0.1)
        self.m52_params = self.exp_params.copy()
        self.m52_model = UniVarModel(t, y, yerr, m52_kernel, zeromean=False)

        # Precompile log probability functions
        self.exp_log_prob = _precompile_log_prob(self.exp_model, self.exp_params)
        self.m32_log_prob = _precompile_log_prob(self.m32_model, self.m32_params)
        self.m52_log_prob = _precompile_log_prob(self.m52_model, self.m52_params)

    def time_run_exp_logp(self, _):
        self.exp_log_prob(self.exp_params).block_until_ready()

    def time_run_m32_logp(self, _):
        self.m32_log_prob(self.m32_params).block_until_ready()

    def time_run_m52_logp(self, _):
        self.m52_log_prob(self.m52_params).block_until_ready()


class KernelMultVarSuite:
    """Timing benchmarks for various Multivariate kernels"""

    # Size of lightcurve `n`
    params = [50, 200, 500, 2_000]
    repeat = 5
    sample_time = 0.2

    def setup(self, n) -> None:
        X, y, yerr = generate_drw_multivar(n)
        rand_lag = jr.uniform(jr.PRNGKey(0), minval=0.0, maxval=10.0)

        # Exp kernel
        exp_kernel = ekq.Exp(scale=10.0, sigma=0.1)
        self.exp_params = {
            "log_kernel_param": jnp.log(jax.flatten_util.ravel_pytree(exp_kernel)[0]),
            "log_amp_scale": jnp.log(1.0),
            "lag": rand_lag,
            "mean": 0.0,
        }
        self.exp_model = MultiVarModel(
            X, y, yerr, exp_kernel, NBANDS, zeromean=False, has_lag=True
        )

        # Matern32 kernel
        m32_kernel = ekq.Matern32(scale=10.0, sigma=0.1)
        self.m32_params = self.exp_params.copy()

        self.m32_model = MultiVarModel(
            X, y, yerr, m32_kernel, NBANDS, zeromean=False, has_lag=True
        )

        # Matern52 kernel
        m52_kernel = ekq.Matern52(scale=10.0, sigma=0.1)
        self.m52_params = self.exp_params.copy()
        self.m52_model = MultiVarModel(
            X, y, yerr, m52_kernel, NBANDS, zeromean=False, has_lag=True
        )

        # Precompile log probability functions
        self.exp_log_prob = _precompile_log_prob(self.exp_model, self.exp_params)
        self.m32_log_prob = _precompile_log_prob(self.m32_model, self.m32_params)
        self.m52_log_prob = _precompile_log_prob(self.m52_model, self.m52_params)

    def time_run_exp_logp(self, _):
        self.exp_log_prob(self.exp_params).block_until_ready()

    def time_run_m32_logp(self, _):
        self.m32_log_prob(self.m32_params).block_until_ready()

    def time_run_m52_logp(self, _):
        self.m52_log_prob(self.m52_params).block_until_ready()


def generate_drw_univar(n) -> tuple[JAXArray, JAXArray, JAXArray]:
    """Generate single band light curve of size `n`"""
    log_kernel_param = jnp.stack(
        [jnp.log(DRW_PARAMS["tau"]), jnp.log(DRW_PARAMS["sigma"])]
    )
    t = jnp.arange(0.0, n, 1.0)
    s = UniVarSim(
        ekq.Exp(*jnp.exp(log_kernel_param)),
        min_dt=1.0,
        max_dt=float(t[-1]),
        init_params={"log_kernel_param": log_kernel_param},
        zero_mean=True,
    )

    lc_key, noise_key = jax.random.PRNGKey(11), jax.random.PRNGKey(12)
    t, y = s.fixed_input(t, lc_key)
    yerr = jnp.ones_like(y) * 0.01
    return t, add_noise(y, yerr, noise_key), yerr


def generate_drw_multivar(
    n, num_bands=NBANDS
) -> tuple[tuple[JAXArray, JAXArray], JAXArray, JAXArray]:
    """Generate multiband light curve of size `n` in each band"""

    t, y, yerr = generate_drw_univar(n)
    band = jr.choice(jr.PRNGKey(1), a=num_bands, shape=t.shape, replace=True)
    return (t, band), y, yerr


def _precompile_log_prob(model, params):
    @jax.jit
    def log_prob(params):
        return model.log_prob(params)

    log_prob(params).block_until_ready()
    return log_prob

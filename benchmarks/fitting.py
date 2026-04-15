"""Benchmarks for EzTaoX kernel fitting"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import optax
from tinygp.helpers import JAXArray

from eztaox.fitter import random_search
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


class KernelUniVarPrecompileSuite:
    """Timing benchmarks for univariate precompile cost at a fixed size."""

    params = [2_000]
    repeat = 10
    sample_time = 0.1

    def setup(self, n) -> None:
        self.t, self.y, self.yerr = generate_drw_univar(n)

    def time_precompile_exp_gp(self, _):
        model, params = _build_univar_model_and_params(
            ekq.Exp, self.t, self.y, self.yerr
        )
        _precompile_log_prob(model, params)

    def time_precompile_m32_gp(self, _):
        model, params = _build_univar_model_and_params(
            ekq.Matern32, self.t, self.y, self.yerr
        )
        _precompile_log_prob(model, params)

    def time_precompile_m52_gp(self, _):
        model, params = _build_univar_model_and_params(
            ekq.Matern52, self.t, self.y, self.yerr
        )
        _precompile_log_prob(model, params)


class KernelMultVarPrecompileSuite:
    """Timing benchmarks for multivariate precompile cost at a fixed size."""

    params = [2_000]
    repeat = 10
    sample_time = 0.1

    def setup(self, n) -> None:
        self.X, self.y, self.yerr = generate_drw_multivar(n)
        self.rand_lag = jr.uniform(jr.PRNGKey(0), minval=0.0, maxval=10.0)

    def time_precompile_exp_gp(self, _):
        model, params = _build_multivar_model_and_params(
            ekq.Exp, self.X, self.y, self.yerr, self.rand_lag
        )
        _precompile_log_prob(model, params)

    def time_precompile_m32_gp(self, _):
        model, params = _build_multivar_model_and_params(
            ekq.Matern32, self.X, self.y, self.yerr, self.rand_lag
        )
        _precompile_log_prob(model, params)

    def time_precompile_m52_gp(self, _):
        model, params = _build_multivar_model_and_params(
            ekq.Matern52, self.X, self.y, self.yerr, self.rand_lag
        )
        _precompile_log_prob(model, params)


class RandomSearchSuite:
    """Benchmark univariate random_search across representative batch sizes."""

    params = [1000]
    param_names = ["batch_size"]
    repeat = 5
    sample_time = 0.1
    timeout = 120

    def setup(self, batch_size) -> None:
        del batch_size
        self.x = jnp.linspace(0.0, 2.0 * jnp.pi, 1000)
        self.y = jnp.sin(self.x)
        self.yerr = jnp.ones_like(self.x) * 0.05
        self.kernel = ekq.Exp(scale=1.5, sigma=0.8)
        self.model = UniVarModel(
            self.x,
            self.y,
            self.yerr,
            self.kernel,
            zero_mean=False,
            has_jitter=True,
        )
        self.init_sampler = _univar_random_search_init_sampler
        self.fit_key = jr.PRNGKey(0)

    def time_random_search(self, batch_size):
        best_param, log_likelihood = _run_random_search_benchmark(
            self.model,
            self.init_sampler,
            self.fit_key,
            nSample=2000,
            nBest=5,
            batch_size=batch_size,
        )
        _block_until_ready(best_param, log_likelihood)

    def peakmem_random_search(self, batch_size):
        best_param, log_likelihood = _run_random_search_benchmark(
            self.model,
            self.init_sampler,
            self.fit_key,
            nSample=2000,
            nBest=5,
            batch_size=batch_size,
        )
        _block_until_ready(best_param, log_likelihood)


class RandomSearchMultiVarSuite(RandomSearchSuite):
    """Benchmark multivariate random_search across representative batch sizes."""

    def setup(self, batch_size) -> None:
        del batch_size
        self.X, self.y, self.yerr = generate_drw_multivar(1000)
        self.kernel = ekq.Exp(scale=100.0, sigma=0.1)
        self.model = MultiVarModel(
            self.X,
            self.y,
            self.yerr,
            self.kernel,
            NBANDS,
            zero_mean=True,
            has_lag=True,
        )
        self.init_sampler = _multivar_random_search_init_sampler
        self.fit_key = jr.PRNGKey(0)


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


def _build_univar_model_and_params(kernel_cls, t, y, yerr):
    kernel = kernel_cls(scale=10.0, sigma=0.1)
    params = {
        "log_kernel_param": jnp.log(jax.flatten_util.ravel_pytree(kernel)[0]),
        "mean": 0.0,
    }
    model = UniVarModel(t, y, yerr, kernel, zero_mean=False)
    return model, params


def _build_multivar_model_and_params(kernel_cls, X, y, yerr, lag):
    kernel = kernel_cls(scale=10.0, sigma=0.1)
    params = {
        "log_kernel_param": jnp.log(jax.flatten_util.ravel_pytree(kernel)[0]),
        "log_amp_scale": jnp.log(1.0),
        "lag": lag,
        "mean": 0.0,
    }
    model = MultiVarModel(X, y, yerr, kernel, NBANDS, zero_mean=False, has_lag=True)
    return model, params


def _precompile_log_prob(model, params):
    @jax.jit
    def log_prob(params):
        return model.log_prob(params)

    log_prob(params).block_until_ready()
    return log_prob


def _univar_random_search_init_sampler():
    log_drw_scale = numpyro.sample(
        "drw_scale",
        numpyro.distributions.Uniform(jnp.log(0.1), jnp.log(10.0)),
    )
    log_drw_sigma = numpyro.sample(
        "drw_sigma",
        numpyro.distributions.Uniform(jnp.log(0.01), jnp.log(2.0)),
    )
    return {
        "log_kernel_param": jnp.stack([log_drw_scale, log_drw_sigma]),
        "mean": numpyro.sample(
            "mean", numpyro.distributions.Normal(loc=0.0, scale=0.1)
        ),
        "log_jitter": numpyro.sample(
            "log_jitter",
            numpyro.distributions.Uniform(-6.0, -2.0),
        ),
    }


def _multivar_random_search_init_sampler():
    log_drw_scale = numpyro.sample(
        "drw_scale",
        numpyro.distributions.Uniform(jnp.log(0.1), jnp.log(10.0)),
    )
    log_drw_sigma = numpyro.sample(
        "drw_sigma",
        numpyro.distributions.Uniform(jnp.log(0.01), jnp.log(2.0)),
    )
    return {
        "log_kernel_param": jnp.stack([log_drw_scale, log_drw_sigma]),
        "log_amp_scale": numpyro.sample(
            "log_amp_scale", numpyro.distributions.Uniform(-2.0, 2.0)
        ),
        "mean": numpyro.sample(
            "mean", numpyro.distributions.Normal(loc=0.0, scale=0.1)
        ),
        "lag": numpyro.sample("lag", numpyro.distributions.Uniform(-10.0, 10.0)),
    }


def _run_random_search_benchmark(
    model, init_sampler, fit_key, *, nSample, nBest, batch_size
):
    return random_search(
        model,
        init_sampler,
        fit_key,
        nSample=nSample,
        nBest=nBest,
        batch_size=batch_size,
        optimizer=optax.adam(1e-2),
        n_opt_step=1000,
    )


def _block_until_ready(best_param, log_likelihood):
    jax.tree_util.tree_map(
        lambda value: (
            value.block_until_ready() if hasattr(value, "block_until_ready") else value
        ),
        best_param,
    )
    log_likelihood.block_until_ready()

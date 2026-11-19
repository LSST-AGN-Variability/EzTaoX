"""Benchmarks for EzTaoX kernel fitting"""

import contextlib
import functools

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

# Run JAX computations inline instead of handing them to the CPU client's worker
# thread. The asynchronous dispatch adds a fixed, noisy wake-up latency (tens of
# microseconds on macOS) to every blocking call, which would dominate the small
# log-likelihood benchmarks without saying anything about eztaox itself.
with contextlib.suppress(AttributeError, ValueError, KeyError):
    # older JAX releases have no such option
    jax.config.update("jax_cpu_enable_async_dispatch", False)

DRW_PARAMS = {"tau": 100.0, "sigma": 0.1}
NBANDS = 2
CARMA_BENCHMARK_ALPHA = (0.0002, 0.05)
CARMA_BENCHMARK_BETA = (0.0006, 0.03)


# The log-likelihood suites keep their models in a module-level cache instead of
# a ``setup`` hook. asv_runner >= 0.3 re-runs ``setup`` before every timed call,
# so with a hook each sample would be a single cold call made right after
# seconds of JIT compilation, which inflates these microsecond-scale timings
# 3-4x and makes them noisy. Without a hook asv batches ``number`` warm calls per
# sample; the one-off build and compile lands in asv's warmup phase.
_LOGP_NUMBER = 200


@functools.cache
def _univar_logp_state(n) -> dict[str, tuple[dict, callable]]:
    """Build and precompile the univariate models once per light-curve size."""
    t, y, yerr = generate_drw_univar(n)
    kernels = {
        "exp": ekq.Exp(scale=10.0, sigma=0.1),
        "m32": ekq.Matern32(scale=10.0, sigma=0.1),
        "m52": ekq.Matern52(scale=10.0, sigma=0.1),
        "carma": ekq.CARMA(
            alpha=jnp.asarray(CARMA_BENCHMARK_ALPHA),
            beta=jnp.asarray(CARMA_BENCHMARK_BETA),
        ),
    }
    state = {}
    for name, kernel in kernels.items():
        params = {
            "log_kernel_param": jnp.log(jax.flatten_util.ravel_pytree(kernel)[0]),
            "mean": 0.0,
        }
        model = UniVarModel(t, y, yerr, kernel, zero_mean=False)
        state[name] = (params, _precompile_log_prob(model, params))
    return state


@functools.cache
def _multivar_logp_state(n) -> dict[str, tuple[dict, callable]]:
    """Build and precompile the multivariate models once per light-curve size."""
    X, y, yerr = generate_drw_multivar(n)
    rand_lag = jr.uniform(jr.PRNGKey(0), minval=0.0, maxval=10.0)
    kernels = {
        "exp": ekq.Exp(scale=10.0, sigma=0.1),
        "m32": ekq.Matern32(scale=10.0, sigma=0.1),
        "m52": ekq.Matern52(scale=10.0, sigma=0.1),
    }
    state = {}
    for name, kernel in kernels.items():
        params = {
            "log_kernel_param": jnp.log(jax.flatten_util.ravel_pytree(kernel)[0]),
            "log_amp_scale": jnp.log(1.0),
            "lag": rand_lag,
            "mean": 0.0,
        }
        model = MultiVarModel(X, y, yerr, kernel, NBANDS, zero_mean=False, has_lag=True)
        state[name] = (params, _precompile_log_prob(model, params))
    return state


class KernelUniVarSuite:
    """Timing benchmarks for various Univariate kernels"""

    # Size of lightcurve `n`
    params = [50, 200, 500, 2_000]
    repeat = 5
    number = _LOGP_NUMBER

    def time_run_exp_logp(self, n):
        params, log_prob = _univar_logp_state(n)["exp"]
        log_prob(params).block_until_ready()

    def time_run_m32_logp(self, n):
        params, log_prob = _univar_logp_state(n)["m32"]
        log_prob(params).block_until_ready()

    def time_run_m52_logp(self, n):
        params, log_prob = _univar_logp_state(n)["m52"]
        log_prob(params).block_until_ready()

    def time_run_carma_logp(self, n):
        params, log_prob = _univar_logp_state(n)["carma"]
        log_prob(params).block_until_ready()


class KernelMultiVarSuite:
    """Timing benchmarks for various Multivariate kernels"""

    # Size of lightcurve `n`
    params = [50, 200, 500, 2_000]
    repeat = 5
    number = _LOGP_NUMBER

    def time_run_exp_logp(self, n):
        params, log_prob = _multivar_logp_state(n)["exp"]
        log_prob(params).block_until_ready()

    def time_run_m32_logp(self, n):
        params, log_prob = _multivar_logp_state(n)["m32"]
        log_prob(params).block_until_ready()

    def time_run_m52_logp(self, n):
        params, log_prob = _multivar_logp_state(n)["m52"]
        log_prob(params).block_until_ready()


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

    def time_precompile_carma_gp(self, _):
        model, params = _build_univar_model_and_params(
            ekq.CARMA,
            self.t,
            self.y,
            self.yerr,
            alpha=jnp.asarray(CARMA_BENCHMARK_ALPHA),
            beta=jnp.asarray(CARMA_BENCHMARK_BETA),
        )
        _precompile_log_prob(model, params)


class KernelMultiVarPrecompileSuite:
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


class RandomSearchUniVarSuite:
    """Benchmark univariate random_search."""

    params = [1000]
    param_names = ["batch_size"]
    repeat = 5
    sample_time = 0.1
    timeout = 120

    def setup(self, batch_size) -> None:
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
        )
        self.init_sampler = _init_sampler
        self.fit_key = jr.PRNGKey(0)

    def time_random_search(self, batch_size):
        best_param, log_likelihood = _run_random_search_benchmark(
            self.model,
            self.init_sampler,
            self.fit_key,
            n_sample=2000,
            n_best=5,
            batch_size=batch_size,
        )
        _block_until_ready(best_param, log_likelihood)

    def peakmem_random_search(self, batch_size):
        best_param, log_likelihood = _run_random_search_benchmark(
            self.model,
            self.init_sampler,
            self.fit_key,
            n_sample=2000,
            n_best=5,
            batch_size=batch_size,
        )
        _block_until_ready(best_param, log_likelihood)


class RandomSearchMultiVarSuite(RandomSearchUniVarSuite):
    """Benchmark multivariate random_search."""

    def setup(self, batch_size) -> None:
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
        self.init_sampler = _init_sampler
        self.fit_key = jr.PRNGKey(0)


class MCMCUniSuite:
    """Peak-memory benchmark for univariate MCMC."""

    timeout = 300
    repeat = 5

    def setup(self) -> None:
        self.t, self.y, self.yerr = generate_drw_univar(1000)
        self.kernel = ekq.Exp(scale=100.0, sigma=1.0)
        self.model = UniVarModel(
            self.t,
            self.y,
            self.yerr,
            self.kernel,
            zero_mean=False,
        )
        self.numpyro_model = _make_numpyro_model(_init_sampler)
        self.mcmc_key = jr.PRNGKey(0)

    def peakmem_mcmc(self):
        samples = _run_mcmc_benchmark(
            self.model,
            self.numpyro_model,
            self.mcmc_key,
            num_warmup=1000,
            num_samples=2000,
        )
        _block_until_ready(samples)


class MCMCMultiVarSuite(MCMCUniSuite):
    """Peak-memory benchmark for multivariate MCMC."""

    def setup(self) -> None:
        self.X, self.y, self.yerr = generate_drw_multivar(1000)
        self.kernel = ekq.Exp(scale=100.0, sigma=1.0)
        self.model = MultiVarModel(
            self.X,
            self.y,
            self.yerr,
            self.kernel,
            NBANDS,
            zero_mean=True,
            has_lag=True,
        )
        self.numpyro_model = _make_numpyro_model(_init_sampler)
        self.mcmc_key = jr.PRNGKey(0)


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


def _build_univar_model_and_params(kernel_cls, t, y, yerr, **kernel_kwargs):
    if kernel_kwargs:
        kernel = kernel_cls(**kernel_kwargs)
    else:
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


def _init_sampler():
    log_drw_scale = numpyro.sample(
        "drw_scale",
        numpyro.distributions.Uniform(jnp.log(0.1), jnp.log(10.0)),
    )
    log_drw_sigma = numpyro.sample(
        "drw_sigma",
        numpyro.distributions.Uniform(jnp.log(0.01), jnp.log(2.0)),
    )
    log_kernel_param = jnp.stack([log_drw_scale, log_drw_sigma])
    numpyro.deterministic("log_kernel_param", log_kernel_param)
    return {
        "log_kernel_param": log_kernel_param,
        "log_amp_scale": numpyro.sample(
            "log_amp_scale", numpyro.distributions.Uniform(-2.0, 2.0)
        ),
        "mean": numpyro.sample(
            "mean",
            numpyro.distributions.Uniform(
                low=jnp.asarray([-0.1, -0.1]),
                high=jnp.asarray([0.1, 0.1]),
            ),
        ),
        "lag": numpyro.sample("lag", numpyro.distributions.Uniform(-10.0, 10.0)),
    }


def _make_numpyro_model(init_sampler):
    def numpyro_model(model):
        sample_params = init_sampler()
        model.sample(sample_params)

    return numpyro_model


def _run_random_search_benchmark(
    model, init_sampler, fit_key, *, n_sample, n_best, batch_size
):
    return random_search(
        model,
        init_sampler,
        fit_key,
        n_sample=n_sample,
        n_best=n_best,
        batch_size=batch_size,
        optimizer=optax.adam(1e-2),
        n_opt_step=1000,
    )


def _run_mcmc_benchmark(model, numpyro_model, mcmc_key, *, num_warmup, num_samples):
    nuts_kernel = numpyro.infer.NUTS(
        numpyro_model,
        dense_mass=True,
        target_accept_prob=0.9,
        init_strategy=numpyro.infer.init_to_median,
    )
    mcmc = numpyro.infer.MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(mcmc_key, model)
    return mcmc.get_samples()


def _block_until_ready(*values):
    for value in values:
        jax.tree_util.tree_map(
            lambda leaf: (
                leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf
            ),
            value,
        )

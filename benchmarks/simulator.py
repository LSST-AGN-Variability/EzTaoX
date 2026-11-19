"""Benchmarks for EzTaoX simulation"""

import contextlib
import functools

import jax
import jax.numpy as jnp
import jax.random as jr

import eztaox.kernels.quasisep as ekq
from eztaox.simulator import MultiVarSim, UniVarSim

# Run JAX computations inline instead of handing them to the CPU client's worker
# thread. The asynchronous dispatch adds a fixed, noisy wake-up latency (tens of
# microseconds on macOS) to every blocking call, which would dominate the small
# log-likelihood benchmarks without saying anything about eztaox itself.
with contextlib.suppress(AttributeError, ValueError, KeyError):
    # older JAX releases have no such option
    jax.config.update("jax_cpu_enable_async_dispatch", False)

DRW_PARAMS = {"tau": 100.0, "sigma": 0.1}
NBANDS = 2

# As in fitting.py, the suites build their simulators in cached module-level
# functions rather than a ``setup`` hook: asv_runner >= 0.3 re-runs a hook before
# every timed call, which would rebuild the simulator and put its JIT compilation
# inside each sample. With an explicit ``number`` the compile lands in asv's
# warmup phase instead.
_WARMUP_TIME = 0.1


def _log_kernel_param() -> jnp.ndarray:
    return jnp.stack([jnp.log(DRW_PARAMS["tau"]), jnp.log(DRW_PARAMS["sigma"])])


@functools.cache
def _univar_sim() -> tuple[UniVarSim, jnp.ndarray]:
    """Build the univariate simulator and its sampling grid once per process."""
    drw_true = ekq.Exp(scale=DRW_PARAMS["tau"], sigma=DRW_PARAMS["sigma"])
    t = jnp.arange(0.0, 4000.0, 1.0)
    s = UniVarSim(
        drw_true,
        min_dt=0.01,
        max_dt=float(t[-1]),
        init_params={"log_kernel_param": _log_kernel_param()},
        zero_mean=True,
    )
    return s, t


@functools.cache
def _multivar_sim() -> tuple[MultiVarSim, tuple[jnp.ndarray, jnp.ndarray]]:
    """Build the multivariate simulator and its sampling grid once per process."""
    drw_true = ekq.Exp(scale=DRW_PARAMS["tau"], sigma=DRW_PARAMS["sigma"])
    sim_params = {
        "log_kernel_param": _log_kernel_param(),
        "log_amp_scale": jnp.log(1.0),
        "lag": 10.0,
    }
    t = jnp.arange(0.0, 4000.0, 1.0)
    band = jr.choice(jr.PRNGKey(1), a=NBANDS, shape=t.shape, replace=True)
    s = MultiVarSim(
        drw_true,
        min_dt=0.01,
        max_dt=float(t[-1]),
        n_band=NBANDS,
        init_params=sim_params,
        zero_mean=True,
        has_lag=True,
    )
    return s, (t, band)


class UnivariateSimulatorSuite:
    """Timing benchmarks for the univariate simulator"""

    number = 1
    warmup_time = _WARMUP_TIME

    def time_run_sim(self):
        s, t = _univar_sim()
        return jax.block_until_ready(s.fixed_input(t, jax.random.key(11)))


class MultivariateSimulatorSuite:
    """Timing benchmarks for the multivariate simulator"""

    number = 1
    warmup_time = _WARMUP_TIME

    def time_run_sim(self):
        s, X = _multivar_sim()
        return jax.block_until_ready(s.fixed_input(X, jax.random.PRNGKey(12)))

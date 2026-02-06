"""Benchmarks for EzTaoX kernel fitting"""

import eztaox.kernels.quasisep as ekq
import jax
import jax.numpy as jnp
import tinygp
from eztaox.simulator import UniVarSim
from tinygp import GaussianProcess, kernels


class KernelFittingSuite:
    """Timing benchmarks for various kernel fittings"""

    # Size of lightcurve `n`
    params = [20, 100, 1_000]

    def setup(self, n):
        X, y, diag = generate_data(n)
        fitting_params = {
            "log_kernel_param": jnp.log(jnp.array([100, 0.1])),
            "log_amp_delta": jnp.log(0.6),
            "lag": jnp.array(10),
        }
        self.loss_exp = _precompile_exp_loss(X, y, diag, fitting_params)
        self.loss_m32 = _precompile_m32_loss(X, y, diag, fitting_params)
        self.fitting_params = fitting_params

    def time_run_exp_fitting(self, _):
        self.loss_exp(self.fitting_params).block_until_ready()

    def time_run_m32_fitting(self, _):
        self.loss_m32(self.fitting_params).block_until_ready()


def generate_data(n):
    """Setup data for fitting"""
    t_g, lc_g, yerr_g = generate_lc(n, "g")
    t_i, lc_i, yerr_i = generate_lc(n, "i")
    inds = jnp.argsort(jnp.concatenate((t_g, t_i)))
    X = (
        jnp.concatenate((t_g, t_i))[inds],
        jnp.concatenate(
            (
                jnp.zeros_like(t_g, dtype=int),
                jnp.ones_like(t_i, dtype=int),
            )
        )[inds],
    )
    lc_g -= jnp.median(lc_g)
    lc_i -= jnp.median(lc_i)
    y = jnp.concatenate((lc_g, lc_i))[inds]
    diag = jnp.concatenate((yerr_g, yerr_i))[inds] ** 2
    return X, y, diag


def generate_lc(n, band):
    """Generate single band light curve of size `n`"""
    amps = {"g": 0.35, "i": 0.25}
    taus = {"g": 100, "i": 150}
    snrs = {"g": 5, "i": 3}
    noise_seeds = {"g": 111, "i": 2}
    tau_true = taus[band]
    amps_true = amps[band]
    drw_true = ekq.Exp(scale=tau_true, sigma=amps_true)
    log_kernel_param = jnp.stack([jnp.log(tau_true), jnp.log(amps_true)])
    t = jnp.arange(0.0, n, 1.0)
    s = UniVarSim(
        drw_true,
        min_dt=0.01,
        max_dt=float(t[-1]),
        init_params={"log_kernel_param": log_kernel_param},
        zero_mean=True,
    )
    lc_key = jax.random.PRNGKey(11)
    t, lc = s.fixed_input(t, lc_key)
    noise_key = jax.random.PRNGKey(noise_seeds[band])
    yerr = jax.random.lognormal(noise_key, shape=lc.shape) * (lc / snrs[band])
    return t, lc + yerr, jnp.abs(yerr)


def _precompile_exp_loss(X, y, diag, fitting_params):
    """Precompile the JIT compiled loss function for exponential kernel"""

    @jax.jit
    def loss(params):
        kernel = kernels.quasisep.Exp(*jnp.exp(params["log_kernel_param"]))
        gp = GaussianProcess(
            kernel,
            X[0],
            diag=diag,
            mean=0.0,
            assume_sorted=True,
        )
        return -gp.log_probability(y)

    loss(fitting_params).block_until_ready()
    return loss


def _precompile_m32_loss(X, y, diag, fitting_params):
    """Precompile the JIT compiled loss function for Matern32 with lags"""
    log_amps, lags, t, band, inds = _make_m32_loss_args(X, fitting_params)

    @jax.jit
    def loss(params):
        kernel = MB(
            amplitudes=jnp.exp(log_amps),
            lags=lags,
            kernel=kernels.quasisep.Matern32(*jnp.exp(params["log_kernel_param"])),
        )
        gp = GaussianProcess(
            kernel,
            (t[inds], band[inds]),
            diag=diag[inds],
            mean=0.0,
            assume_sorted=True,
        )
        return -gp.log_probability(y)

    loss(fitting_params).block_until_ready()
    return loss


def _make_m32_loss_args(X, fitting_params):
    def _lag_transform(X, lags):
        t, band = X
        new_t = t - lags[band]
        inds = jnp.argsort(new_t)
        return (new_t, band), inds

    log_amps = jnp.insert(jnp.atleast_1d(fitting_params["log_amp_delta"]), 0, 0.0)
    lags = jnp.insert(jnp.atleast_1d(fitting_params["lag"]), 0, 0.0)
    t = X[0]
    band = X[1]
    new_X, inds = _lag_transform(X, lags)
    return log_amps, lags, t, band, inds


@tinygp.helpers.dataclass
class MB(tinygp.kernels.quasisep.Wrapper):
    amplitudes: jnp.ndarray
    lags: jnp.ndarray

    def coord_to_sortable(self, X):
        return X[0]

    def observation_model(self, X):
        return self.amplitudes[X[1]] * self.kernel.observation_model(X[0])

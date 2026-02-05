"""Benchmarks for EzTaoX kernel fitting"""

import eztaox.kernels.quasisep as ekq
import jax
import jax.numpy as jnp
import tinygp
from eztaox.simulator import UniVarSim
from tinygp import GaussianProcess, kernels


class KernelFittingSuite:
    """Timing benchmarks for various kernel fittings"""

    params = [20, 100, 1_000]

    def setup(self, n):
        self.X, self.y, self.diag = generate_data(n)
        self.fitting_params = {
            "log_kernel_param": jnp.log(jnp.array([100, 0.1])),
            "log_amp_delta": jnp.log(0.6),
            "lag": jnp.array(10),
        }
        self.loss_exp = self._precompile_exp_loss(n)
        self.loss_m32 = self._precompile_m32_loss(n)

    def _precompile_exp_loss(self, n):
        """Precompile the JIT compiled loss function for exponential kernel"""

        @jax.jit
        def loss(params):
            kernel = kernels.quasisep.Exp(*jnp.exp(params["log_kernel_param"]))
            gp = GaussianProcess(
                kernel,
                self.X[0],
                diag=self.diag,
                mean=0.0,
                assume_sorted=True,
            )
            return -gp.log_probability(self.y)

        loss(self.fitting_params).block_until_ready()
        return loss

    def _precompile_m32_loss(self, n):
        """Precompile the JIT compiled loss function for Matern32 with lags"""

        log_amps = amp_transform(self.fitting_params)
        lags = jnp.insert(jnp.atleast_1d(self.fitting_params["lag"]), 0, 0.0)

        t = self.X[0]
        band = self.X[1]

        new_X, inds = lag_transform(self.X, lags)
        # t, band = new_X

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
                diag=self.diag[inds],
                mean=0.0,
                assume_sorted=True,
            )
            return -gp.log_probability(self.y)

        loss(self.fitting_params).block_until_ready()
        return loss

    def time_run_exp_fitting(self, n):
        self.loss_exp(self.fitting_params).block_until_ready()

    def time_run_m32_fitting(self, n):
        self.loss_m32(self.fitting_params).block_until_ready()


def generate_lc(n, band):
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


# GP class
@tinygp.helpers.dataclass
class MB(tinygp.kernels.quasisep.Wrapper):
    amplitudes: jnp.ndarray
    lags: jnp.ndarray

    def coord_to_sortable(self, X):
        return X[0]

    def observation_model(self, X):
        return self.amplitudes[X[1]] * self.kernel.observation_model(X[0])


def amp_transform(params):
    return jnp.insert(jnp.atleast_1d(params["log_amp_delta"]), 0, 0.0)


def lag_transform(X, lags):
    t, band = X
    new_t = t - lags[band]
    inds = jnp.argsort(new_t)
    return (new_t, band), inds

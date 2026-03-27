"""Benchmarks for EzTaoX simulation"""

import jax
import jax.numpy as jnp
import jax.random as jr

import eztaox.kernels.quasisep as ekq
from eztaox.simulator import MultiVarSim, UniVarSim

DRW_PARAMS = {"tau": 100.0, "sigma": 0.1}
NBANDS = 2


class UnivariateSimulatorSuite:
    """Timing benchmarks for the univariate simulator"""

    def setup(self):
        drw_true = ekq.Exp(scale=DRW_PARAMS["tau"], sigma=DRW_PARAMS["sigma"])
        log_kernel_param = jnp.stack(
            [jnp.log(DRW_PARAMS["tau"]), jnp.log(DRW_PARAMS["sigma"])]
        )
        self.t = jnp.arange(0.0, 4000.0, 1.0)
        self.s = UniVarSim(
            drw_true,
            min_dt=0.01,
            max_dt=float(self.t[-1]),
            init_params={"log_kernel_param": log_kernel_param},
            zero_mean=True,
        )
        self.lc_key = jax.random.PRNGKey(11)

    def time_run_sim(self):
        return jax.block_until_ready(self.s.fixed_input(self.t, self.lc_key))


class MultivariateSimulatorSuite:
    """Timing benchmarks for the multivariate simulator"""

    def setup(self):
        drw_true = ekq.Exp(scale=DRW_PARAMS["tau"], sigma=DRW_PARAMS["sigma"])
        log_kernel_param = jnp.stack(
            [jnp.log(DRW_PARAMS["tau"]), jnp.log(DRW_PARAMS["sigma"])]
        )
        sim_params = {
            "log_kernel_param": log_kernel_param,
            "log_amp_scale": jnp.log(1.0),
            "lag": 10.0,
        }
        self.t = jnp.arange(0.0, 4000.0, 1.0)
        self.band = jr.choice(jr.PRNGKey(1), a=NBANDS, shape=self.t.shape, replace=True)
        self.s = MultiVarSim(
            drw_true,
            min_dt=0.01,
            max_dt=float(self.t[-1]),
            nBand=NBANDS,
            init_params=sim_params,
            zero_mean=True,
            has_lag=True,
        )
        self.lc_key = jax.random.PRNGKey(12)

    def time_run_sim(self):
        return jax.block_until_ready(
            self.s.fixed_input((self.t, self.band), self.lc_key)
        )

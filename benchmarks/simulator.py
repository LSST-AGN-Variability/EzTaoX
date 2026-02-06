"""Benchmarks for EzTaoX simulation"""

import eztaox.kernels.quasisep as ekq
import jax
import jax.numpy as jnp
from eztaox.simulator import UniVarSim


class UnivariateSimulatorSuite:
    """Timing benchmarks for the univariate simulator"""

    def setup(self):
        tau_true = 5.891242982962032
        sigma_true = 0.13896505738419102
        drw_true = ekq.Exp(scale=tau_true, sigma=sigma_true)
        log_kernel_param = jnp.stack([jnp.log(tau_true), jnp.log(sigma_true)])
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
        return self.s.fixed_input(self.t, self.lc_key)

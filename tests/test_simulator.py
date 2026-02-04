"""
Basic tests of the simulator function.
"""

import jax
import jax.numpy as jnp

import eztaox.kernels.quasisep as ekq
from eztaox.simulator import UniVarSim


def test_simulator_run_univarsim() -> None:
    """
    Test that the UniVarSim runs without error.
    """
    tau_true = 5.891242982962032
    sigma_true = 0.13896505738419102
    drw_true = ekq.Exp(scale=tau_true, sigma=sigma_true)

    t = jnp.arange(0.0, 4000.0, 1.0)
    s = UniVarSim(
        drw_true,
        0.01,
        float(t[-1]),
        init_params={
            "log_kernel_param": jnp.stack([jnp.log(tau_true), jnp.log(sigma_true)])
        },
        zero_mean=True,
    )

    sim_t, sim_y = s.fixed_input(t, jax.random.PRNGKey(11))
    assert sim_t.shape == sim_y.shape == t.shape

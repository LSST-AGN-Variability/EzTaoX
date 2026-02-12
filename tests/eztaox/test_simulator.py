"""
Basic tests of the simulator function.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import binned_statistic, ks_2samp

import eztaox.kernels.quasisep as ekq
from eztaox.simulator import UniVarSim


## Empirical structure function calculation
def _sf2(t, y, bins):
    """Calculate structure function squared

    Short description goes here

    Parameters
    ----------
    t : `np.array` [`float`]
        Times at which the measurment was conducted
    y : `np.array` [`float`]
        Measurment values
    bins : `np.array` [`float`]
        Bin edges for binned statistics
    """

    # dt
    dt_matrix = t.reshape((1, t.size)) - t.reshape((t.size, 1))
    dts = dt_matrix[dt_matrix > 0].flatten().astype(np.float16)

    # dm
    dm_matrix = y.reshape((1, y.size)) - y.reshape((y.size, 1))
    dms = dm_matrix[dt_matrix > 0].flatten().astype(np.float16)

    ## SF for each pair of observations
    sfs = dms**2

    # SF for at specific dt
    # the line below will throw error if the bins are not covering the whole range
    SFs, bin_edgs, _ = binned_statistic(dts, sfs, "mean", bins)

    return SFs, (bin_edgs[0:-1] + bin_edgs[1:]) / 2


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


def test_simulator_fixed_input_fast() -> None:
    """
    Test that the fixed_input_fast method returns the same output
    (on a statistical level) as fixed_input for the same input and random seed.
    """

    ## DRW simulator setup
    drw_scale, drw_sigma = 100.0, 0.2
    mindt, maxdt = 0.1, 2000.0
    master_key = jax.random.PRNGKey(0)
    sim_params = {
        "log_kernel_param": jnp.array([jnp.log(drw_scale), jnp.log(drw_sigma)]),
    }
    drw = ekq.Exp(scale=drw_scale, sigma=drw_sigma)
    s = UniVarSim(drw, mindt, maxdt, sim_params, init_seed=master_key)

    ## simulation configs
    nsim = 500
    npt = 100
    nbins = 10
    input_t = jnp.linspace(mindt, maxdt, npt)
    bins = np.logspace(np.log10(maxdt / npt), np.log10(maxdt), nbins)

    ## simulate using fixed_input
    SFs_fixed_input = []
    for i in range(nsim):
        key_sim = jax.random.PRNGKey(i)
        sim_t, sim_y = s.fixed_input(input_t, key_sim)
        SFs, _ = _sf2(sim_t, sim_y, bins=bins)
        SFs_fixed_input.append(SFs)
    SFs_fixed_input = np.array(SFs_fixed_input)

    ## simulate using fixed_input_fast
    SFs_fixed_input_fast = []
    for i in range(nsim):
        key_sim = jax.random.PRNGKey(i)
        sim_t, sim_y = s.fixed_input_fast(input_t, key_sim)
        SFs, _ = _sf2(sim_t, sim_y, bins=bins)
        SFs_fixed_input_fast.append(SFs)
    SFs_fixed_input_fast = np.array(SFs_fixed_input_fast)

    for i in range(nbins - 2):
        assert ks_2samp(SFs_fixed_input_fast[:, i], SFs_fixed_input[:, i]).pvalue > 0.05

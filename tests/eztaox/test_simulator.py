"""
Basic tests of the simulator function.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import binned_statistic, ks_2samp
from tinygp.kernels import Exp as Exp_nonqs

import eztaox.kernels.quasisep as ekq
from eztaox.simulator import MultiVarSim, UniVarSim


@pytest.fixture(
    params=[
        ekq.Matern32(sigma=1.8, scale=1.5),
        ekq.Matern32(1.5),
        ekq.Matern52(sigma=1.8, scale=1.5),
        ekq.Matern52(1.5),
        ekq.Celerite(1.1, 0.8, 0.9, 0.1),
        ekq.SHO(omega=1.5, quality=0.5, sigma=1.3),
        ekq.SHO(omega=1.5, quality=3.5, sigma=1.3),
        ekq.SHO(omega=1.5, quality=0.1, sigma=1.3),
        ekq.Exp(sigma=1.8, scale=1.5),
        ekq.Exp(1.5),
        1.5 * ekq.Matern52(1.5) + 0.3 * ekq.Exp(1.5),
        ekq.Matern52(1.5) * ekq.SHO(omega=1.5, quality=0.1),
        1.5 * ekq.Matern52(1.5) * ekq.Celerite(1.1, 0.8, 0.9, 0.1),
        ekq.Cosine(sigma=1.8, scale=1.5),
        1.8 * ekq.Cosine(1.5),
        # ekq.CARMA(alpha=jnp.array([1.4, 2.3, 1.5]), beta=jnp.array([0.1, 0.5])),
        ekq.CARMA(alpha=jnp.array([1, 1.2]), beta=jnp.array([1.0, 3.0])),
        ekq.CARMA(alpha=jnp.array([0.1, 1.1]), beta=jnp.array([1.0, 3.0])),
        ekq.CARMA(alpha=jnp.array([1.0 / 100]), beta=jnp.array([0.3])),
        ekq.CARMA(alpha=jnp.array([1, 1.2]), beta=jnp.array([1.0, 3.0]))
        * ekq.Matern52(1.5)
        + ekq.CARMA(alpha=jnp.array([1.4, 2.3, 1.5]), beta=jnp.array([0.1, 0.5]))
        * ekq.SHO(omega=1.5, quality=0.1, sigma=1.3),
    ]
)
def kernel(request) -> ekq.Kernel:
    return request.param


def _is_sorted(arr) -> bool:
    """Check if an array is sorted in non-decreasing order."""
    return jnp.all(arr[:-1] <= arr[1:])


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


def test_simulator_run_univarsim(kernel) -> None:
    """
    Test that the UniVarSim runs without error.
    """

    t = jnp.arange(0.0, 4000.0, 1.0)
    s = UniVarSim(
        kernel,
        0.01,
        float(t[-1]),
        init_params={
            "log_kernel_param": jnp.log(jax.flatten_util.ravel_pytree(kernel)[0]),
        },
        zero_mean=True,
    )

    sim_t, sim_y = s.fixed_input(t, jax.random.PRNGKey(11))
    assert sim_t.shape == sim_y.shape == t.shape
    assert not jnp.isnan(sim_t).any()
    assert not jnp.isnan(sim_y).any()
    assert _is_sorted(sim_t)


def test_simulator_fixed_input_fast() -> None:
    """
    Test that the fixed_input_fast method returns the same output
    (on a statistical level) as fixed_input for the same input and random seed.
    """

    ## DRW simulator setup
    drw_scale, drw_sigma = 100.0, 0.2
    mindt, maxdt = 0.1, 2000.0
    main_key = jax.random.PRNGKey(0)
    sim_params = {
        "log_kernel_param": jnp.array([jnp.log(drw_scale), jnp.log(drw_sigma)]),
    }
    drw = ekq.Exp(scale=drw_scale, sigma=drw_sigma)
    s = UniVarSim(drw, mindt, maxdt, sim_params, init_seed=main_key)

    ## simulation configs
    nsim = 500
    npt = 100
    nbins = 10
    input_t = jnp.linspace(mindt, maxdt, npt)
    bins = np.logspace(np.log10(maxdt / npt), np.log10(maxdt), nbins)

    # split keys for each simulation
    sim_keys = jax.random.split(main_key, nsim)

    ## simulate using fixed_input
    SFs_fixed_input = []
    for i in range(nsim):
        key_sim = sim_keys[i]
        sim_t, sim_y = s.fixed_input(input_t, key_sim)
        SFs, _ = _sf2(sim_t, sim_y, bins=bins)
        SFs_fixed_input.append(SFs)
    SFs_fixed_input = np.array(SFs_fixed_input)

    ## simulate using fixed_input_fast
    SFs_fixed_input_fast = []
    for i in range(nsim):
        key_sim, _ = jax.random.split(sim_keys[i], 2)
        sim_t, sim_y = s.fixed_input_fast(input_t, key_sim)
        SFs, _ = _sf2(sim_t, sim_y, bins=bins)
        SFs_fixed_input_fast.append(SFs)
    SFs_fixed_input_fast = np.array(SFs_fixed_input_fast)

    for i in range(nbins - 2):
        assert ks_2samp(SFs_fixed_input_fast[:, i], SFs_fixed_input[:, i]).pvalue > 0.05


def test_simulator_multivar(kernel) -> None:
    """Test that the MultiVarSim runs without error and produces sorted outputs."""
    mindt, maxdt = 0.1, 2000.0
    n_random = 1000
    nband = 2
    main_key = jax.random.PRNGKey(101)
    sim_keys = jax.random.split(main_key, 5)
    sim_params = {
        "log_kernel_param": jnp.log(jax.flatten_util.ravel_pytree(kernel)[0]),
        "log_amp_scale": jnp.array([0.0]),
        "lag": 10.0,
    }

    for has_lag in [False, True]:
        s = MultiVarSim(kernel, mindt, maxdt, nband, sim_params, has_lag=has_lag)

        # full simulation
        simX_full, simY_full = s.full(sim_keys[0])
        assert not jnp.isnan(simX_full[0]).any()
        assert not jnp.isnan(simX_full[1]).any()
        assert not jnp.isnan(simY_full).any()
        assert _is_sorted(simX_full[0][simX_full[1] == 0])
        assert _is_sorted(simX_full[0][simX_full[1] == 1])

        # random simulation
        simX_rand, simY_rand = s.random(n_random, sim_keys[1], sim_keys[2])
        assert not jnp.isnan(simX_rand[0]).any()
        assert not jnp.isnan(simX_rand[1]).any()
        assert not jnp.isnan(simY_rand).any()
        assert _is_sorted(simX_rand[0][simX_rand[1] == 0])
        assert _is_sorted(simX_rand[0][simX_rand[1] == 1])

        # fixed input simulation
        inputX = (jnp.linspace(0, 100, 6), jnp.asarray([0, 1, 1, 1, 0, 1]))
        simX_fixed, simY_fixed = s.fixed_input(inputX, sim_keys[3])
        assert (
            simX_fixed[0][inputX[1] == 0].shape
            == simY_fixed[inputX[1] == 0].shape
            == inputX[0][inputX[1] == 0].shape
        )
        assert (
            simX_fixed[0][inputX[1] == 1].shape
            == simY_fixed[inputX[1] == 1].shape
            == inputX[0][inputX[1] == 1].shape
        )
        assert not jnp.isnan(simX_fixed[0]).any()
        assert not jnp.isnan(simX_fixed[1]).any()
        assert not jnp.isnan(simY_fixed).any()
        assert _is_sorted(simX_fixed[0][simX_fixed[1] == 0])
        assert _is_sorted(simX_fixed[0][simX_fixed[1] == 1])

        # fixed input fast simulation
        simX_fixed_fast, simY_fixed_fast = s.fixed_input_fast(inputX, sim_keys[4])
        assert (
            simX_fixed_fast[0][inputX[1] == 0].shape
            == simY_fixed_fast[inputX[1] == 0].shape
            == inputX[0][inputX[1] == 0].shape
        )
        assert (
            simX_fixed_fast[0][inputX[1] == 1].shape
            == simY_fixed_fast[inputX[1] == 1].shape
            == inputX[0][inputX[1] == 1].shape
        )
        assert not jnp.isnan(simX_fixed_fast[0]).any()
        assert not jnp.isnan(simX_fixed_fast[1]).any()
        assert not jnp.isnan(simY_fixed_fast).any()
        assert _is_sorted(simX_fixed_fast[0][simX_fixed_fast[1] == 0])
        assert _is_sorted(simX_fixed_fast[0][simX_fixed_fast[1] == 1])


def test_simulator_nonqs_exp_minimal() -> None:
    """
    Minimal smoke test for a non-quasisep kernel.

    Keep this tiny so it provides quick sanity coverage without the
    runtime/memory cost of running the full kernel-parametrized suite.
    """
    kernel = 1.5 * Exp_nonqs(scale=1.8)
    t = jnp.linspace(0.0, 20.0, 32)
    s = UniVarSim(
        kernel,
        0.1,
        float(t[-1]),
        init_params={
            "log_kernel_param": jnp.log(jax.flatten_util.ravel_pytree(kernel)[0]),
        },
        zero_mean=True,
    )

    sim_t, sim_y = s.fixed_input(t, jax.random.PRNGKey(123))
    assert sim_t.shape == sim_y.shape == t.shape
    assert not jnp.isnan(sim_t).any()
    assert not jnp.isnan(sim_y).any()
    assert _is_sorted(sim_t)

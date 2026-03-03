"""Tests for fitting simulated data and make sure returned parameters are
well recoved"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
from joblib import Parallel, delayed
from numpyro import distributions as dist
from scipy.stats import median_abs_deviation as mad

from eztaox.fitter import random_search
from eztaox.kernels.quasisep import Exp
from eztaox.models import MultiVarModel


def test_multivar_drw(test_data_dir, basekey_seed) -> None:
    """
    Test multivariate DRW fitting.
    """

    # load test data
    data = np.load(test_data_dir + "/unit_test_lc.npz")
    ts = data["ts"]
    bands = data["bands"]
    ys = data["ys"]

    # config for fitting
    nSample = 10_000
    nBest = 10
    fit_bands = [0, 1]

    # sampler function
    def initSampler():  # noqa: N802
        # GP kernel param
        log_drw_scale = numpyro.sample(
            "drw_scale", dist.Uniform(jnp.log(0.1), jnp.log(10000))
        )
        log_drw_sigma = numpyro.sample(
            "drw_sigma", dist.Uniform(jnp.log(0.01), jnp.log(2))
        )
        log_kernel_param = jnp.stack([log_drw_scale, log_drw_sigma])
        numpyro.deterministic("log_kernel_param", log_kernel_param)

        # amp scale
        log_amp_scale = numpyro.sample("log_amp_scale", dist.Uniform(-2, 2))

        # mean
        mean = numpyro.sample("mean", dist.Normal(loc=0.0, scale=0.1))
        lag = numpyro.sample("lag", dist.Uniform(-10.0, 10.0))

        sample_params = {
            "log_kernel_param": log_kernel_param,
            "log_amp_scale": log_amp_scale,
            "mean": mean,
            "lag": lag,
        }
        return sample_params

    # fit function for parallelization
    def fit(X, y, yerr, nBand, basekey_seed, key_index):
        m = MultiVarModel(
            X, y, yerr, Exp(scale=100.0, sigma=0.1), nBand, zero_mean=True, has_lag=True
        )
        fit_key = jr.fold_in(jr.PRNGKey(basekey_seed), key_index)
        bestP, ll = random_search(m, initSampler, fit_key, nSample, nBest)
        return bestP

    # parallelized fitting
    bestPs = Parallel(n_jobs=-1)(
        delayed(fit)(
            (
                ts[i][np.isin(bands[i], fit_bands)],
                bands[i][np.isin(bands[i], fit_bands)],
            ),
            ys[i][np.isin(bands[i], fit_bands)],
            jnp.ones_like(ys[i][np.isin(bands[i], fit_bands)]) * 1e-6,
            nBand=len(fit_bands),
            basekey_seed=basekey_seed,
            key_index=i,
        )
        for i in range(len(ts))
    )

    # format results: list of dict to dict of list
    bestP_all = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *bestPs)

    # check if the best parameters are close to the true parameters
    true_params = {
        "log_kernel_param": jnp.array([jnp.log(50.0), jnp.log(0.2)]),
        "log_amp_scale": jnp.array(0.0),
        "mean": jnp.array(0.0),
        "lag": jnp.array(2.0),
    }

    # check DRW tau
    tau_diff = bestP_all["log_kernel_param"][:, 0] - true_params["log_kernel_param"][0]
    assert np.mean(tau_diff) < 0.1
    assert mad(tau_diff, scale="normal") < 0.3

    # check DRW amp
    amp_diff = bestP_all["log_kernel_param"][:, 1] - true_params["log_kernel_param"][1]
    assert np.mean(amp_diff) < 0.1
    assert mad(amp_diff, scale="normal") < 0.3

    # check interband lag
    lag_diff = bestP_all["lag"] - true_params["lag"]
    assert np.mean(lag_diff) < 0.2
    assert mad(lag_diff, scale="normal") < 1

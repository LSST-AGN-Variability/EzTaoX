"""Tests for fitting simulated data and make sure returned parameters are
well recoved"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
from joblib import Parallel, delayed
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median
from scipy.stats import median_abs_deviation as mad

from eztaox.fitter import random_search, random_search_adam
from eztaox.kernels.quasisep import Exp
from eztaox.models import MultiVarModel


# sampler function
def initSampler():  # noqa: N802
    # GP kernel param
    log_drw_scale = numpyro.sample(
        "drw_scale", dist.Uniform(jnp.log(0.1), jnp.log(10000))
    )
    log_drw_sigma = numpyro.sample("drw_sigma", dist.Uniform(jnp.log(0.01), jnp.log(2)))
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


def test_multivar_drw(test_data, basekey_seed) -> None:
    """
    Test multivariate DRW fitting.
    """

    # load test data
    ts = test_data["ts"]
    bands = test_data["bands"]
    ys = test_data["ys"]

    # config for fitting
    nSample = 10_000
    nBest = 10
    fit_bands = [0, 1]

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


def test_multivar_drw_adam(test_data, basekey_seed) -> None:
    """
    Test multivariate DRW fitting with Adam refinement.
    """

    # load test data
    ts = test_data["ts"]
    bands = test_data["bands"]
    ys = test_data["ys"]

    # config for fitting
    nSample = 2_000
    nBest = 5
    fit_bands = [0, 1]

    def fit(X, y, yerr, nBand, basekey_seed, key_index):
        m = MultiVarModel(
            X, y, yerr, Exp(scale=100.0, sigma=0.1), nBand, zero_mean=True, has_lag=True
        )
        fit_key = jr.fold_in(jr.PRNGKey(basekey_seed), key_index)
        bestP, ll = random_search_adam(
            m,
            initSampler,
            fit_key,
            nSample,
            nBest,
            nStep=400,
            learning_rate=1e-2,
        )
        assert jnp.ndim(ll) == 0
        return bestP

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
        for i in range(min(len(ts), 4))
    )

    bestP_all = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *bestPs)

    true_params = {
        "log_kernel_param": jnp.array([jnp.log(50.0), jnp.log(0.2)]),
        "lag": jnp.array(2.0),
    }

    tau_diff = bestP_all["log_kernel_param"][:, 0] - true_params["log_kernel_param"][0]
    assert np.mean(np.abs(np.asarray(tau_diff))) < 0.5

    amp_diff = bestP_all["log_kernel_param"][:, 1] - true_params["log_kernel_param"][1]
    assert np.mean(np.abs(np.asarray(amp_diff))) < 0.5

    lag_diff = bestP_all["lag"] - true_params["lag"]
    assert np.mean(np.abs(np.asarray(lag_diff))) < 2.0


def test_multivar_drw_mcmc(test_data, basekey_seed) -> None:
    """
    Test multivariate DRW fitting with MCMC.
    """

    # load test data
    ts = test_data["ts"]
    bands = test_data["bands"]
    ys = test_data["ys"]

    # define model parameters
    has_lag = True  # True: Fit for inter-band lag
    zero_mean = True  # True: Fit for light curve mean
    nBand = 2  # number of bands in the provide light curve (X, y, yerr)
    fit_bands = [0, 1]  # which bands to fit

    # format input data for fitting, only use the specified bands for fitting
    X = (
        ts[0][np.isin(bands[0], fit_bands)],
        bands[0][np.isin(bands[0], fit_bands)],
    )
    y = ys[0][np.isin(bands[0], fit_bands)]
    yerr = jnp.ones_like(y) * 1e-6

    def numpyro_model(m, X, yerr, y=None):
        sample_params = initSampler()
        m.sample(sample_params)

    # initialize a GP kernel, note the initial parameters are not used in the fitting
    k = Exp(scale=100.0, sigma=1.0)
    m = MultiVarModel(X, y, yerr, k, nBand, has_lag=has_lag, zero_mean=zero_mean)

    nuts_kernel = NUTS(
        numpyro_model,
        dense_mass=True,
        target_accept_prob=0.9,
        init_strategy=init_to_median,
    )

    mcmc = MCMC(
        nuts_kernel,
        num_warmup=500,
        num_samples=1000,
        num_chains=1,
        progress_bar=False,
    )

    mcmc.run(jax.random.PRNGKey(basekey_seed), m, X, yerr, y=y)

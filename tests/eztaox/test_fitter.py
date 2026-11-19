"""Tests for fitting simulated data and make sure returned parameters are
well recoved"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import optax
import pytest
from joblib import Parallel, delayed
from numpyro import distributions as dist
from scipy.stats import median_abs_deviation as mad

import eztaox.fitter as fitter_module
from eztaox.fitter import random_search, simple_optimizer
from eztaox.kernels.quasisep import Exp
from eztaox.models import MultiVarModel, UniVarModel


# sampler function
def init_sampler():  # noqa: N802
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


@pytest.mark.parametrize(
    ("optimizer", "use_value_and_grad_from_state", "n_opt_step", "n_runs"),
    [
        pytest.param(
            optax.adam(1e-2),
            False,
            1000,
            50,
            id="adam",
        ),
        # pytest.param(
        #     optax.lbfgs(),
        #     True,
        #     100,
        #     50,
        #     id="lbfgs",
        # ),
    ],
)
def test_multivar_drw(
    test_data,
    basekey_seed,
    optimizer,
    use_value_and_grad_from_state: bool,
    n_opt_step: int,
    n_runs: int,
) -> None:
    """
    Test multivariate DRW fitting with Adam and L-BFGS refinement.
    """

    # load test data
    ts = test_data["ts"]
    bands = test_data["bands"]
    ys = test_data["ys"]

    # config for fitting
    n_sample = 2_000
    n_best = 5
    fit_bands = [0, 1]

    def fit(X, y, yerr, n_band, basekey_seed, key_index):
        m = MultiVarModel(
            X,
            y,
            yerr,
            Exp(scale=100.0, sigma=0.1),
            n_band,
            zero_mean=True,
            has_lag=True,
        )
        fit_key = jr.fold_in(jr.PRNGKey(basekey_seed), key_index)
        bestP, ll = random_search(
            m,
            init_sampler,
            fit_key,
            n_sample,
            n_best,
            optimizer=optimizer,
            n_opt_step=n_opt_step,
            use_value_and_grad_from_state=use_value_and_grad_from_state,
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
            n_band=len(fit_bands),
            basekey_seed=basekey_seed,
            key_index=i,
        )
        for i in range(len(ts))[:: int(len(ts) / n_runs)]
    )

    bestP_all = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *bestPs)

    true_params = {
        "log_kernel_param": jnp.array([jnp.log(50.0), jnp.log(0.2)]),
        "lag": jnp.array(2.0),
    }

    # check DRW tau
    tau_diff = bestP_all["log_kernel_param"][:, 0] - true_params["log_kernel_param"][0]
    assert np.mean(np.abs(np.asarray(tau_diff))) < 0.5
    assert np.mean(tau_diff) < 0.1
    assert mad(tau_diff, scale="normal") < 0.3

    # check DRW amp
    amp_diff = bestP_all["log_kernel_param"][:, 1] - true_params["log_kernel_param"][1]
    assert np.mean(np.abs(np.asarray(amp_diff))) < 0.5
    assert np.mean(amp_diff) < 0.1
    assert mad(amp_diff, scale="normal") < 0.3

    # check interband lag
    lag_diff = bestP_all["lag"] - true_params["lag"]
    assert np.mean(np.abs(np.asarray(lag_diff))) < 2.0
    assert np.mean(lag_diff) < 0.2
    assert mad(lag_diff, scale="normal") < 1


@pytest.mark.parametrize(
    ("optimizer", "use_value_and_grad_from_state", "n_step"),
    [
        (optax.adam(1e-2), False, 5),
        (optax.lbfgs(), True, 3),
    ],
)
def test_simple_optimizer_runs(
    optimizer, use_value_and_grad_from_state: bool, n_step: int
) -> None:
    """Smoke test simple_optimizer with both plain and state-aware optimizers."""
    x = jnp.linspace(0.0, 2.0 * jnp.pi, 32)
    y = jnp.sin(x)
    yerr = jnp.ones_like(x) * 0.05
    kernel = Exp(scale=1.5, sigma=0.8)
    init_sample = {
        "log_kernel_param": jnp.log(jax.flatten_util.ravel_pytree(kernel)[0]),
        "mean": jnp.array(0.1),
        "log_jitter": jnp.array(-4.0),
    }
    model = UniVarModel(x, y, yerr, kernel, zero_mean=False, has_jitter=True)

    params, (param_hist, loss_hist, grad_hist) = simple_optimizer(
        model,
        init_sample,
        optimizer=optimizer,
        n_step=n_step,
        use_value_and_grad_from_state=use_value_and_grad_from_state,
    )

    assert set(params) == set(init_sample)
    assert loss_hist.shape == (n_step,)
    assert param_hist["log_kernel_param"].shape[0] == n_step
    assert grad_hist["log_kernel_param"].shape[0] == n_step
    assert jnp.all(jnp.isfinite(loss_hist))
    assert jnp.isfinite(model.log_prob(params))


def test_random_search_uses_fixed_loop(monkeypatch) -> None:
    """random_search should keep using n_opt_step unless both stop args are set."""
    x = jnp.linspace(0.0, 2.0 * jnp.pi, 32)
    y = jnp.sin(x)
    yerr = jnp.ones_like(x) * 0.05
    kernel = Exp(scale=1.5, sigma=0.8)
    init_sample = {
        "log_kernel_param": jnp.log(jax.flatten_util.ravel_pytree(kernel)[0]),
        "mean": jnp.array(0.1),
        "log_jitter": jnp.array(-4.0),
    }
    model = UniVarModel(x, y, yerr, kernel, zero_mean=False, has_jitter=True)

    def init_sampler():
        return init_sample

    step_calls = {"count": 0}
    original_step = fitter_module._optimizer_step

    def counting_step(*args, **kwargs):
        step_calls["count"] += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(fitter_module, "_optimizer_step", counting_step)

    params, log_likelihood = random_search(
        model,
        init_sampler,
        jr.PRNGKey(0),
        n_sample=1,
        n_best=1,
        optimizer=optax.adam(1e-2),
        n_opt_step=3,
        max_opt_step=10,
        tol=None,
    )

    assert step_calls["count"] == 3
    assert set(params) == set(init_sample)
    assert jnp.ndim(log_likelihood) == 0


def test_random_search_stops_early_with_tol(monkeypatch) -> None:
    """random_search should stop after the first step when tol is very large."""
    x = jnp.linspace(0.0, 2.0 * jnp.pi, 32)
    y = jnp.sin(x)
    yerr = jnp.ones_like(x) * 0.05
    kernel = Exp(scale=1.5, sigma=0.8)
    init_sample = {
        "log_kernel_param": jnp.log(jax.flatten_util.ravel_pytree(kernel)[0]),
        "mean": jnp.array(0.1),
        "log_jitter": jnp.array(-4.0),
    }
    model = UniVarModel(x, y, yerr, kernel, zero_mean=False, has_jitter=True)

    def init_sampler():
        return init_sample

    step_calls = {"count": 0}
    original_step = fitter_module._optimizer_step_from_state

    def counting_step(*args, **kwargs):
        step_calls["count"] += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(fitter_module, "_optimizer_step_from_state", counting_step)

    params, log_likelihood = random_search(
        model,
        init_sampler,
        jr.PRNGKey(0),
        n_sample=1,
        n_best=1,
        optimizer=optax.lbfgs(),
        n_opt_step=3,
        max_opt_step=10,
        tol=1e6,
        use_value_and_grad_from_state=True,
    )

    assert step_calls["count"] == 1
    assert set(params) == set(init_sample)
    assert jnp.ndim(log_likelihood) == 0

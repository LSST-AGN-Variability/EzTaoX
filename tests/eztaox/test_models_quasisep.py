"""Tests for Models taking Quasisep kernels."""

import jax
import jax.flatten_util
import jax.numpy as jnp
import pytest
from numpy import random as np_random
from tinygp import GaussianProcess
from tinygp.helpers import JAXArray
from tinygp.kernels import Exp as Exp_nonqs
from tinygp.test_utils import assert_allclose

from eztaox.kernels import quasisep
from eztaox.models import MultiVarModel, UniVarModel


@pytest.fixture
def random():
    return np_random.default_rng(12345)


@pytest.fixture
def data(random) -> tuple[JAXArray, JAXArray, JAXArray]:
    x = jnp.sort(random.uniform(-3, 3, 50))
    y = jnp.sin(x)
    b = jnp.array(random.choice(3, 50))
    return x, y, b


@pytest.fixture(
    params=[
        quasisep.Matern32(sigma=1.8, scale=1.5),
        quasisep.Matern32(1.5),
        quasisep.Matern52(sigma=1.8, scale=1.5),
        quasisep.Matern52(1.5),
        quasisep.Celerite(1.1, 0.8, 0.9, 0.1),
        quasisep.SHO(omega=1.5, quality=0.5, sigma=1.3),
        quasisep.SHO(omega=1.5, quality=3.5, sigma=1.3),
        quasisep.SHO(omega=1.5, quality=0.1, sigma=1.3),
        quasisep.Exp(sigma=1.8, scale=1.5),
        quasisep.Exp(1.5),
        1.5 * quasisep.Matern52(1.5) + 0.3 * quasisep.Exp(1.5),
        quasisep.Matern52(1.5) * quasisep.SHO(omega=1.5, quality=0.1),
        1.5 * quasisep.Matern52(1.5) * quasisep.Celerite(1.1, 0.8, 0.9, 0.1),
        quasisep.Cosine(sigma=1.8, scale=1.5),
        1.8 * quasisep.Cosine(1.5),
        quasisep.CARMA(alpha=jnp.array([1.4, 2.3, 1.5]), beta=jnp.array([0.1, 0.5])),
        quasisep.CARMA(alpha=jnp.array([1, 1.2]), beta=jnp.array([1.0, 3.0])),
        quasisep.CARMA(alpha=jnp.array([0.1, 1.1]), beta=jnp.array([1.0, 3.0])),
        quasisep.CARMA(alpha=jnp.array([1.0 / 100]), beta=jnp.array([0.3])),
        quasisep.CARMA(alpha=jnp.array([1, 1.2]), beta=jnp.array([1.0, 3.0]))
        * quasisep.Matern52(1.5)
        + quasisep.CARMA(alpha=jnp.array([1.4, 2.3, 1.5]), beta=jnp.array([0.1, 0.5]))
        * quasisep.SHO(omega=1.5, quality=0.1, sigma=1.3),
    ]
)
def kernel(request):
    return request.param


def test_univar(data, kernel, random) -> None:
    x, y, _ = data

    model_param = {
        "log_kernel_param": jnp.log(jax.flatten_util.ravel_pytree(kernel)[0]),
        "mean": jnp.array(random.uniform(-1, 1)),
        "log_jitter": jnp.array(random.uniform(-20, 5)),
    }

    # native tinygp gp
    gp1 = GaussianProcess(
        kernel,
        x,
        diag=0.01 + jnp.exp(model_param["log_jitter"]) ** 2,
        mean=model_param["mean"],
    )

    # extao gp
    m = UniVarModel(
        x, y, jnp.ones_like(x) * 0.1, kernel, zero_mean=False, has_jitter=True
    )
    gp2, _ = m._build_gp(model_param)

    # check consistency for log probability and grad of log probability
    assert_allclose(gp1.log_probability(y), gp2.log_probability(y))
    assert_allclose(jax.grad(gp1.log_probability)(y), jax.grad(gp2.log_probability)(y))


def test_univar_jit_grad(data, kernel, random) -> None:
    x, y, _ = data

    model_param = {
        "log_kernel_param": jnp.log(jax.flatten_util.ravel_pytree(kernel)[0]),
        "mean": jnp.array(random.uniform(-1, 1)),
        "log_jitter": jnp.array(random.uniform(-20, 5)),
    }
    m = UniVarModel(
        x, y, jnp.ones_like(x) * 0.1, kernel, zero_mean=False, has_jitter=True
    )

    @jax.jit
    def loss(params) -> JAXArray:
        return m.log_prob(params)

    loss(model_param)
    jax.grad(loss)(model_param)


def test_multivar(data, kernel, random) -> None:
    x, y, b = data

    model_param = {
        "log_kernel_param": jnp.log(jax.flatten_util.ravel_pytree(kernel)[0]),
        "log_amp_scale": jnp.array(random.uniform(-1, 1, 2)),
        "mean": jnp.array(random.uniform(-1, 1, 3)),
        "log_jitter": jnp.array(random.uniform(-20, 5, 3)),
    }

    # native tinygp gp
    amplitudes = jnp.exp(
        jnp.insert(jnp.atleast_1d(model_param["log_amp_scale"]), 0, 0.0)
    )
    gp1 = GaussianProcess(
        quasisep.MultibandLowRank(
            params={"amplitudes": amplitudes},
            kernel=kernel,
        ),
        (x, b),
        diag=0.01 + (jnp.exp(model_param["log_jitter"]) ** 2)[b],
        mean=0.0,
        assume_sorted=True,
    )

    # extaox gp
    m = MultiVarModel(
        (x, b), y, jnp.ones_like(x) * 0.1, kernel, 3, zero_mean=False, has_jitter=True
    )
    gp2, _ = m._build_gp(model_param)

    # check consistency for log probability and grad of log probability
    assert_allclose(
        gp1.log_probability(y - model_param["mean"][b]), gp2.log_probability(y)
    )
    assert_allclose(
        jax.grad(gp1.log_probability)(y - model_param["mean"][b]),
        jax.grad(gp2.log_probability)(y),
    )


def test_multivar_jit_grad(data, kernel, random) -> None:
    x, y, b = data

    model_param = {
        "log_kernel_param": jnp.log(jax.flatten_util.ravel_pytree(kernel)[0]),
        "log_amp_scale": jnp.array(random.uniform(-1, 1, 2)),
        "mean": jnp.array(random.uniform(-1, 1, 3)),
        "log_jitter": jnp.array(random.uniform(-20, 5, 3)),
    }

    # extaox gp
    m = MultiVarModel(
        (x, b), y, jnp.ones_like(x) * 0.1, kernel, 3, zero_mean=False, has_jitter=True
    )

    @jax.jit
    def loss(params) -> JAXArray:
        return m.log_prob(params)

    loss(model_param)
    jax.grad(loss)(model_param)


def test_lag_transform(data, kernel, random) -> None:
    x, y, b = data

    # randomize the order -> test if eztao model does the sorting
    x_unsort = random.choice(x, len(x))

    # model
    model_param = {
        "log_kernel_param": jnp.log(jax.flatten_util.ravel_pytree(kernel)[0]),
        "log_amp_scale": jnp.array(random.uniform(-1, 1, 2)),
        "mean": jnp.array(random.uniform(-1, 1, 3)),
        "log_jitter": jnp.array(random.uniform(-20, 5, 3)),
        "lag": jnp.array(random.uniform(-10, 100, 2)),
    }
    m = MultiVarModel(
        (x_unsort, b),
        y,
        jnp.ones_like(x) * 0.1,
        kernel,
        3,
        zero_mean=False,
        has_jitter=True,
        has_lag=True,
    )

    # lag_transform using model function
    new_X, inds = m.lag_transform(True, model_param, (x_unsort, b))

    # correct time shift
    assert_allclose(new_X[0][b == 1] + model_param["lag"][0], x_unsort[b == 1])
    assert_allclose(new_X[0][b == 2] + model_param["lag"][1], x_unsort[b == 2])

    # new_X[0] should be sorted by inds
    assert jnp.all(new_X[0][inds][:-1] <= new_X[0][inds][1:])


def test_multivar_pred_unsorted_inputs(data, kernel, random) -> None:
    """Test prediction with unsorted inputs."""
    t, y, b = data
    yerr = jnp.ones_like(t) * 0.05

    params = {
        "log_kernel_param": jnp.log(jax.flatten_util.ravel_pytree(kernel)[0]),
        "log_amp_scale": jnp.array([0.0, 0.0]),
        "lag": jnp.array([1.0, 2.0]),
    }

    model = MultiVarModel((t, b), y, yerr, kernel, nBand=3, has_lag=True)

    # unsorted inputs
    t_pred = jnp.array([4.2, 0.3, 3.4, 1.7])
    b_pred = jnp.array([1, 2, 1, 0])

    # predict with unsorted inputs
    mean_unsorted, std_unsorted = model.pred(params, (t_pred, b_pred))

    # predict with sorted inputs
    idx = jnp.argsort(t_pred)
    mean_sorted, std_sorted = model.pred(params, (t_pred[idx], b_pred[idx]))

    # check if the prediction is the same
    assert_allclose(mean_unsorted[idx], mean_sorted)
    assert_allclose(std_unsorted[idx], std_sorted)


def test_aic_bic(data, kernel, random) -> None:
    x, y, b = data

    model_param = {
        "log_kernel_param": jnp.log(jax.flatten_util.ravel_pytree(kernel)[0]),
        "log_amp_scale": jnp.array(random.uniform(-1, 1, 2)),
        "mean": jnp.array(random.uniform(-1, 1, 3)),
        "log_jitter": jnp.array(random.uniform(-20, 5, 3)),
    }

    m = MultiVarModel(
        (x, b), y, jnp.ones_like(x) * 0.1, kernel, 3, zero_mean=False, has_jitter=True
    )
    log_prob = m.log_prob(model_param)
    n_params = len(jax.flatten_util.ravel_pytree(model_param)[0])
    n_data = len(y)

    aic = m.aic(model_param)
    bic = m.bic(model_param)

    expected_aic = 2 * n_params - 2 * log_prob
    expected_bic = n_params * jnp.log(n_data) - 2 * log_prob

    assert_allclose(aic, expected_aic)
    assert_allclose(bic, expected_bic)


def test_univar_qs_vs_nonqs_exp_same(data, random):
    x, y, _ = data
    yerr = jnp.ones_like(x) * 0.1

    sigma = 1.8
    scale = 1.5

    # Quasisep kernel
    k_qs = quasisep.Exp(sigma=sigma, scale=scale)

    # Non-quasisep tinygp kernel
    k_nonqs = sigma**2 * Exp_nonqs(scale=scale)

    mean = jnp.array(random.uniform(-1, 1))
    log_jitter = jnp.array(random.uniform(-20, 5))

    # Build models
    m_qs = UniVarModel(x, y, yerr, k_qs, zero_mean=False, has_jitter=True)
    m_nonqs = UniVarModel(x, y, yerr, k_nonqs, zero_mean=False, has_jitter=True)

    # Each model gets params built from its own kernel pytree
    theta_qs, _ = jax.flatten_util.ravel_pytree(k_qs)
    theta_nonqs, _ = jax.flatten_util.ravel_pytree(k_nonqs)

    p_qs = {
        "log_kernel_param": jnp.log(theta_qs),
        "mean": mean,
        "log_jitter": log_jitter,
    }

    p_nonqs = {
        "log_kernel_param": jnp.log(theta_nonqs),
        "mean": mean,
        "log_jitter": log_jitter,
    }

    # Reformat params
    lk = p_nonqs["log_kernel_param"]
    p_nonqs["log_kernel_param"] = jnp.stack([lk[1], 0.5 * lk[0]])

    # Log-prob equality
    assert_allclose(m_qs.log_prob(p_qs), m_nonqs.log_prob(p_nonqs))

    # Gradient equality
    g_qs = jax.grad(m_qs.log_prob)(p_qs)
    g_nonqs = jax.grad(m_nonqs.log_prob)(p_nonqs)

    # Reformat gradients
    gk_nonqs = g_nonqs["log_kernel_param"]
    g_nonqs["log_kernel_param"] = jnp.stack([gk_nonqs[1], 2.0 * gk_nonqs[0]])

    assert_allclose(g_qs["log_kernel_param"], g_nonqs["log_kernel_param"])
    assert_allclose(g_qs["mean"], g_nonqs["mean"])
    assert_allclose(g_qs["log_jitter"], g_nonqs["log_jitter"])

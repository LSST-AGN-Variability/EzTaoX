import equinox as eqx
import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from jax import numpy as jnp
from tinygp.test_utils import assert_allclose

from eztaox.fitter import random_search
from eztaox.kernels import quasisep
from eztaox.kernels.quasisep import LaguerreSeries, Quasisep, _Laguerre
from eztaox.models import UniVarModel
from eztaox.simulator import UniVarSim


class LaguerreKernel(_Laguerre, Quasisep):
    """Single Laguerre basis function as a quasisep kernel. For testing."""

    scale: jax.Array | float
    order: int = eqx.field(static=True)

    def __init__(self, order: int, scale: jax.Array | float):
        """Initialize with order and scale."""
        self.order = order
        self.scale = scale

    def observation_model(self, X: jax.Array) -> jax.Array:
        """Return the observation model vector."""
        del X
        return super().observation_model()

    def transition_matrix(self, X1: jax.Array, X2: jax.Array) -> jax.Array:
        """Return the state transition matrix between X1 and X2."""
        return super().transition_matrix(X2 - X1)


def laguerre_eval(x, *, order, scale):
    return (
        np.sqrt(2.0 / scale)
        * np.exp(-x / scale)
        * np.polynomial.laguerre.lagval(2.0 / scale * x, [0.0] * order + [1.0])
    )


def test_laguerre_evaluate() -> None:
    """Test Laguerre kernel values."""
    test_x = np.r_[0, np.geomspace(1e-2, 1e2, 10)]
    scale = 10.0
    for order in [0, 1, 2, 3]:
        k = LaguerreKernel(order=order, scale=scale)
        actual = [k.evaluate(jnp.array(x), jnp.array(0.0)) for x in test_x]
        expected = laguerre_eval(test_x, order=order, scale=scale)
        assert_allclose(
            np.asarray(actual), expected, err_msg=f"Failed for order {order}"
        )


def test_laguerre_inv() -> None:
    """Test inverse Laguerre kernel matrix."""
    test_x = np.r_[0, np.geomspace(1e-2, 1e2, 10)]
    dx_matrix = jnp.abs(test_x[:, None] - test_x[None, :])
    scale = 10.0
    for order in [0, 1, 2, 3]:
        k = LaguerreKernel(order=order, scale=scale)
        sim_qsm = k.to_symm_qsm(test_x)
        actual = sim_qsm.inv().to_dense()
        kernel_matrix = laguerre_eval(dx_matrix, order=order, scale=scale)
        expected = np.linalg.inv(kernel_matrix)
        assert_allclose(actual, expected, err_msg=f"Failed for order {order}")


@pytest.mark.parametrize(
    ("kernel", "laguerre_scale", "atol"),
    [
        (quasisep.Exp(scale=10.0, sigma=5.0), 10.0, 1e-9),
        (quasisep.Matern32(scale=10.0, sigma=0.4), 10.0 / np.sqrt(3), 1e-3),
        # (tinygp.kernels.ExpSquared(scale=10.0), 10.0, 1e-2),
    ],
)
def test_laguerre_series(kernel, laguerre_scale, atol) -> None:
    """Test LaguerreSeries approximation of kernels."""
    test_x = np.r_[0, np.geomspace(1e-2, 1e2, 10)]
    dx_matrix = jnp.abs(test_x[:, None] - test_x[None, :])

    laguerre = LaguerreSeries(kernel=kernel, order=10, n_quad=100, scale=laguerre_scale)

    # Check parameter count matches
    kernel_leaves = jax.tree_util.tree_leaves(kernel)
    laguerre_leaves = jax.tree_util.tree_leaves(laguerre)
    assert len(laguerre_leaves) == len(kernel_leaves)

    # Check kernel matrix is positive semi-definite
    kernel_matrix = laguerre.to_symm_qsm(test_x).to_dense()
    eigenvalues = jnp.linalg.eigvalsh(kernel_matrix)
    assert jnp.all(
        eigenvalues >= -1e-10
    ), f"Kernel matrix not PSD: min={eigenvalues.min()}"

    laguerre_eval = jax.jit(jax.vmap(lambda x: laguerre.evaluate(x, jnp.array(0.0))))
    actual_vals = laguerre_eval(test_x)
    kernel_eval = jax.jit(jax.vmap(lambda x: kernel.evaluate(x, jnp.array(0.0))))
    desired_vals = kernel_eval(test_x)
    assert_allclose(jnp.asarray(actual_vals), desired_vals, atol=atol)

    laguerre_inv = jax.jit(lambda x: laguerre.to_symm_qsm(x).inv().to_dense())
    actual_inv = laguerre_inv(test_x)
    desired_inv = jnp.linalg.inv(
        kernel_eval(dx_matrix.flatten()).reshape(dx_matrix.shape)
    )
    assert_allclose(actual_inv, desired_inv, atol=atol)


def simulate(kernel, tau, sigma):
    t = jnp.arange(0.0, 4000.0, 1.0)

    sim = UniVarSim(
        kernel,
        0.01,
        float(t[-1]),
        init_params={"log_kernel_param": jnp.array([jnp.log(tau), jnp.log(sigma)])},
        zero_mean=True,
    )

    return sim.fixed_input_fast(
        t,
        jax.random.PRNGKey(11),
    )


@pytest.mark.parametrize(
    ("kernel_cls", "order", "laguerre_scale_factor", "atol"),
    [
        (quasisep.Exp, 2, 1.0, 1e-7),
        (quasisep.Matern32, 3, 1 / np.sqrt(3), 1e-3),
    ],
)
def test_laguerre_decomposition_kernel_sim(
    kernel_cls, order, laguerre_scale_factor, atol
) -> None:
    """End-to-end test of laguerre decomposition for fitting."""
    tau_true = 412.0
    sigma_true = 0.9

    kernel = kernel_cls(scale=tau_true, sigma=sigma_true)

    expected_t, expected_y = simulate(kernel, tau=tau_true, sigma=sigma_true)

    decomposed_kernel = LaguerreSeries(
        kernel, order=order, n_quad=50, scale=tau_true * laguerre_scale_factor
    )
    actual_t, actual_y = simulate(decomposed_kernel, tau=tau_true, sigma=sigma_true)

    assert_allclose(actual_t, expected_t, atol=1e-10)
    assert_allclose(actual_y, expected_y, atol=atol)


@pytest.mark.parametrize(
    ("kernel_cls", "order", "laguerre_scale_factor", "rtol"),
    [
        (quasisep.Exp, 2, 1.0, 1e-2),
        (quasisep.Matern32, 3, 1 / np.sqrt(3), 1e-2),
    ],
)
def test_laguerre_decomposition_kernel_fit(
    kernel_cls, order, laguerre_scale_factor, rtol
) -> None:
    def init_sampler():
        # GP kernel param
        log_drw_scale = numpyro.sample(
            "drw_scale", dist.Uniform(jnp.log(0.01), jnp.log(1000))
        )
        log_drw_sigma = numpyro.sample(
            "drw_sigma", dist.Uniform(jnp.log(0.01), jnp.log(10))
        )
        log_kernel_param = jnp.stack([log_drw_scale, log_drw_sigma])
        numpyro.deterministic("log_kernel_param", log_kernel_param)

        sample_params = {"log_kernel_param": log_kernel_param}
        return sample_params

    fit_key = jax.random.PRNGKey(1)
    n_sample = 10_000
    n_best = 10  # it seems like this number needs to be high

    true_tau = 78.55
    true_sigma = 5.0
    orig_kernel = kernel_cls(scale=true_tau, sigma=true_sigma)
    t, y = simulate(orig_kernel, tau=true_tau, sigma=true_sigma)
    yerr = jnp.full_like(t, 1e-6)

    orig_model = UniVarModel(t, y, yerr, orig_kernel, zero_mean=True)
    orig_ln_params, orig_ll = random_search(
        orig_model, init_sampler, fit_key, n_sample, n_best
    )
    orig_params = jnp.exp(orig_ln_params["log_kernel_param"])
    assert_allclose(orig_params, jnp.array([true_tau, true_sigma]), rtol=1e-1)

    decomposed_kernel = LaguerreSeries(
        orig_kernel, order=order, n_quad=50, scale=laguerre_scale_factor * true_tau
    )
    decomposed_model = UniVarModel(t, y, yerr, decomposed_kernel, zero_mean=True)
    decomposed_ln_params, decomposed_ll = random_search(
        decomposed_model, init_sampler, fit_key, n_sample, n_best
    )
    decomposed_params = jnp.exp(decomposed_ln_params["log_kernel_param"])
    assert_allclose(decomposed_params, jnp.array([true_tau, true_sigma]), rtol=1e-1)

    assert_allclose(orig_ll, decomposed_ll, rtol=rtol)
    assert_allclose(orig_params, decomposed_params, rtol=rtol)

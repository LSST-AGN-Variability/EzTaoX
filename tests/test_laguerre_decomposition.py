import equinox as eqx
import jax
import numpy as np
import pytest
import tinygp
from jax import numpy as jnp
from tinygp.test_utils import assert_allclose

from eztaox.kernels import quasisep
from eztaox.kernels.quasisep import LaguerreSeries, Quasisep, _Laguerre
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
    ("kernel", "atol"),
    [
        (quasisep.Exp(scale=10.0, sigma=1.0), 1e-9),
        (quasisep.Matern32(scale=10.0, sigma=1.0), 1e-6),
        (tinygp.kernels.ExpSquared(scale=10.0), 1e-2),
    ],
)
def test_laguerre_series(kernel, atol) -> None:
    """Test LaguerreSeries approximation of kernels."""
    test_x = np.r_[0, np.geomspace(1e-2, 1e2, 10)]

    laguerre = LaguerreSeries(kernel=kernel, order=10, n_quad=100)

    # Check parameter count matches
    kernel_leaves = jax.tree_util.tree_leaves(kernel)
    laguerre_leaves = jax.tree_util.tree_leaves(laguerre)
    assert len(laguerre_leaves) == len(kernel_leaves)

    laguerre_eval = jax.jit(jax.vmap(lambda x: laguerre.evaluate(x, jnp.array(0.0))))
    actual = laguerre_eval(test_x)

    kernel_eval = jax.jit(jax.vmap(lambda x: kernel.evaluate(x, jnp.array(0.0))))
    desired = kernel_eval(test_x)

    assert_allclose(jnp.asarray(actual), desired, atol=atol)


@pytest.mark.parametrize(
    ("kernel_cls", "order", "atol"),
    [
        (quasisep.Exp, 2, 1e-7),
        (quasisep.Matern32, 3, 1e-3),
    ],
)
def test_laguerre_decomposition_kernel_e2e(kernel_cls, order, atol) -> None:
    """End-to-end test of laguerre decomposition for fitting."""
    tau_true = 412.0
    sigma_true = 0.9

    t = jnp.arange(0.0, 4000.0, 1.0)

    kernel = kernel_cls(scale=tau_true, sigma=sigma_true)

    expected_t, expected_y = UniVarSim(
        kernel,
        0.01,
        float(t[-1]),
        init_params={
            "log_kernel_param": jnp.array([jnp.log(tau_true), jnp.log(sigma_true)])
        },
        zero_mean=True,
    ).fixed_input_fast(
        t,
        jax.random.PRNGKey(11),
    )

    decomposed_kernel = LaguerreSeries(kernel, order=order, n_quad=50)
    actual_t, actual_y = UniVarSim(
        decomposed_kernel,
        0.01,
        float(t[-1]),
        init_params={
            "log_kernel_param": jnp.array([jnp.log(tau_true), jnp.log(sigma_true)])
        },
        zero_mean=True,
    ).fixed_input_fast(
        t,
        jax.random.PRNGKey(11),
    )

    assert_allclose(actual_t, expected_t, atol=1e-10)
    assert_allclose(actual_y, expected_y, atol=atol)

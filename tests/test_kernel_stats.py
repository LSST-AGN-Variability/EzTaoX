"""
Test for second-order statistics of GP kernels: autocorrelation function
(ACF), structure function (SF), and power-spectral density (PSD).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tinygp
from eztao.carma import carma_acf, carma_psd, carma_sf, drw_acf, drw_psd, drw_sf
import equinox as eqx
from eztaox.kernels.quasisep import _Laguerre, LaguerreSeries, Quasisep


class LaguerreKernel(_Laguerre, Quasisep):
    """Single Laguerre basis function as a quasisep kernel. For testing."""

    scale: jax.Array | float
    order: int = eqx.field(static=True)

    def __init__(self, order: int, scale: jax.Array | float):
        self.order = order
        self.scale = scale

    def observation_model(self, X: jax.Array) -> jax.Array:
        del X
        return super().observation_model()

    def transition_matrix(self, X1: jax.Array, X2: jax.Array) -> jax.Array:
        return super().transition_matrix(X2 - X1)
from tinygp.test_utils import assert_allclose

from eztaox.simulator import UniVarSim
from eztaox.kernel_stat2 import carma_acf as carma_acf_local
from eztaox.kernel_stat2 import carma_sf as carma_sf_local
from eztaox.kernel_stat2 import gpStat2
from eztaox.kernels import quasisep

jax.config.update("jax_enable_x64", True)


def test_drw() -> None:
    """
    Test the DRW ACF, SF, and PSD.
    """
    tau = 100.0
    amp = 0.1
    ts = np.linspace(0, 1000, 100)
    fs = np.logspace(-5, 5, 100)

    # gpStat2
    drw = quasisep.Exp(scale=tau, sigma=amp)
    drw_stat2 = gpStat2(drw)

    assert_allclose(drw_acf(tau)(ts), drw_stat2.acf(ts, jnp.array([tau, amp])))
    assert_allclose(drw_sf(amp, tau)(ts), drw_stat2.sf(ts, jnp.array([tau, amp])))
    assert_allclose(drw_psd(amp, tau)(fs), drw_stat2.psd(fs))


def test_carma20() -> None:
    """
    Test the CARMA(2,0) ACF and SF.
    """

    ts = np.linspace(0.001, 1000, 100)
    fs = np.logspace(-5, 5, 100)

    ## CARMA(2,0)
    ar20_1, ma20_1 = np.array([2.0, 1.1]), np.array([0.5])
    ar20_2, ma20_2 = np.array([2.0, 0.8]), np.array([2.0])

    # from GP
    c20_k1 = quasisep.CARMA(alpha=ar20_1[::-1], beta=ma20_1)
    c20_k2 = quasisep.CARMA(alpha=ar20_2[::-1], beta=ma20_2)
    c20_stat2_1 = gpStat2(c20_k1)
    c20_stat2_2 = gpStat2(c20_k2)

    # from eztao
    eztao_acf1 = carma_acf(ar20_1, ma20_1)
    eztao_acf2 = carma_acf(ar20_2, ma20_2)
    eztao_sf1 = carma_sf(ar20_1, ma20_1)
    eztao_sf2 = carma_sf(ar20_2, ma20_2)
    eztao_psd1 = carma_psd(ar20_1, ma20_1)
    eztao_psd2 = carma_psd(ar20_2, ma20_2)

    # ---------- ACF ----------
    # eztao vs. eztaoX
    assert_allclose(eztao_acf1(ts), carma_acf_local(ts, ar20_1[::-1], ma20_1))
    assert_allclose(eztao_acf2(ts), carma_acf_local(ts, ar20_2[::-1], ma20_2))
    # eztao vs GP
    assert_allclose(
        c20_stat2_1.acf(ts, jnp.concat([ar20_1[::-1], ma20_1])),
        eztao_acf1(ts),
    )
    assert_allclose(
        c20_stat2_2.acf(ts, jnp.concat([ar20_2[::-1], ma20_2])),
        eztao_acf2(ts),
    )

    # ---------- SF ----------
    # eztao vs. eztaoX
    assert_allclose(eztao_sf1(ts), carma_sf_local(ts, ar20_1[::-1], ma20_1))
    assert_allclose(eztao_sf2(ts), carma_sf_local(ts, ar20_2[::-1], ma20_2))
    # eztao vs GP
    assert_allclose(
        c20_stat2_1.sf(ts, jnp.concat([ar20_1[::-1], ma20_1])),
        eztao_sf1(ts),
    )
    assert_allclose(
        c20_stat2_2.sf(ts, jnp.concat([ar20_2[::-1], ma20_2])),
        eztao_sf2(ts),
    )

    # ---------- PSD ----------
    # eztao vs. GP
    assert_allclose(c20_stat2_1.psd(fs), eztao_psd1(fs))
    assert_allclose(
        c20_stat2_2.psd(fs, jnp.concat([ar20_2[::-1], ma20_2])), eztao_psd2(fs)
    )


def test_carma21() -> None:
    """
    Test the CARMA(2,1) ACF and SF.
    """

    ts = np.linspace(0.0001, 1000, 100)
    fs = np.logspace(-5, 5, 100)

    ## CARMA(2,1)
    ar21_1, ma21_1 = np.array([2.0, 1.2]), np.array([1.0, 2.0])
    ar21_2, ma21_2 = np.array([2.0, 0.8]), np.array([1.0, 0.5])

    # from GP
    c21_k1 = quasisep.CARMA(alpha=ar21_1[::-1], beta=ma21_1)
    c21_k2 = quasisep.CARMA(alpha=ar21_2[::-1], beta=ma21_2)
    c21_stat2_1 = gpStat2(c21_k1)
    c21_stat2_2 = gpStat2(c21_k2)

    # from eztao
    eztao_acf1 = carma_acf(ar21_1, ma21_1)
    eztao_acf2 = carma_acf(ar21_2, ma21_2)
    eztao_sf1 = carma_sf(ar21_1, ma21_1)
    eztao_sf2 = carma_sf(ar21_2, ma21_2)
    eztao_psd1 = carma_psd(ar21_1, ma21_1)
    eztao_psd2 = carma_psd(ar21_2, ma21_2)

    # ---------- ACF ----------
    # eztao vs. eztaoX
    assert_allclose(eztao_acf1(ts), carma_acf_local(ts, ar21_1[::-1], ma21_1))
    assert_allclose(eztao_acf2(ts), carma_acf_local(ts, ar21_2[::-1], ma21_2))
    # eztao vs GP
    assert_allclose(
        c21_stat2_1.acf(ts, jnp.concat([ar21_1[::-1], ma21_1])),
        eztao_acf1(ts),
    )
    assert_allclose(
        c21_stat2_2.acf(ts, jnp.concat([ar21_2[::-1], ma21_2])),
        eztao_acf2(ts),
    )

    # ---------- SF ----------
    # eztao vs. eztaoX
    assert_allclose(eztao_sf1(ts), carma_sf_local(ts, ar21_1[::-1], ma21_1))
    assert_allclose(eztao_sf2(ts), carma_sf_local(ts, ar21_2[::-1], ma21_2))
    # eztao vs GP
    assert_allclose(
        c21_stat2_1.sf(ts, jnp.concat([ar21_1[::-1], ma21_1])),
        eztao_sf1(ts),
    )
    assert_allclose(
        c21_stat2_2.sf(ts, jnp.concat([ar21_2[::-1], ma21_2])),
        eztao_sf2(ts),
    )

    # ---------- PSD ----------
    # eztao vs. GP
    assert_allclose(c21_stat2_1.psd(fs), eztao_psd1(fs))
    assert_allclose(
        c21_stat2_2.psd(fs, jnp.concat([ar21_2[::-1], ma21_2])), eztao_psd2(fs)
    )


def test_carma30() -> None:
    """
    Test the CARMA(3,0) ACF and SF.
    """

    ts = np.linspace(0.0001, 1000, 100)
    fs = np.logspace(-5, 5, 100)

    # CARMA(3,0)
    ar30_1, ma30_1 = np.array([3.0, 2.8, 0.8]), np.array([1.0])

    # from GP
    c30_k1 = quasisep.CARMA(alpha=ar30_1[::-1], beta=ma30_1)
    # c30_k2 = quasisep.CARMA(alpha=ar30_2[::-1], beta=ma30_2)
    c30_stat2_1 = gpStat2(c30_k1)
    # c30_stat2_2 = gpStat2(c30_k2)

    # from eztao
    eztao_acf1 = carma_acf(ar30_1, ma30_1)
    # eztao_acf2 = carma_acf(ar30_2, ma30_2)
    eztao_sf1 = carma_sf(ar30_1, ma30_1)
    # eztao_sf2 = carma_sf(ar30_2, ma30_2)
    eztao_psd1 = carma_psd(ar30_1, ma30_1)
    # eztao_psd2 = carma_psd(ar30_2, ma30_2)

    # ---------- ACF ----------
    # eztao vs. eztaoX
    assert_allclose(eztao_acf1(ts), carma_acf_local(ts, ar30_1[::-1], ma30_1))
    # assert_allclose(eztao_acf2(ts), carma_acf_local(ts, ar30_2[::-1], ma30_2))
    # eztao vs GP
    assert_allclose(
        c30_stat2_1.acf(ts, jnp.concat([ar30_1[::-1], ma30_1])),
        eztao_acf1(ts),
    )
    # assert_allclose(
    #     c30_stat2_2.acf(ts, jnp.concat([ar30_2[::-1], ma30_2])),
    #     eztao_acf2(ts),
    # )

    # ---------- SF ----------
    # eztao vs. eztaoX
    assert_allclose(eztao_sf1(ts), carma_sf_local(ts, ar30_1[::-1], ma30_1))
    # assert_allclose(eztao_sf2(ts), carma_sf_local(ts, ar30_2[::-1], ma30_2))
    # eztao vs GP
    assert_allclose(
        c30_stat2_1.sf(ts, jnp.concat([ar30_1[::-1], ma30_1])),
        eztao_sf1(ts),
    )
    # assert_allclose(
    #     c30_stat2_2.sf(ts, jnp.concat([ar30_2[::-1], ma30_2])),
    #     eztao_sf2(ts),
    # )

    # ---------- PSD ----------
    # eztao vs. GP
    assert_allclose(c30_stat2_1.psd(fs), eztao_psd1(fs))
    # assert_allclose(
    # c30_stat2_2.psd(fs, jnp.concat([ar30_2[::-1], ma30_2])), eztao_psd2(fs)
    # )

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
        assert_allclose(np.asarray(actual), expected, err_msg=f"Failed for order {order}")


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


def _exp(x, scale):
    return jnp.exp(-jnp.abs(x) / scale)


def _exp_squared(x, scale):
    return jnp.exp(-0.5 * jnp.square(x / scale))


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


@pytest.mark.parametrize(("kernel_cls", "order", "atol"), [
    (quasisep.Exp, 2, 1e-7),
    (quasisep.Matern32, 3, 1e-3),
])
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
        init_params={"log_kernel_param": jnp.array([jnp.log(tau_true), jnp.log(sigma_true)])},
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
        init_params={"log_kernel_param": jnp.array([jnp.log(tau_true), jnp.log(sigma_true)])},
        zero_mean=True,
    ).fixed_input_fast(
        t,
        jax.random.PRNGKey(11),
    )

    assert_allclose(actual_t, expected_t, atol=1e-10)
    assert_allclose(actual_y, expected_y, atol=atol)

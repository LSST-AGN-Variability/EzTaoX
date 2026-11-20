"""
Test for second-order statistics of GP kernels: autocorrelation function
(ACF), structure function (SF), and power-spectral density (PSD).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from eztao.carma import carma_acf, carma_psd, carma_sf, drw_acf, drw_psd, drw_sf
from tinygp.test_utils import assert_allclose

from eztaox.kernel_stat2 import gpStat2
from eztaox.kernels import quasisep

jax.config.update("jax_enable_x64", True)

# kernels whose psd must integrate to the kernel variance
PSD_KERNELS = [
    pytest.param(quasisep.Exp(scale=10.0, sigma=1.3), id="exp"),
    pytest.param(quasisep.Matern32(scale=10.0, sigma=1.3), id="matern32"),
    pytest.param(quasisep.Matern52(scale=10.0, sigma=1.3), id="matern52"),
    pytest.param(quasisep.SHO(omega=0.5, quality=3.0, sigma=1.3), id="sho"),
    pytest.param(quasisep.Celerite(a=1.5, b=0.3, c=0.1, d=0.5), id="celerite"),
    pytest.param(
        quasisep.Lorentzian(omega=0.5, quality=5.0, sigma=1.3), id="lorentzian_highQ"
    ),
    pytest.param(
        quasisep.Lorentzian(omega=0.5, quality=0.6, sigma=1.3), id="lorentzian_lowQ"
    ),
    pytest.param(quasisep.Cosine(scale=10.0, sigma=1.3, psd_width=0.01), id="cosine"),
    pytest.param(
        quasisep.CARMA(alpha=jnp.array([1.2, 2.0]), beta=jnp.array([1.0, 2.0])),
        id="carma21",
    ),
    pytest.param(
        quasisep.Exp(scale=10.0, sigma=1.3) + quasisep.Matern32(scale=3.0, sigma=0.7),
        id="sum",
    ),
    pytest.param(2.5 * quasisep.Exp(scale=10.0, sigma=1.3), id="scale"),
]


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
    # eztao vs. gpStat2
    assert_allclose(
        c20_stat2_1.acf(ts, jnp.concat([ar20_1[::-1], ma20_1])),
        eztao_acf1(ts),
    )
    assert_allclose(
        c20_stat2_2.acf(ts, jnp.concat([ar20_2[::-1], ma20_2])),
        eztao_acf2(ts),
    )

    # ---------- SF ----------
    # eztao vs. gpStat2
    assert_allclose(
        c20_stat2_1.sf(ts, jnp.concat([ar20_1[::-1], ma20_1])),
        eztao_sf1(ts),
    )
    assert_allclose(
        c20_stat2_2.sf(ts, jnp.concat([ar20_2[::-1], ma20_2])),
        eztao_sf2(ts),
    )

    # ---------- PSD ----------
    # eztao vs. gpStat2
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
    # eztao vs. gpStat2
    assert_allclose(
        c21_stat2_1.acf(ts, jnp.concat([ar21_1[::-1], ma21_1])),
        eztao_acf1(ts),
    )
    assert_allclose(
        c21_stat2_2.acf(ts, jnp.concat([ar21_2[::-1], ma21_2])),
        eztao_acf2(ts),
    )

    # ---------- SF ----------
    # eztao vs. gpStat2
    assert_allclose(
        c21_stat2_1.sf(ts, jnp.concat([ar21_1[::-1], ma21_1])),
        eztao_sf1(ts),
    )
    assert_allclose(
        c21_stat2_2.sf(ts, jnp.concat([ar21_2[::-1], ma21_2])),
        eztao_sf2(ts),
    )

    # ---------- PSD ----------
    # eztao vs. gpStat2
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
    ar30_2, ma30_2 = np.array([3.0, 3.2, 1.2]), np.array([1.0])

    # from GP
    c30_k1 = quasisep.CARMA(alpha=ar30_1[::-1], beta=ma30_1)
    c30_k2 = quasisep.CARMA(alpha=ar30_2[::-1], beta=ma30_2)
    c30_stat2_1 = gpStat2(c30_k1)
    c30_stat2_2 = gpStat2(c30_k2)

    # from eztao
    eztao_acf1 = carma_acf(ar30_1, ma30_1)
    eztao_acf2 = carma_acf(ar30_2, ma30_2)
    eztao_sf1 = carma_sf(ar30_1, ma30_1)
    eztao_sf2 = carma_sf(ar30_2, ma30_2)
    eztao_psd1 = carma_psd(ar30_1, ma30_1)
    eztao_psd2 = carma_psd(ar30_2, ma30_2)

    # ---------- ACF ----------
    # eztao vs. gpStat2
    assert_allclose(
        c30_stat2_1.acf(ts, jnp.concat([ar30_1[::-1], ma30_1])),
        eztao_acf1(ts),
    )
    assert_allclose(
        c30_stat2_2.acf(ts, jnp.concat([ar30_2[::-1], ma30_2])),
        eztao_acf2(ts),
    )

    # ---------- SF ----------
    # eztao vs. gpStat2
    assert_allclose(
        c30_stat2_1.sf(ts, jnp.concat([ar30_1[::-1], ma30_1])),
        eztao_sf1(ts),
    )
    assert_allclose(
        c30_stat2_2.sf(ts, jnp.concat([ar30_2[::-1], ma30_2])),
        eztao_sf2(ts),
    )

    # ---------- PSD ----------
    # eztao vs. gpStat2
    assert_allclose(c30_stat2_1.psd(fs), eztao_psd1(fs))
    assert_allclose(
        c30_stat2_2.psd(fs, jnp.concat([ar30_2[::-1], ma30_2])), eztao_psd2(fs)
    )


def test_stat2_vmap() -> None:
    """
    The jitted ACF/SF/PSD must support vmap over a batch of kernel parameters,
    as done with posterior draws in the docs.
    """
    ts = np.linspace(0.001, 1000, 50)
    fs = np.logspace(-4, 2, 50)

    ar, ma = np.array([2.0, 1.1]), np.array([0.5])
    stat2 = gpStat2(quasisep.CARMA(alpha=ar[::-1], beta=ma))

    params = jnp.stack([stat2.kernel_params] * 3)
    batched_acf = jax.vmap(stat2.acf, in_axes=(None, 0))(ts, params)
    batched_sf = jax.vmap(stat2.sf, in_axes=(None, 0))(ts, params)
    batched_psd = jax.vmap(stat2.psd, in_axes=(None, 0))(fs, params)

    assert batched_acf.shape == (3, ts.size)
    assert batched_sf.shape == (3, ts.size)
    assert batched_psd.shape == (3, fs.size)
    for i in range(3):
        assert_allclose(batched_acf[i], stat2.acf(ts))
        assert_allclose(batched_sf[i], stat2.sf(ts))
        assert_allclose(batched_psd[i], stat2.psd(fs))


@pytest.mark.parametrize("kernel", PSD_KERNELS)
def test_psd_normalization(kernel: quasisep.Quasisep) -> None:
    """
    Every PSD follows the two-sided Kelly et al. (2014) convention: it is even
    in frequency, so twice its integral over f >= 0 must equal the variance.
    """
    fs = np.logspace(-6, 4, 400_001)
    psd = np.asarray(gpStat2(kernel).psd(fs))
    variance = kernel.evaluate_diag(jnp.array(0.0))

    assert np.all(psd >= 0)
    assert_allclose(2 * np.trapezoid(psd, fs), variance, rtol=2e-3)


def test_lorentzian_matches_celerite() -> None:
    """
    A Lorentzian is a Celerite kernel with b = 0, so the ACF and PSD must agree.
    """
    omega, quality, sigma = 0.5, 0.6, 1.3
    ts = jnp.linspace(0.0, 50.0, 101)
    fs = np.logspace(-4, 2, 101)

    lor = gpStat2(quasisep.Lorentzian(omega=omega, quality=quality, sigma=sigma))
    cel = gpStat2(
        quasisep.Celerite(a=sigma**2, b=0.0, c=omega / (2 * quality), d=omega)
    )

    assert_allclose(lor.acf(ts), cel.acf(ts))
    assert_allclose(lor.psd(fs), cel.psd(fs))


def test_psd_not_implemented() -> None:
    """
    Kernels without a PSD raise instead of returning an exception class.
    """
    product = quasisep.Exp(scale=10.0, sigma=1.0) * quasisep.Exp(scale=5.0, sigma=1.0)
    with pytest.raises(NotImplementedError):
        gpStat2(product).psd(np.array([0.1]))

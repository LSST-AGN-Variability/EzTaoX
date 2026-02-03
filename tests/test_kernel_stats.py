"""
Test for second-order statistics of GP kernels: autocorrelation function
(ACF), structure function (SF), and power-spectral density (PSD).
"""

import jax
import jax.numpy as jnp
import numpy as np
from eztao.carma import carma_acf, carma_psd, carma_sf, drw_acf, drw_psd, drw_sf
from eztaox.kernel_stat2 import (
    carma_acf as carma_acf_local,
)
from eztaox.kernel_stat2 import (
    carma_sf as carma_sf_local,
)
from eztaox.kernel_stat2 import (
    gpStat2,
)
from eztaox.kernels import quasisep
from tinygp.test_utils import assert_allclose

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

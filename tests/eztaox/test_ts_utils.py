"""Test the time series utilities."""

import numpy as np

from eztaox.ts_utils import downsampleByTime, formatlc


def test_downsampleByTime() -> None:  # noqa: N802
    """Test the time series downsampling utility."""

    tIn = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    tOut = np.array([0.2, 2.7, 4.5])

    # Downsample
    expected = np.array([0.0, 3.0, 4.0])
    res = np.array(downsampleByTime(tIn, tOut))

    # Verify output
    assert np.allclose(expected, res)


def test_formatlc() -> None:
    """Test the light curve formatting utility."""

    ts, ys, yerrs = {}, {}, {}
    band_order = {"g": 0, "r": 1, "i": 2}
    for band in band_order:
        ts[band] = np.array([1.0, 2.0, 3.0])
        ys[band] = np.array([-0.2, 0.7, 0.1])
        yerrs[band] = np.array([0.08, 0.1, 0.03])

    # Format light curves
    X, y, yerr = formatlc(ts, ys, yerrs, band_order)

    # Verify outputs
    expected_X = (
        np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
        np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
    )
    assert np.allclose(X[0], expected_X[0])
    assert np.allclose(X[1], expected_X[1])

    expected_y = np.array([-0.2, 0.7, 0.1, -0.2, 0.7, 0.1, -0.2, 0.7, 0.1])
    assert np.allclose(y, expected_y)

    expected_yerr = np.array([0.08, 0.1, 0.03, 0.08, 0.1, 0.03, 0.08, 0.1, 0.03])
    assert np.allclose(yerr, expected_yerr)

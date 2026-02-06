"""Test the time series utilities."""

import numpy as np

from eztaox.ts_utils import _get_nearest_idx, downsampleByTime, formatlc


def test_get_nearest_idx() -> None:
    """Test the nearest index utility."""

    tIn = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    # Case: Simple, rounds to nearest
    x = 0.1
    expected = 0
    res = int(_get_nearest_idx(tIn, x))
    assert expected == res

    # Case: Value in the middle of two elements (rounds down)
    x = 4.5
    expected = 4
    res = int(_get_nearest_idx(tIn, x))
    assert expected == res

    # Case: Value less than least element (clamps to first index)
    x = -0.1
    expected = 0
    res = int(_get_nearest_idx(tIn, x))
    assert expected == res

    # Case: Value greater than greatest element (clamps to last index)
    x = 42.0
    expected = 5
    res = int(_get_nearest_idx(tIn, x))
    assert expected == res


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
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
    )
    assert np.allclose(X[0], expected_X[0])
    assert np.allclose(X[1], expected_X[1])
    assert X[1][0].dtype == int

    expected_y = np.array([-0.2, 0.7, 0.1, -0.2, 0.7, 0.1, -0.2, 0.7, 0.1])
    assert np.allclose(y, expected_y)

    expected_yerr = np.array([0.08, 0.1, 0.03, 0.08, 0.1, 0.03, 0.08, 0.1, 0.03])
    assert np.allclose(yerr, expected_yerr)

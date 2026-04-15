"""Test the time series utilities."""

import numpy as np

from eztaox.ts_utils import (
    _get_nearest_idx,
    add_noise,
    down_sample_by_time,
    formatlc,
    merge_sort,
)


def test_get_nearest_idx() -> None:
    """Test the nearest index utility."""

    t_in = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    # Case: Simple, rounds to nearest
    x = 0.1
    expected = 0
    res = int(_get_nearest_idx(t_in, x))
    assert expected == res

    # Case: Value in the middle of two elements (rounds down)
    x = 4.5
    expected = 4
    res = int(_get_nearest_idx(t_in, x))
    assert expected == res

    # Case: Value less than least element (clamps to first index)
    x = -0.1
    expected = 0
    res = int(_get_nearest_idx(t_in, x))
    assert expected == res

    # Case: Value greater than greatest element (clamps to last index)
    x = 42.0
    expected = 5
    res = int(_get_nearest_idx(t_in, x))
    assert expected == res


def test_down_sample_by_time() -> None:  # noqa: N802
    """Test the time series downsampling utility."""

    t_in = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    t_out = np.array([0.2, 2.7, 4.5])

    # Downsample
    expected = np.array([0.0, 3.0, 4.0])
    res = np.array(down_sample_by_time(t_in, t_out))

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


def test_addNoise() -> None:  # noqa: N802
    """Test the noise addition utility."""

    import jax
    import jax.numpy as jnp
    from scipy.stats import kstest

    key = jax.random.PRNGKey(0)
    key, k1, k2, k3 = jax.random.split(key, 4)
    y = jax.random.normal(k1, (10000,))
    yerr = jax.random.uniform(k2, (10000,), minval=0.1, maxval=0.5)

    # Add noise
    noisy_y = add_noise(y, yerr, k3)
    # Verify that noise has been added (not equal to original)
    assert not jnp.allclose(noisy_y, y)

    # Verify that the noise is Gaussian
    std_diff = (noisy_y - y) / yerr
    res = kstest(std_diff, "norm", args=(0, 1))

    print(res.pvalue)
    assert res.pvalue > 0.05  # Fail if p-value < 5%


def test_merge_sort() -> None:
    """Test that the merge_sort gives consistent results as jnp.argsort."""

    r = np.random.default_rng(49382)
    t1 = np.sort(r.uniform(0, 10, 10_000))
    t2 = np.sort(r.uniform(0, 10, 2_000))
    t3 = np.sort(r.uniform(0, 10, 100))

    # Test merge_sort
    perm = merge_sort(t1, t2, t3)
    expected_perm = np.argsort(np.concatenate([t1, t2, t3]))

    assert np.array_equal(perm, expected_perm)


def test_merge_sort_requires_index_remap_for_interleaved_inputs() -> None:
    """Test that merge_sort indices must be remapped for observation-order arrays."""

    t = np.array([1.0, 1.1, 2.0, 2.1, 3.0, 3.1])
    band = np.array([0, 1, 0, 1, 0, 1])
    lags = np.array([0.0, 1.5])

    new_t = t - lags[band]
    expected_perm = np.argsort(new_t)

    t_in_bands = [t[band == i] for i in range(2)]
    inds_in_bands = [np.where(band == i)[0] for i in range(2)]
    shifted_t_in_bands = [time - lags[i] for i, time in enumerate(t_in_bands)]

    naive_perm = np.array(merge_sort(*shifted_t_in_bands))
    concat_inds = np.concatenate(inds_in_bands)
    remapped_perm = concat_inds[naive_perm]

    assert not np.array_equal(naive_perm, expected_perm)
    assert np.array_equal(remapped_perm, expected_perm)

"""Utility functions for time series processing."""

import jax
import jax.numpy as jnp
from numpy.typing import NDArray
from tinygp.helpers import JAXArray


@jax.jit
def _get_nearest_idx(tIn, x) -> int:
    """
    Get the index of the nearest value in `tIn` to `x`.

    Args:
        tIn (JAXArray): Array of time values.
        x (float): The value to find the nearest index for.
    """
    return jnp.argmin(jnp.abs(tIn - x))


def downsampleByTime(tIn, tOut) -> JAXArray:  # noqa: N802
    """
    Downsample `tIn` to match the time points in `tOut`.

    Args:
        tIn (JAXArray): Array of time values to be downsampled.
        tOut (JAXArray): Array of target time values.

    Returns:
        JAXArray: Downsampled array of time values.
    """
    return tIn[jax.vmap(_get_nearest_idx, in_axes=(None, 0))(tIn, tOut)]


def formatlc(
    ts: dict[str, NDArray | JAXArray],
    ys: dict[str, NDArray | JAXArray],
    yerrs: dict[str, NDArray | JAXArray],
    band_order: dict[str, int],
) -> tuple[tuple[JAXArray, JAXArray], JAXArray, JAXArray]:
    """Transform light curves in dictionary to EzTaoX friendly format.

    Args:
        ts (dict[str, NDArray  |  JAXArray]): Time stamps for observation in each band.
        ys (dict[str, NDArray  |  JAXArray]): Observed values in each band.
        yerrs (dict[str, NDArray  |  JAXArray]): Uncertainties in observed values for
            each band.
        band_order (dict[str, int]): Mapping of band names to band indices.

    Returns:
        tuple[tuple[JAXArray, JAXArray], JAXArray, JAXArray]: Light curves formatted as
            ((time stamps, band indices), observed values, uncertainties).
    """

    band_keys = band_order.keys()
    tss = jnp.concat([ts[key] for key in band_keys])
    yss = jnp.concat([ys[key] for key in band_keys])
    yerrss = jnp.concat([yerrs[key] for key in band_keys])
    band_idxs = jnp.concat(
        [jnp.ones(len(ts[x])) * band_order[x] for x in band_keys]
    ).astype(int)

    return (tss, band_idxs), yss, yerrss


def add_noise(y: JAXArray, yerr: JAXArray, key: jax.random.PRNGKey) -> JAXArray:
    """
    Add Gaussian noise to a time series given measurement uncertainties.
    JAX-compatible (works with jit/vmap).

    Args:
        y (JAXArray): Input values to which noise will be added.
        yerr (JAXArray): Associated errors for the input values.
        key (jax.random.PRNGKey): Pseudorandom number generator key used to
            draw the noise samples.

    Returns:
        JAXArray: Array of the same shape as ``y`` with additive Gaussian noise applied.
    """
    y = jnp.asarray(y)
    yerr = jnp.asarray(yerr)

    # JAX needs an explicit PRNG key
    noise = jax.random.normal(key, shape=y.shape, dtype=y.dtype) * yerr

    return y + noise

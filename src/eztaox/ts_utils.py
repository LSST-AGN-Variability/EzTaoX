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
        yerrs (dict[str, NDArray  |  JAXArray]): Uncertainties in observed values for each
            band.
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

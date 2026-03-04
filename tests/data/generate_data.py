"""Generate simulated multiband light curve dataset for unit testing."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from tinygp.helpers import JAXArray

import eztaox.kernels.quasisep as ekq
from eztaox.simulator import MultiVarSim

base_key = jr.PRNGKey(0)


def generate_simulated_dataset(
    nBand: int,
    drw_scale: float,
    drw_sigma: float,
    mindt: float,
    maxdt: float,
    npt: int,
    seed: int = 0,
) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
    """Generate simulated multiband light curve dataset.

    Args:
        nBand (int): Number of bands.
        drw_scale (float): DRW kernel scale parameter.
        drw_sigma (float): DRW kernel sigma parameter.
        mindt (float): Minimum time difference.
        maxdt (float): Maximum time difference.
        npt (int): Number of points per band.
        seed (int, optional): Seed number for random number generation. Defaults to 0.

    Returns:
        tuple[tuple[JAXArray, JAXArray], JAXArray]: Simulated input (time, band)
            and noisy observations.
    """
    sim_params = {
        "log_kernel_param": jnp.array([jnp.log(drw_scale), jnp.log(drw_sigma)]),
        "log_amp_scale": jnp.log(jnp.linspace(0.9, 0.4, nBand)),
        "lag": jnp.linspace(1, nBand, nBand) * 2.0,
    }

    drw = ekq.Exp(scale=drw_scale, sigma=drw_sigma)
    s = MultiVarSim(drw, mindt, maxdt, nBand, sim_params, has_lag=True)

    lc_key, random_key = jr.split(jr.fold_in(base_key, seed), 2)
    sim_X, sim_y = s.random(npt * nBand, lc_key, random_key)

    return sim_X, sim_y


# simulate N light curves for unit testing
N = 100
sim_ts, sim_bands, sim_ys = [], [], []
for i in range(N):
    (sim_t, sim_band), sim_y = generate_simulated_dataset(
        nBand=2,
        drw_scale=50.0,
        drw_sigma=0.2,
        mindt=0.01,
        maxdt=5000.0,
        npt=200,
        seed=i,
    )
    sim_ts.append(sim_t)
    sim_bands.append(sim_band)
    sim_ys.append(sim_y)


# save the simulated dataset to a .npz file
np.savez("unit_test_lc.npz", ts=sim_ts, bands=sim_bands, ys=sim_ys)

"""Benchmarks for EzTaoX kernel fitting"""

import jax
import jax.numpy as jnp
import numpy as np
import tinygp
from eztao.carma import DRW_term
from eztao.ts import addNoise, gpSimFull
from tinygp import GaussianProcess, kernels

jax.config.update("jax_enable_x64", True)


@tinygp.helpers.dataclass
class MB(tinygp.kernels.quasisep.Wrapper):
    """GP class"""

    amplitudes: jnp.ndarray
    lags: jnp.ndarray

    def coord_to_sortable(self, X):
        return X[0]

    def observation_model(self, X):
        return self.amplitudes[X[1]] * self.kernel.observation_model(X[0])


### helper functions ###
def amp_transform(params):
    return jnp.insert(jnp.atleast_1d(params["log_amp_delta"]), 0, 0.0)


def lag_transform(X, lags):
    t, band = X
    new_t = t - lags[band]
    inds = jnp.argsort(new_t)
    return (new_t, band), inds


class KernelFittingSuite:
    """Timing benchmarks for various kernel fittings"""

    params = [20, 100, 1_000]

    def setup(self, n):
        ## set up full data
        amps = {"g": 0.35, "i": 0.25}
        taus = {"g": 100, "i": 150}
        snrs = {"g": 5, "i": 3}
        noise_seeds = {"g": 111, "i": 2}

        self.ts, self.ys, self.yerrs = {}, {}, {}
        self.ys_noisy = {}
        seed = 1

        for band in "gi":
            DRW_kernel = DRW_term(np.log(amps[band]), np.log(taus[band]))
            t, y, yerr = gpSimFull(
                DRW_kernel,
                snrs[band],
                365 * 10,
                n,
                lc_seed=seed,
            )

            # add to dict
            self.ts[band] = t
            self.ys[band] = y
            self.yerrs[band] = yerr
            self.ys_noisy[band] = addNoise(
                self.ys[band], self.yerrs[band], seed=noise_seeds[band] + seed
            )

        inds = jnp.argsort(jnp.concatenate((self.ts["g"], self.ts["i"])))
        X = (
            jnp.concatenate((self.ts["g"], self.ts["i"]))[inds][:n],
            jnp.concatenate(
                (
                    jnp.zeros_like(self.ts["g"], dtype=int),
                    jnp.ones_like(self.ts["i"], dtype=int),
                )
            )[inds][:n],
        )
        self.ys_noisy["g"] -= jnp.median(self.ys_noisy["g"])
        self.ys_noisy["i"] -= jnp.median(self.ys_noisy["i"])
        y = jnp.concatenate((self.ys_noisy["g"], self.ys_noisy["i"]))[inds][:n]
        diag = jnp.concatenate((self.yerrs["g"], self.yerrs["i"]))[inds][:n] ** 2

        self.params = {
            "log_kernel_param": jnp.log(jnp.array([100, 0.1])),
            "log_amp_delta": jnp.log(0.6),
            "lag": jnp.array(10),
        }

        self.X = X
        self.y = y
        self.diag = diag

        return

    # def time_run_setup(self, n):
    #    return self

    def time_run_drw_fitting(self, n):
        @jax.jit
        def loss(params):
            kernel = kernels.quasisep.Exp(*jnp.exp(params["log_kernel_param"]))
            gp = GaussianProcess(
                kernel,
                self.X[0],
                diag=self.diag,
                mean=0.0,
                assume_sorted=True,
            )
            return -gp.log_probability(self.y)

        loss(self.params).block_until_ready()

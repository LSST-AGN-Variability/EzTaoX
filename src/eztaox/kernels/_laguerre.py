"""Laguerre series approximation of stationary kernels.

.. warning::
    This is experimental functionality and is not covered by automated tests.

Provides :class:`LaguerreSeries`, which wraps any stationary kernel and
approximates its autocovariance using a truncated Laguerre series expansion,
enabling O(N) GP computations.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from tinygp.kernels import Kernel

from eztaox.kernels.eqx_utils import find_param_by_name
from eztaox.kernels.quasisep import Quasisep


class _Laguerre:
    """Laguerre basis function math helper.

    .. warning::
        This is experimental functionality and is not covered by automated tests.

    Not an equinox module.
    """

    def __init__(self, order: int, scale: jax.Array | float):
        self.order = order
        self.scale = scale

    def design_matrix(self) -> jax.Array:
        p = 1.0 / self.scale
        N = self.order + 1
        diag = -p * jnp.ones(N)
        sub_diag = jnp.ones(N - 1)
        return jnp.diag(diag) + jnp.diag(sub_diag, k=-1)

    def stationary_covariance(self) -> jax.Array:
        p = 1.0 / self.scale
        m = self.order
        N = m + 1

        k_indices = jnp.arange(N)
        prefactor = jnp.sqrt(2 * p)

        log_binom = (
            jax.scipy.special.gammaln(m + 1)
            - jax.scipy.special.gammaln(k_indices + 1)
            - jax.scipy.special.gammaln(m - k_indices + 1)
        )
        binom = jnp.exp(log_binom)
        b = prefactor * binom * ((-2 * p) ** k_indices)

        P = jnp.eye(N)
        P = P.at[:, 0].set(b)
        P = P.at[0, :].set(b)
        P = P.at[0, 0].set(b[0])

        return P

    def observation_model(self) -> jax.Array:
        return jnp.concatenate([jnp.ones(1), jnp.zeros(self.order)])

    def transition_matrix(self, dt: jax.Array) -> jax.Array:
        p = 1.0 / self.scale
        N = self.order + 1

        k = jnp.arange(N)
        log_fact = jax.scipy.special.gammaln(k + 1)
        safe_dt = jnp.where(dt == 0, 1.0, dt)
        powers = jnp.exp(k * jnp.log(jnp.abs(safe_dt)) - log_fact)
        powers = jnp.where(dt == 0, jnp.where(k == 0, 1.0, 0.0), powers)

        i, j = jnp.indices((N, N))
        mask = i >= j
        coeffs = jnp.where(mask, powers[i - j], 0.0)

        return jnp.exp(-p * dt) * coeffs


class LaguerreSeries(Quasisep):
    """Laguerre series approximation of a stationary kernel.

    .. warning::
        This is experimental functionality and is not covered by automated tests.

    Wraps a kernel and approximates its autocovariance using a truncated
    Laguerre series expansion, enabling O(N) GP computations. The coefficients
    are derived from the wrapped kernel's parameters at construction time.

    Args:
        kernel: The kernel to approximate (must have a `scale` attribute).
        order: Maximum Laguerre polynomial order.
        n_quad: Number of quadrature points for coefficient computation.
    """

    kernel: Kernel
    order: int = eqx.field(static=True)
    n_quad: int = eqx.field(static=True)

    def __post_init__(self):
        if find_param_by_name(self.kernel, "scale") is None:
            raise ValueError("Kernel must have a 'scale' parameter.")

    @property
    def _vmap_func(self) -> Callable[[jax.Array], jax.Array]:
        return jax.vmap(lambda x: self.kernel.evaluate(x, jnp.array(0.0)))

    def _quadrature(self) -> tuple[NDArray, NDArray]:
        """Get quadrature nodes and weights."""
        return np.polynomial.laguerre.laggauss(self.n_quad)

    def _laguerre_vals(self, nodes: NDArray) -> NDArray:
        """Evaluate Laguerre polynomials at 2*nodes for orders 0..order."""
        return np.polynomial.laguerre.lagval(2 * nodes, np.eye(self.order + 1)).T

    def _scale(self):
        nodes, _ = self._quadrature()
        kernel_scales = find_param_by_name(self.kernel, "scale")
        kernel_scale = sum(kernel_scales) / len(kernel_scales)
        x_nodes = nodes * kernel_scale
        ln_f_vals = jnp.log(self._vmap_func(x_nodes))
        fit_scale = -jnp.sum(jnp.square(x_nodes)) / jnp.sum(ln_f_vals * x_nodes)
        # Clip for bad fits
        fit_scale = jnp.clip(fit_scale, 1e-2 * kernel_scale, 1e2 * kernel_scale)
        return fit_scale

    def _coeffs(self) -> jax.Array:
        """Compute Laguerre coefficients from the wrapped kernel."""
        nodes, weights = self._quadrature()
        laguerre_vals = self._laguerre_vals(nodes)
        laguerre_scale = self._scale()

        x_nodes = nodes * laguerre_scale
        f_vals = self._vmap_func(x_nodes)

        prefactor = jnp.sqrt(2.0 * laguerre_scale)
        weighted_signal = weights * f_vals
        return prefactor * (weighted_signal @ laguerre_vals)

    def _basis(self) -> list[_Laguerre]:
        """Create Laguerre basis functions for each order."""
        laguerre_scale = self._scale()
        return [_Laguerre(order=i, scale=laguerre_scale) for i in range(self.order + 1)]

    def design_matrix(self) -> jax.Array:
        """Block diagonal of individual design matrices."""
        basis = self._basis()
        return jax.scipy.linalg.block_diag(*[b.design_matrix() for b in basis])

    def stationary_covariance(self) -> jax.Array:
        """Block diagonal of scaled stationary covariances."""
        basis = self._basis()
        return jax.scipy.linalg.block_diag(
            *(
                c * b.stationary_covariance()
                for c, b in zip(self._coeffs(), basis, strict=True)
            )
        )

    def observation_model(self, X: jax.Array) -> jax.Array:
        """Concatenation of observation models."""
        del X
        basis = self._basis()
        return jnp.concatenate([b.observation_model() for b in basis])

    def transition_matrix(self, X1: jax.Array, X2: jax.Array) -> jax.Array:
        """Block diagonal of transition matrices."""
        dt = X2 - X1
        basis = self._basis()
        return jax.scipy.linalg.block_diag(*[b.transition_matrix(dt) for b in basis])

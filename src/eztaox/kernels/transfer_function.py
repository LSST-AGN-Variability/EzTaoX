"""Transfer functions"""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import tinygp
from jax import numpy as jnp
from tinygp.helpers import JAXArray

from eztaox.kernels.eqx_utils import find_param_by_name
from eztaox.kernels.quasisep import Quasisep


class TransferFunction(eqx.Module):
    """Base class for transfer functions :math:`\\Psi(\\Delta t)`."""

    width: float
    shift: float = eqx.field(default=0.0, static=True)

    @abstractmethod
    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """Evaluate the transfer function at two points."""
        del X1, X2
        raise NotImplementedError


class GaussianTransferFunction(TransferFunction):
    """Gaussian transfer function: :math:`\\propto e^{-((\\Delta t-\\Delta t_0)/w)^2}`.

    where :math:`\\Delta t_0=\\mathrm{shift}`.
    The unity-normalization coefficient is:
    :math:`\\frac{1}{\\sqrt{\\pi}w}`.
    """

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """Evaluate the normalized transfer function at two points.

        Normalized so that
        :math:`\\int_{-\\infty}^{\\infty}\\Psi(\\Delta t)\\,d\\Delta
        t=1`.
        """
        dt = X2 - X1 - self.shift
        norm = jnp.sqrt(jnp.pi) * self.width
        return jnp.exp(-jnp.square(dt / self.width)) / norm


class ExponentialTransferFunction(TransferFunction):
    """Exponential transfer function: :math:`\\propto e^{-|\\Delta t-\\Delta t_0|/w}`.

    where :math:`\\Delta t_0=\\mathrm{shift}`.
    The unity-normalization coefficient is:
    :math:`\\frac{1}{2w}`.
    """

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """Evaluate the normalized transfer function at two points.

        Normalized so that
        :math:`\\int_{-\\infty}^{\\infty}\\Psi(\\Delta t)\\,d\\Delta
        t=1`.
        """
        dt = X2 - X1 - self.shift
        norm = 2.0 * self.width
        return jnp.exp(-jnp.abs(dt) / self.width) / norm


class CausalGaussianTransferFunction(TransferFunction):
    """Causal Gaussian: :math:`\\propto e^{-((\\Delta t-\\Delta t_0)/w)^2},\\Delta t\\ge0`.

    where :math:`\\Delta t_0=\\mathrm{shift}`.
    The unity-normalization coefficient is:
    :math:`\\left[\\frac{\\sqrt{\\pi}}{2}w\\left(1+\\mathrm{erf}(\\mathrm{shift}/w)\\right)\\right]^{-1}`.
    """  # noqa: E501

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """Evaluate the normalized transfer function at two points.

        Normalized so that
        :math:`\\int_{-\\infty}^{\\infty}\\Psi(\\Delta t)\\,d\\Delta
        t=1` for any shift.
        """
        ds = X2 - X1
        dt = ds - self.shift
        norm = (
            jnp.sqrt(jnp.pi)
            / 2
            * self.width
            * (1 + jax.scipy.special.erf(self.shift / self.width))
        )
        return jnp.where(ds >= 0, jnp.exp(-jnp.square(dt / self.width)) / norm, 0.0)


class CausalExponentialTransferFunction(TransferFunction):
    """Causal exponential: :math:`\\propto e^{-(\\Delta t-\\Delta t_0)/w},\\Delta t\\ge\\Delta t_0`.

    where :math:`\\Delta t_0=\\mathrm{shift}`.
    Defined for :math:`\\Delta t\\ge\\Delta t_0`, zero otherwise.
    The unity-normalization coefficient is:
    :math:`\\frac{1}{w}`.
    """  # noqa: E501

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """Evaluate the normalized transfer function at two points.

        Normalized so that
        :math:`\\int_{-\\infty}^{\\infty}\\Psi(\\Delta t)\\,d\\Delta
        t=1` for any shift.
        """
        dt = X2 - X1 - self.shift
        return jnp.where(dt >= 0, jnp.exp(-dt / self.width) / self.width, 0.0)


class ConvolvedKernel(tinygp.kernels.Kernel):
    """Kernel convolved with a transfer function via FFT.

    Computes the convolved kernel using the Wiener-Khinchin relation:
        :math:`S_{\\mathrm{conv}}(f)=S_{\\mathrm{base}}(f)\\,|\\hat{\\Psi}(f)|^2`
        :math:`k_{\\mathrm{conv}}(\\tau)=\\mathrm{IFFT}[S_{\\mathrm{conv}}](\\tau)`

    where :math:`\\hat{\\Psi}` is the Fourier transform of the transfer function
    and :math:`S_{\\mathrm{base}}` is the power spectral density of the base
    kernel.
    """

    # We actually need .power(), so it could be extended to "direct" kernels
    base_kernel: Quasisep
    transfer_function: TransferFunction
    n_grid: int = eqx.field(static=True)
    truncation_factor: float = eqx.field(static=True, default=6.0)

    def coord_to_sortable(self, X) -> JAXArray:
        """Extract the time-sortable component of the coordinates."""
        return X[0]

    @property
    def _half_width(self):
        """Half-width of integration grid around center."""
        scales = find_param_by_name(self.base_kernel, "scale")
        scale = sum(scales) / len(scales)
        width = self.transfer_function.width
        return (scale + width) * self.truncation_factor

    @property
    def _center(self):
        """Center of integration grid."""
        return self.transfer_function.shift

    def _compute_k_conv(self) -> tuple[JAXArray, JAXArray]:
        """Run FFT once and return the lag grid and convolved kernel values."""
        hw = self._half_width
        center = self._center
        n = self.n_grid

        # Uniform grid covering the TF support with zero-padding (2× support)
        grid_len = 4 * hw
        ds = grid_len / n
        s_grid = center - 2 * hw + jnp.arange(n) * ds

        # Evaluate Ψ on the grid
        zero = jnp.zeros(n)
        psi_vals = self.transfer_function.evaluate(zero, s_grid)

        # FFT-based computation: S_conv(f) = S_base(f) × |Ψ̂(f)|²
        psi_fft = jnp.fft.rfft(psi_vals)
        freqs = jnp.fft.rfftfreq(n, d=ds)
        psd_base = self.base_kernel.power(freqs)
        psd_conv = psd_base * jnp.abs(psi_fft) ** 2

        # IFFT → k_conv on uniform lag grid
        k_conv = ds * jnp.fft.irfft(psd_conv, n=n)

        # First half = non-negative lags
        n_half = n // 2 + 1
        tau_grid = jnp.arange(n_half) * ds
        return tau_grid, k_conv[:n_half]

    def evaluate(self, X1, X2) -> JAXArray:
        """Evaluate the transfer function at two points."""
        tau_grid, k_vals = self._compute_k_conv()
        tau = jnp.abs(X1 - X2)
        return jnp.interp(tau, tau_grid, k_vals)

    # Override __call__ so the FFT runs once per (X1, X2) call instead of once
    # per pair via vmap(evaluate).
    def __call__(self, X1: JAXArray, X2: JAXArray | None = None) -> JAXArray:
        """Evaluate the kernel matrix (or diagonal) for arrays of input coordinates.

        Overrides the parent implementation to run the FFT-based computation once
        for the entire input, rather than once per pair via ``vmap``.

        Args:
            X1: Array of input coordinates, shape ``(n1,)``.
            X2: Array of input coordinates, shape ``(n2,)``. If ``None``,
                returns the diagonal ``k(X1[i], X1[i])``, shape ``(n1,)``.

        Returns:
            Kernel matrix of shape ``(n1, n2)``, or diagonal of shape ``(n1,)``
            when ``X2`` is ``None``.
        """
        tau_grid, k_vals = self._compute_k_conv()

        if X2 is None:
            # Diagonal: stationary kernel, so k(X, X) = k(tau=0) for all X
            k = jnp.full(X1.shape[0], jnp.interp(jnp.zeros(()), tau_grid, k_vals))
            if k.ndim != 1:
                raise ValueError(
                    "Invalid kernel diagonal shape: "
                    f"expected ndim = 1, got ndim={k.ndim}"
                )
            return k

        # Full matrix: tau[i, j] = |X1[i] - X2[j]|
        tau = jnp.abs(X1[:, None] - X2[None, :])
        k = jnp.interp(tau, tau_grid, k_vals)
        if k.ndim != 2:
            raise ValueError(
                "Invalid kernel shape: " f"expected ndim = 2, got ndim={k.ndim}"
            )
        return k

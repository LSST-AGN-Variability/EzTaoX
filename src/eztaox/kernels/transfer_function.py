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
    """Base class for transfer functions :math:`\\Psi(\\Delta t)`.

    Normalized so that :math:`\\int_{-\\infty}^{\\infty}\\Psi(\\Delta t)\\,d\\Delta t=1`.
    """  # noqa: E501

    width: float
    shift: JAXArray | float = eqx.field(default_factory=lambda: jnp.zeros(()))

    @abstractmethod
    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """Evaluate the transfer function at two points."""
        del X1, X2
        raise NotImplementedError


class GaussianTransferFunction(TransferFunction):
    """Gaussian transfer function: :math:`\\Psi(s)\\propto\\exp(-((s-\\mathrm{shift})/w)^2)`."""  # noqa: E501

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """Evaluate the normalized transfer function at two points.

        Normalized so that :math:`\\int_{-\\infty}^{\\infty}\\Psi(s)\\,ds=1`.
        """
        dt = X2 - X1 - self.shift
        norm = jnp.sqrt(jnp.pi) * self.width
        return jnp.exp(-jnp.square(dt / self.width)) / norm


class ExponentialTransferFunction(TransferFunction):
    """Exponential transfer function: :math:`\\Psi(s)=(1/w)\\exp(-|s-\\mathrm{shift}|/w)`."""  # noqa: E501

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """Evaluate the normalized transfer function at two points.

        Normalized so that :math:`\\int_{-\\infty}^{\\infty}\\Psi(s)\\,ds=1`.
        """
        dt = X2 - X1 - self.shift
        norm = 2.0 * self.width
        return jnp.exp(-jnp.abs(dt) / self.width) / norm


class CausalGaussianTransferFunction(TransferFunction):
    """Causal Gaussian transfer function.

    :math:`\\Psi(s)\\propto\\exp(-((s-\\mathrm{shift})/w)^2)` for
    :math:`s\\ge 0`, zero otherwise.
    """

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """Evaluate the normalized transfer function at two points.

        Normalized so that :math:`\\int_{-\\infty}^{\\infty}\\Psi(s)\\,ds=1`
        for any shift.
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
    """Causal exponential transfer function.

    :math:`\\Psi(s)=(1/w)\\exp(-(s-\\mathrm{shift})/w)` for
    :math:`s\\ge\\mathrm{shift}`, zero otherwise.
    """

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """Evaluate the normalized transfer function at two points.

        Normalized so that :math:`\\int_{-\\infty}^{\\infty}\\Psi(s)\\,ds=1`
        for any :math:`\\mathrm{shift}\\ge 0`.
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

    def coord_to_sortable(self, X) -> JAXArray:  # noqa: D102
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

    def evaluate(self, X1, X2) -> JAXArray:  # noqa: D102
        tau = jnp.abs(X1 - X2)

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

        # Interpolate at desired lag (first half = non-negative lags)
        n_half = n // 2 + 1
        tau_grid = jnp.arange(n_half) * ds

        return jnp.interp(tau, tau_grid, k_conv[:n_half])

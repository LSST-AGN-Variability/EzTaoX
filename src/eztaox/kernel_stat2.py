"""Second-Order statistic functions for kernels in kernels.quasisep."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import tinygp
from numpy.typing import NDArray
from tinygp.helpers import JAXArray


class gpStat2(eqx.Module):  # noqa: N801
    """Base class for second-order statistics of GP kernels.

    The :meth:`acf`, :meth:`sf` and :meth:`psd` methods are compiled with
    :func:`equinox.filter_jit`, so the first call for a given input shape traces
    and compiles; later calls reuse the compiled function. They can also be
    transformed with :func:`jax.vmap`, e.g. to evaluate a statistic over a batch
    of posterior parameter draws.

    Args:
        kernel (Callable): A kernel function from kernels.quasisep.

    Attributes:
        kernel_def (Callable): A function to reconstruct kernel objects from
            its parameters
        kernel_params (JAXArray): parameters for kernel function
    """

    kernel_def: Callable
    kernel_params: JAXArray

    def __init__(self, kernel: Callable) -> None:
        self.kernel_def = jax.flatten_util.ravel_pytree(kernel)[1]
        self.kernel_params = jax.flatten_util.ravel_pytree(kernel)[0]

    def _build_kernel(
        self, params: JAXArray | NDArray
    ) -> tuple[tinygp.kernels.Kernel, JAXArray]:
        kernel = self.kernel_def(jnp.asarray(params))
        return kernel, kernel.evaluate_diag(0.0)

    @eqx.filter_jit
    def acf(
        self, ts: JAXArray | NDArray, params: JAXArray | NDArray | None = None
    ) -> JAXArray:
        """Compute the autocorrelation function (ACF) for given time lags.

        Args:
            ts (JAXArray | NDArray): Time lags for which to compute the ACF.
            params (JAXArray | NDArray | None, optional): Parameters of the GP kernel if
                different from those used to initialize the object. Defaults to None.

        Returns:
            JAXArray: The computed ACF values for the given time lags.
        """
        params = self.kernel_params if params is None else params
        kernel, amp2 = self._build_kernel(params)
        acvf = jax.vmap(kernel.evaluate, in_axes=(None, 0))(0.0, jnp.asarray(ts))
        return acvf / amp2

    @eqx.filter_jit
    def sf(
        self, ts: JAXArray | NDArray, params: JAXArray | NDArray | None = None
    ) -> JAXArray:
        """Compute the structure function (SF) for given time lags.

        Args:
            ts (JAXArray | NDArray): Time lags for which to compute the SF.
            params (JAXArray | NDArray | None, optional): Parameters of the GP kernel if
                different from those used to initialize the object. Defaults to None.

        Returns:
            JAXArray: The computed SF values for the given time lags.
        """
        params = self.kernel_params if params is None else params
        kernel, amp2 = self._build_kernel(params)
        acf = jax.vmap(kernel.evaluate, in_axes=(None, 0))(0.0, jnp.asarray(ts)) / amp2
        return jnp.sqrt(2 * amp2 * (1 - acf))

    @eqx.filter_jit
    def psd(
        self,
        fs: JAXArray | NDArray,
        params: JAXArray | NDArray | None = None,
        df: float | JAXArray | None = 0.01,
    ) -> JAXArray:
        """Compute the power spectral density (PSD) for given frequencies.

        Args:
            fs (JAXArray | NDArray): Frequencies for which to compute the PSD.
            params (JAXArray | NDArray | None, optional): Parameters of the GP kernel if
                different from those used to initialize the object. Defaults to None.
            df (float | JAXArray | None, optional): Frequency width for create convolved
                PSDs (not in use). Defaults to 0.01.

        Returns:
            JAXArray: The computed PSD values for the given frequencies.
        """
        params = self.kernel_params if params is None else params
        kernel, _ = self._build_kernel(params)
        return jnp.stack(jax.vmap(kernel.power, in_axes=(0, None))(jnp.asarray(fs), df))

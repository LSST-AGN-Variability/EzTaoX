"""Utilities for parameterized kernel specifications."""

from collections.abc import Callable

import jax.flatten_util
import jax.numpy as jnp
import tinygp.kernels as tk
from tinygp.helpers import JAXArray

KernelSpec = tk.Kernel | type[tk.Kernel] | Callable[..., tk.Kernel]


def _call_with_kernel_params(
    kernel_factory: Callable[..., tk.Kernel], params: JAXArray
) -> tk.Kernel:
    return kernel_factory(*jnp.atleast_1d(params))


def build_kernel_def(
    kernel: KernelSpec,
) -> tuple[Callable[[JAXArray], tk.Kernel], bool]:
    """Return a kernel builder and whether the spec is known to be quasiseparable.

    Initialized kernel instances are interpreted as pytree definitions, preserving the
    original API. Kernel classes and callable factories are called with the
    exponentiated entries of ``params["log_kernel_param"]`` as positional arguments.
    """
    if isinstance(kernel, tk.Kernel):
        return jax.flatten_util.ravel_pytree(kernel)[1], isinstance(
            kernel, tk.quasisep.Quasisep
        )

    if isinstance(kernel, type):
        if not issubclass(kernel, tk.Kernel):
            msg = (
                "Expected a tinygp kernel instance, tinygp kernel class, or callable "
                f"kernel factory; got {kernel!r}."
            )
            raise TypeError(msg)
        return (
            lambda params: _call_with_kernel_params(kernel, params),
            issubclass(kernel, tk.quasisep.Quasisep),
        )

    if callable(kernel):
        return lambda params: _call_with_kernel_params(kernel, params), False

    msg = (
        "Expected a tinygp kernel instance, tinygp kernel class, or callable "
        f"kernel factory; got {type(kernel).__name__}."
    )
    raise TypeError(msg)

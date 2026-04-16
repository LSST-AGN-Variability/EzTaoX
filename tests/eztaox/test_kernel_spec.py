"""Tests for kernel specification helpers."""

import jax.numpy as jnp
import pytest
import tinygp.kernels as tk
import tinygp.kernels.quasisep as tkq

from eztaox.kernels._spec import build_kernel_def


def test_build_kernel_def_accepts_kernel_instance() -> None:
    kernel_def, is_quasisep = build_kernel_def(tk.Exp(scale=1.0))

    kernel = kernel_def(jnp.array([2.0]))

    assert isinstance(kernel, tk.Kernel)
    assert is_quasisep is False


def test_build_kernel_def_accepts_quasisep_kernel_class() -> None:
    kernel_def, is_quasisep = build_kernel_def(tkq.Exp)

    kernel = kernel_def(jnp.array([2.0, 0.5]))

    assert isinstance(kernel, tkq.Quasisep)
    assert is_quasisep is True


def test_build_kernel_def_accepts_kernel_factory() -> None:
    def kernel_factory(scale, amplitude):
        return amplitude * tk.Exp(scale=scale)

    kernel_def, is_quasisep = build_kernel_def(kernel_factory)

    kernel = kernel_def(jnp.array([2.0, 0.5]))

    assert isinstance(kernel, tk.Kernel)
    assert is_quasisep is False


def test_build_kernel_def_rejects_non_kernel_class() -> None:
    with pytest.raises(TypeError, match="kernel instance"):
        build_kernel_def(float)


def test_build_kernel_def_rejects_invalid_object() -> None:
    with pytest.raises(TypeError, match="got int"):
        build_kernel_def(1)

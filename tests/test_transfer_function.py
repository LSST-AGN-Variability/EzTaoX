import jax.numpy as jnp
import pytest
from tinygp.test_utils import assert_allclose

from eztaox.kernels import quasisep
from eztaox.kernels.transfer_function import (
    CausalExponentialTransferFunction,
    CausalGaussianTransferFunction,
    ConvolvedKernel,
    ExponentialTransferFunction,
    GaussianTransferFunction,
    TransferFunction,
)


def _analytic_convolved_exp(dt, tau, w):
    """Analytic result for Exp kernel convolved with exponential transfer function.

    K_conv(Δt) = A exp(-|Δt|/τ) - B exp(-|Δt|/w) - C |Δt| exp(-|Δt|/w)

    where D = τ² - w², A = τ⁴/D², B = wτ(3τ²-w²)/2D², C = τ/2D.

    The base kernel is Exp(scale=τ): K(Δt) = exp(-|Δt|/τ).
    The transfer function is Ψ(Δt) = (1/w) exp(-|Δt|/w).
    Requires τ ≠ w.
    """
    abs_dt = jnp.abs(dt)
    D = tau**2 - w**2
    A = tau**4 / D**2
    B = 0.5 * w * tau * (3 * tau**2 - w**2) / D**2
    C = 0.5 * tau / D
    return (
        A * jnp.exp(-abs_dt / tau)
        - B * jnp.exp(-abs_dt / w)
        - C * abs_dt * jnp.exp(-abs_dt / w)
    )


def test_convolved_kernel_analytic():
    """Test ConvolvedKernel against analytic result for Exp + exponential TF."""
    lags = jnp.array([0.0, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0])

    for tau, w in [(400.0, 50.0), (50.0, 10.0)]:
        tf = ExponentialTransferFunction(width=w)
        base = quasisep.Exp(scale=tau)
        convolved_kernel = ConvolvedKernel(base, tf, n_grid=2000, truncation_factor=6.0)

        for lag in lags:
            expected = _analytic_convolved_exp(lag, tau, w)
            actual = convolved_kernel.evaluate(lag, jnp.array(0.0))
            assert_allclose(
                actual, expected, atol=1e-2, err_msg=f"Failed for tau={tau} and w={w}"
            )


@pytest.mark.parametrize("shift", [0.0, 100.0, 500.0])
def test_convolved_kernel_shift_invariance(shift):
    """Check that ConvolvedKernel gives the same result regardless of shift."""
    lags = jnp.array([0.0, 10.0, 50.0, 200.0])
    tau, w = 50.0, 10.0
    base = quasisep.Exp(scale=tau)

    tf_ref = ExponentialTransferFunction(width=w)
    ck_ref = ConvolvedKernel(base, tf_ref, n_grid=80, truncation_factor=6.0)

    tf_shifted = ExponentialTransferFunction(width=w, shift=shift)
    ck_shifted = ConvolvedKernel(base, tf_shifted, n_grid=80, truncation_factor=6.0)

    for lag in lags:
        expected = ck_ref.evaluate(lag, jnp.array(0.0))
        actual = ck_shifted.evaluate(lag, jnp.array(0.0))
        assert_allclose(actual, expected, atol=1e-4)


@pytest.mark.parametrize(
    "transfer_function",
    [
        GaussianTransferFunction(width=10.0),
        GaussianTransferFunction(width=50.0, shift=20.0),
        ExponentialTransferFunction(width=10.0, shift=5.0),
        ExponentialTransferFunction(width=50.0),
        CausalGaussianTransferFunction(width=10.0),
        CausalGaussianTransferFunction(width=50.0, shift=30.0),
        CausalExponentialTransferFunction(width=10.0, shift=10.0),
        CausalExponentialTransferFunction(width=50.0),
    ],
    ids=lambda tf: f"{type(tf).__name__}(w={tf.width},s={tf.shift})",
)
def test_transfer_function_normalization(transfer_function: TransferFunction):
    """Check that ∫₋∞^∞ Ψ(s) ds = 1 for all transfer functions."""
    center = float(transfer_function.shift)
    hw = 10.0 * transfer_function.width
    s = jnp.linspace(center - hw, center + hw, 10_000)
    psi = transfer_function.evaluate(jnp.zeros_like(s), s)
    result = jnp.trapezoid(psi, s)
    assert_allclose(result, jnp.array(1.0), atol=1e-4)

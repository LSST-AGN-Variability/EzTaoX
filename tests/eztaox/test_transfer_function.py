import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose
from tinygp.kernels import Exp as ExpDirect

from eztaox.kernels import quasisep
from eztaox.kernels.transfer_function import (
    CausalExponentialTransferFunction,
    CausalGaussianTransferFunction,
    ConvolvedKernel,
    ExponentialTransferFunction,
    GaussianTransferFunction,
    TransferFunction,
)
from eztaox.simulator import UniVarSim


def simulate_exp(tau, sigma, transfer_function, rng):
    n = 1000
    t = jnp.linspace(0.0, 1000.0, n)
    dt = t[1] - t[0]

    sim = UniVarSim(
        ExpDirect(scale=tau),
        1.0,
        float(t[-1]),
        init_params={"log_kernel_param": jnp.array([jnp.log(tau)])},
        zero_mean=True,
    )

    _, latent = sim.fixed_input_fast(
        t,
        jax.random.PRNGKey(rng),
    )

    # symmetric kernel centered at t = 0
    t_kernel = jnp.r_[-t[::-1][1:], t]  # length 2n-1, zero at center
    kernel = jax.vmap(transfer_function.evaluate)(jnp.zeros_like(t_kernel), t_kernel)

    # zero-phase, index-aligned convolution
    result = jax.scipy.signal.fftconvolve(latent, kernel, mode="same") * dt

    return t, latent, result


def test_simulate_exp():
    tau = 100.0
    sigma = 1.0
    psi_width = 20.0
    rng = 1

    transfer_function = GaussianTransferFunction(width=psi_width)
    base_kernel = quasisep.Exp(scale=tau, sigma=sigma)
    convolved_kernel = ConvolvedKernel(base_kernel, transfer_function, n_grid=2000)

    t, lc_latent, lc_expected = simulate_exp(tau, sigma, transfer_function, rng)

    sim = UniVarSim(
        convolved_kernel,
        1.0,
        float(t[-1]),
        init_params={
            "log_kernel_param": jnp.array(
                [jnp.log(tau), jnp.log(sigma), jnp.log(psi_width)]
            )
        },
        zero_mean=True,
    )

    _, lc_actual = sim.fixed_input_fast(t, jax.random.PRNGKey(rng))

    def empirical_sf(lc, ks):
        return jnp.array([jnp.sqrt(jnp.mean((lc[k:] - lc[:-k]) ** 2)) for k in ks])

    dt_val = float(t[1] - t[0])
    min_lag_idx = max(1, round(0.1 * psi_width / dt_val))
    max_lag_idx = round(2.0 * tau / dt_val)
    lag_indices = jnp.unique(
        jnp.geomspace(min_lag_idx, max_lag_idx, 100).round().astype(int)
    )

    sf_expected = empirical_sf(lc_expected, lag_indices)
    sf_actual = empirical_sf(lc_actual, lag_indices)

    assert_allclose(sf_expected, sf_actual, rtol=0.1, atol=0)


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


def test_convolved_kernel_call_matches_evaluate():
    """ConvolvedKernel.__call__ agrees with evaluate for both matrix and diagonal."""
    tau, w = 100.0, 20.0
    base = quasisep.Exp(scale=tau)
    tf = ExponentialTransferFunction(width=w)
    ck = ConvolvedKernel(base, tf, n_grid=500)

    t1 = jnp.array([0.0, 10.0, 50.0, 100.0])
    t2 = jnp.array([5.0, 25.0, 75.0])

    K_call = ck(t1, t2)
    K_eval = jax.vmap(jax.vmap(ck.evaluate, (None, 0)), (0, None))(t1, t2)
    assert_allclose(K_call, K_eval)

    diag_call = ck(t1, None)
    diag_eval = jax.vmap(ck.evaluate)(t1, t1)
    assert_allclose(diag_call, diag_eval)


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

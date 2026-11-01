"""Tests for eztaox.kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tinygp
from celerite import GP
from eztao.carma import CARMA_term
from numpy import random as np_random
from numpy.random import Generator
from packaging.version import Version
from tinygp import GaussianProcess
from tinygp.helpers import JAXArray
from tinygp.test_utils import assert_allclose

from eztaox.kernels import quasisep


@pytest.fixture
def random() -> Generator:
    return np_random.default_rng(84930)


@pytest.fixture
def data(random) -> tuple[JAXArray, JAXArray, JAXArray]:
    x = jnp.sort(random.uniform(-3, 3, 50))
    y = jnp.sin(x)
    t = jnp.sort(random.uniform(-3, 3, 12))
    return x, y, t


def test_carma(data) -> None:
    x, y, _ = data
    # CARMA kernels implemented using JAX
    jax_carma_kernels = [
        quasisep.CARMA(alpha=jnp.array([0.01]), beta=jnp.array([0.1])),
        quasisep.CARMA(alpha=jnp.array([1.0, 1.2]), beta=jnp.array([1.0, 3.0])),
        quasisep.CARMA(alpha=jnp.array([0.1, 1.1]), beta=jnp.array([1.0, 3.0])),
        quasisep.CARMA(alpha=jnp.array([3.0, 3.2, 1.2]), beta=jnp.array([1.0])),
        quasisep.CARMA(alpha=jnp.array([1.2, 3.2, 3.0]), beta=jnp.array([1.0])),
    ]

    # CARMA kernels implemented by EzTao
    eztao_carma_kernels = [
        CARMA_term(np.log(np.array([0.01])), np.log(np.array([0.1]))),
        CARMA_term(np.log(np.array([1.2, 1.0])), np.log(np.array([1.0, 3.0]))),
        CARMA_term(np.log(np.array([1.1, 0.1])), np.log(np.array([1.0, 3.0]))),
        CARMA_term(np.log(np.array([1.2, 3.2, 3.0])), np.log(np.array([1.0]))),
        CARMA_term(np.log(np.array([3.0, 3.2, 1.2])), np.log(np.array([1.0]))),
    ]
    # Equivalent Celerite+Exp kernels for validation
    jax_validate_kernels = [
        quasisep.Exp(scale=100.0, sigma=jnp.sqrt(0.5)),
        quasisep.Celerite(25.0 / 6, 2.5, 0.6, -0.8),
        quasisep.Exp(1.0, jnp.sqrt(4.04040404))
        + quasisep.Exp(10.0, jnp.sqrt(4.5959596)),
    ]

    # Compare log_probability & normalization under tinygp implementation
    for i in range(len(jax_validate_kernels)):
        gp1 = GaussianProcess(jax_carma_kernels[i], x, diag=0.1)
        gp2 = GaussianProcess(jax_validate_kernels[i], x, diag=0.1)

        assert_allclose(gp1.log_probability(y), gp2.log_probability(y))
        assert_allclose(
            jax.grad(gp1.log_probability)(y), jax.grad(gp2.log_probability)(y)
        )
        assert_allclose(gp1.solver.normalization(), gp2.solver.normalization())

    # Compare log_probability between tinygp and eztao implementation
    for i in range(len(jax_carma_kernels)):
        gp1 = GaussianProcess(jax_carma_kernels[i], x, diag=0.1)
        gp2 = GP(eztao_carma_kernels[i], mean=0.0)
        gp2.compute(x, yerr=np.sqrt(0.1) * np.ones_like(x))

        assert_allclose(gp1.log_probability(y), gp2.log_likelihood(y))


def test_carma_quads():
    alpha = jnp.array([1.5, 2.3, 1.4])
    beta = jnp.array([0.1, 0.5])
    alpha_quads = quasisep.carma_poly2quads(jnp.append(alpha, 1.0))
    beta_quads = quasisep.carma_poly2quads(beta)

    # seperate quad coeffs from mult_f
    alpha_quads = alpha_quads[:-1]
    beta_mult = beta_quads[-1]
    beta_quads = beta_quads[:-1]

    carma31 = quasisep.CARMA(alpha=alpha, beta=beta)
    carma31_quads = quasisep.CARMA.from_quads(
        alpha_quads=alpha_quads, beta_quads=beta_quads, beta_mult=beta_mult
    )

    # if two constructor give the same model
    assert_allclose(carma31.arroots, carma31_quads.arroots)


def test_carma_jit_grad(data) -> None:
    x, y, _ = data

    def build_gp(params) -> GaussianProcess:
        carma_kernel = quasisep.CARMA(alpha=params["alpha"], beta=params["beta"])
        return GaussianProcess(carma_kernel, x, diag=0.01, mean=0.0)

    @jax.jit
    def loss(params) -> JAXArray:
        gp = build_gp(params)
        return -gp.log_probability(y)

    params = {"alpha": jnp.array([1.0, 1.2]), "beta": jnp.array([1.0, 3.0])}
    loss(params)
    jax.grad(loss)(params)


@pytest.mark.parametrize(
    ("alpha", "sigma_w"),
    [
        (jnp.array([0.01]), 1.0),
        (jnp.array([1.0, 1.2]), 1.0),
        (jnp.array([0.5, 1.5, 1.0]), 0.7),
    ],
)
def test_carma_root_stationary_covariance(alpha, sigma_w) -> None:
    arroots = quasisep.carma_roots(jnp.append(alpha, 1.0))
    p = arroots.shape[0]

    expected = np.zeros((p, p), dtype=np.complex128)
    for i in range(p):
        for j in range(p):
            value = 0.0j
            for k in range(p):
                denom = 2.0 * np.real(arroots[k])
                for l in range(p):
                    if l == k:
                        continue
                    denom *= (arroots[l] - arroots[k]) * (
                        np.conj(arroots[l]) + arroots[k]
                    )
                value += (arroots[k] ** i) * ((-arroots[k]) ** j) / denom
            expected[i, j] = -(sigma_w**2) * value

    expected = np.real(0.5 * (expected + expected.T.conj()))
    actual = quasisep.carma_root_stationary_covariance(arroots, sigma_w)

    assert_allclose(actual, expected)


_tinygp_nonreversible = pytest.mark.skipif(
    Version(tinygp.__version__) < Version("0.3.2.dev0"),
    reason="MixedBLRKernel needs tinygp's non-reversible quasisep convention",
)


@_tinygp_nonreversible
@pytest.mark.parametrize(
    "base",
    [
        quasisep.Exp(scale=150.0, sigma=0.3),
        quasisep.SHO(omega=2 * np.pi / 300, quality=0.4, sigma=0.3),
    ],
)
def test_mixed_blr_kernel(base) -> None:
    from scipy.stats import gamma

    lag, order = 60.0, 3
    params = {
        "lag": jnp.asarray(lag),
        "amp_cont": jnp.array([1.0, 0.8]),
        "amp_blr": jnp.array([0.0, 0.5]),
    }
    kernel = quasisep.MixedBLRKernel(params=params, kernel=base, order=order)

    def k_x(d):
        d = jnp.atleast_1d(jnp.asarray(d, dtype=float))
        return np.asarray(jax.vmap(lambda x: base.evaluate(jnp.zeros(()), x))(d))

    # numerical convolution with the Erlang transfer function
    u = np.linspace(0.0, 1500.0, 6001)
    du = u[1] - u[0]
    psi = gamma.pdf(u, order, scale=lag / order)
    uu = (u[:, None] - u[None, :])[::5, ::5]
    var_l = (k_x(uu.ravel()).reshape(uu.shape) * np.outer(psi[::5], psi[::5])).sum()
    var_l *= (5 * du) ** 2
    r_tilde = 0.5 * np.sqrt(k_x(0.0)[0] / var_l)
    for d in [0.0, 30.0, 60.0, 200.0]:
        expected = 0.8 * k_x(d)[0] + r_tilde * (k_x(d - u) * psi).sum() * du
        forward = kernel.evaluate((jnp.asarray(0.0), 0), (jnp.asarray(d), 1))
        assert_allclose(forward, expected, atol=1e-4 * k_x(0.0)[0])
        # the BLR term only correlates with the continuum's past
        backward = kernel.evaluate((jnp.asarray(d), 0), (jnp.asarray(0.0), 1))
        expected_back = 0.8 * k_x(d)[0] + r_tilde * (k_x(d + u) * psi).sum() * du
        assert_allclose(backward, expected_back, atol=1e-4 * k_x(0.0)[0])

    # the continuum band reproduces the driver kernel
    assert_allclose(
        kernel.evaluate((jnp.asarray(0.0), 0), (jnp.asarray(40.0), 0)), k_x(40.0)[0]
    )

    # the closed-form transition matches the matrix exponential
    A = kernel.design_matrix()
    for d in [1.0, 60.0, 600.0]:
        phi = kernel.transition_matrix((jnp.asarray(0.0), 0), (jnp.asarray(d), 0)).T
        assert_allclose(phi, jax.scipy.linalg.expm(A * d))

"""Tests for eztaox.kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from celerite import GP
from eztao.carma import CARMA_term
from matplotlib.pylab import Generator
from numpy import random as np_random
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
    ]

    # CARMA kernels implemented by EzTao
    eztao_carma_kernels = [
        CARMA_term(np.log(np.array([0.01])), np.log(np.array([0.1]))),
        CARMA_term(np.log(np.array([1.2, 1.0])), np.log(np.array([1.0, 3.0]))),
        CARMA_term(np.log(np.array([1.1, 0.1])), np.log(np.array([1.0, 3.0]))),
    ]
    # Equivalent Celerite+Exp kernels for validation
    jax_validate_kernels = [
        quasisep.Exp(scale=100.0, sigma=jnp.sqrt(0.5)),
        quasisep.Celerite(25.0 / 6, 2.5, 0.6, -0.8),
        quasisep.Exp(1.0, jnp.sqrt(4.04040404))
        + quasisep.Exp(10.0, jnp.sqrt(4.5959596)),
    ]

    # Compare log_probability & normalization under tinygp implementation
    for i in range(len(jax_carma_kernels)):
        gp1 = GaussianProcess(jax_carma_kernels[i], x, diag=0.1)
        gp2 = GaussianProcess(jax_validate_kernels[i], x, diag=0.1)

        assert_allclose(gp1.log_probability(y), gp2.log_probability(y))
        assert_allclose(gp1.solver.normalization(), gp2.solver.normalization())

    # Compare log_probability between tinygp and extao implementation
    for i in range(len(jax_carma_kernels))[:2]:
        gp1 = GaussianProcess(jax_carma_kernels[i], x, diag=0.1)
        gp2 = GP(eztao_carma_kernels[i], mean=0.0)
        gp2.compute(x, yerr=np.sqrt(0.1) * np.ones_like(x))

        assert_allclose(gp1.log_probability(y), gp2.log_likelihood(y))


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

"Kernels module"
from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import tinygp
from numpy.typing import NDArray
from tinygp.helpers import JAXArray
from tinygp.kernels.quasisep import Quasisep


class mb_kernel(tinygp.kernels.quasisep.Wrapper):
    amplitudes: jnp.ndarray

    def coord_to_sortable(self, X) -> JAXArray:
        return X[0]

    def observation_model(self, X) -> JAXArray:
        return self.amplitudes[X[1]] * self.kernel.observation_model(X[0])


class CARMA(Quasisep):
    r"""A continuous-time autoregressive moving average (CARMA) process kernel

    This process has the power spectrum density (PSD)

    .. math::

        P(\omega) = \sigma^2\,\frac{|\sum_{q} \beta_q\,(i\,\omega)^q|^2}{|\sum_{p}
            \alpha_p\,(i\,\omega)^p|^2}

    defined following Equation 1 in `Kelly et al. (2014)
    <https://arxiv.org/abs/1402.5978>`_, where :math:`\alpha_p` and :math:`\beta_0`
    are set to 1. In this implementation, we absorb :math:`\sigma` into the
    definition of :math:`\beta` parameters. That is :math:`\beta_{new}` =
    :math:`\beta * \sigma`.

    .. note::
        To construct a stationary CARMA kernel/process, the roots of the
        characteristic polynomials for Equation 1 in `Kelly et al. (2014)` must
        have negative real parts. This condition can be met automatically by
        requiring positive input parameters when instantiating the kernel using
        the :func:`init` method for CARMA(1,0), CARMA(2,0), and CARMA(2,1)
        models or by requiring positive input parameters when instantiating the
        kernel using the :func:`from_quads` method.

    .. note:: Implementation details

        The logic behind this implementation is simple---finding the correct
        combination of real/complex exponential kernels that resembles the
        autocovariance function of the CARMA model. Note that the order also
        matters. This task is achieved using the `acvf` method. Then the rest
        is copied from the `Exp` and `Celerite` kernel.

        Given the requirement of negative roots for stationarity, the
        `from_quads` method is implemented to facilitate consturcting
        stationary higher-order CARMA models beyond CARMA(2,1). The inputs for
        `from_quads` are the coefficients of the quadratic equations factorized
        out of the full characteristic polynomial. `poly2quads` is used to
        factorize a polynomial into a product of said quadractic equations, and
        `quads2poly` is used for the reverse process.

        One last trick is the use of `_real_mask`, `_complex_mask`, and
        `complex_select`, which are arrays of 0s and 1s. They are implemented
        to avoid control flows. More specifically, some intermediate quantities
        are computed regardless, but are only used if there is a matching real
        or complex exponential kernel for the specific CARMA kernel.

    Args:
        alpha: The parameter :math:`\alpha` in the definition above, exlcuding
            :math:`\alpha_p`. This should be an array of length `p`.
        beta: The product of parameters :math:`\beta` and parameter :math:`\sigma`
            in the definition above. This should be an array of length `q+1`,
            where `q+1 <= p`.
    """

    alpha: JAXArray = eqx.field(converter=jnp.asarray)
    beta: JAXArray = eqx.field(converter=jnp.asarray)
    sigma: float = 1.0

    def __init__(self, alpha: JAXArray | NDArray, beta: JAXArray | NDArray) -> None:
        alpha = jnp.atleast_1d(jnp.asarray(alpha))
        beta = jnp.atleast_1d(jnp.asarray(beta))
        assert alpha.ndim == 1
        assert beta.ndim == 1
        p = alpha.shape[0]
        assert beta.shape[0] <= p

        self.alpha = alpha
        self.beta = beta
        # self.sigma = jnp.ones(())

    @classmethod
    def init(cls, alpha: JAXArray, beta: JAXArray) -> CARMA:
        return cls(alpha, beta)

    @classmethod
    def from_quads(
        cls,
        alpha_quads: JAXArray | NDArray,
        beta_quads: JAXArray | NDArray,
        beta_mult: JAXArray | NDArray,
    ) -> CARMA:
        r"""Construct a CARMA kernel using the roots of its characteristic polynomials

        The roots can be parameterized as the 0th and 1st order coefficients of a set
        of quadratic equations (2nd order coefficient equals 1). The product of
        those quadratic equations gives the characteristic polynomials of CARMA.
        The input of this method are said coefficients of the quadratic equations.
        See Equation 30 in `Kelly et al. (2014) <https://arxiv.org/abs/1402.5978>`_.
        for more detail.

        Args:
            alpha_quads: Coefficients of the auto-regressive (AR) quadratic
                equations corresponding to the :math:`\alpha` parameters. This should
                be an array of length `p`.
            beta_quads: Coefficients of the moving-average (MA) quadratic
                equations corresponding to the :math:`\beta` parameters. This should
                be an array of length `q`.
            beta_mult: A multiplier of the MA coefficients, equivalent to
                :math:`\beta_q`---the last entry of the :math:`\beta` parameters input
                to the :func:`init` method.
        """

        alpha_quads = jnp.atleast_1d(alpha_quads)
        beta_quads = jnp.atleast_1d(beta_quads)
        beta_mult = jnp.atleast_1d(beta_mult)

        alpha = carma_quads2poly(jnp.append(alpha_quads, jnp.array([1.0])))[:-1]
        beta = carma_quads2poly(jnp.append(beta_quads, beta_mult))

        return cls(alpha, beta)

    def design_matrix(self) -> JAXArray:
        (
            arroots,
            acf,
            _real_mask,
            _complex_mask,
            _complex_select,
            om_real,
            om_complex,
        ) = _compute(self.alpha, self.beta, self.sigma)

        # for real exponential components
        dm_real = jnp.diag(arroots.real * _real_mask)

        # for complex exponential components
        dm_complex_diag = jnp.diag(arroots.real * _complex_mask)

        # upper triangle entries
        dm_complex_u = jnp.diag((arroots.imag * _complex_select)[:-1], k=1)

        return dm_real + dm_complex_diag + -dm_complex_u.T + dm_complex_u

    def stationary_covariance(self) -> JAXArray:
        (
            arroots,
            acf,
            _real_mask,
            _complex_mask,
            _complex_select,
            om_real,
            om_complex,
        ) = _compute(self.alpha, self.beta, self.sigma)
        p = acf.shape[0]

        # for real exponential components
        diag = jnp.diag(jnp.where(acf.real > 0, jnp.ones(p), -jnp.ones(p)))

        # for complex exponential components
        denom = jnp.where(_real_mask, 1.0, arroots.imag)
        diag_complex = jnp.diag(
            2
            * jnp.square(
                arroots.real / denom * jnp.roll(_complex_select, 1) * _complex_mask
            )
        )
        c_over_d = arroots.real / denom

        # upper triangular entries
        sc_complex_u = jnp.diag((-c_over_d * _complex_select)[:-1], k=1)

        return diag + diag_complex + sc_complex_u + sc_complex_u.T

    def observation_model(self, X: JAXArray) -> JAXArray:
        del X
        (
            arroots,
            acf,
            _real_mask,
            _complex_mask,
            _complex_select,
            om_real,
            om_complex,
        ) = _compute(self.alpha, self.beta, self.sigma)

        # return self.obsmodel
        return jnp.where(
            _real_mask,
            om_real,
            jnp.ravel(om_complex)[::2],
        )

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        (
            arroots,
            acf,
            _real_mask,
            _complex_mask,
            _complex_select,
            om_real,
            om_complex,
        ) = _compute(self.alpha, self.beta, self.sigma)

        dt = X2 - X1
        c = -arroots.real
        d = -arroots.imag
        decay = jnp.exp(-c * dt)
        sin = jnp.sin(d * dt)

        tm_real = jnp.diag(decay * _real_mask)
        tm_complex_diag = jnp.diag(decay * jnp.cos(d * dt) * _complex_mask)
        tm_complex_u = jnp.diag(
            (decay * sin * _complex_select)[:-1],
            k=1,
        )

        return tm_real + tm_complex_diag + -tm_complex_u.T + tm_complex_u


@jax.jit
def carma_roots(poly_coeffs: JAXArray) -> JAXArray:
    roots = jnp.roots(poly_coeffs[::-1], strip_zeros=False)
    return roots[jnp.argsort(roots.real)]


@jax.jit
def carma_quads2poly(quads_coeffs: JAXArray) -> JAXArray:
    """Expand a product of quadractic equations into a polynomial

    Args:
        quads_coeffs: The 0th and 1st order coefficients of the quadractic
            equations. The last entry is a multiplier, which corresponds
            to the coefficient of the highest order term in the output full
            polynomial.

    Returns:
        Coefficients of the full polynomial. The first entry corresponds to
        the lowest order term.
    """

    size = quads_coeffs.shape[0] - 1
    remain = size % 2
    nPair = size // 2
    mult_f = quads_coeffs[-1:]  # The coeff of highest order term in the output

    poly = jax.lax.cond(
        remain == 1,
        lambda x: jnp.array([1.0, x]),
        lambda _: jnp.array([0.0, 1.0]),
        quads_coeffs[-2],
    )
    poly = poly[-remain + 1 :]

    for p in jnp.arange(nPair):
        poly = jnp.convolve(
            poly,
            jnp.append(
                jnp.array([quads_coeffs[p * 2], quads_coeffs[p * 2 + 1]]),
                jnp.ones((1,)),
            )[::-1],
        )

    # the returned is low->high following Kelly+14
    return poly[::-1] * mult_f


def carma_poly2quads(poly_coeffs: JAXArray) -> JAXArray:
    """Factorize a polynomial into a product of quadratic equations

    Args:
        poly_coeffs: Coefficients of the input characteristic polynomial. The
            first entry corresponds to the lowest order term.

    Returns:
        The 0th and 1st order coefficients of the quadractic equations. The last
        entry is a multiplier, which corresponds to the coefficient of the highest
        order term in the full polynomial.
    """

    quads = jnp.empty(0)
    mult_f = poly_coeffs[-1]
    roots = carma_roots(poly_coeffs / mult_f)
    odd = bool(len(roots) & 0x1)

    rootsComp = roots[roots.imag != 0]
    rootsReal = roots[roots.imag == 0]
    nCompPair = len(rootsComp) // 2
    nRealPair = len(rootsReal) // 2

    for i in range(nCompPair):
        root1 = rootsComp[i]
        root2 = rootsComp[i + 1]
        quads = jnp.append(quads, (root1 * root2).real)
        quads = jnp.append(quads, -(root1.real + root2.real))

    for i in range(nRealPair):
        root1 = rootsReal[i]
        root2 = rootsReal[i + 1]
        quads = jnp.append(quads, (root1 * root2).real)
        quads = jnp.append(quads, -(root1.real + root2.real))

    if odd:
        quads = jnp.append(quads, -rootsReal[-1].real)

    return jnp.append(quads, jnp.array(mult_f))


def carma_acvf(arroots: JAXArray, arparam: JAXArray, maparam: JAXArray) -> JAXArray:
    r"""Compute the coefficients of the autocovariance function (ACVF)

    Args:
        arroots: The roots of the autoregressive characteristic polynomial.
        arparam: :math:`\alpha` parameters
        maparam: :math:`\beta` parameters

    Returns:
        ACVF coefficients, each entry corresponds to one root.
    """
    from jax._src import dtypes  # type: ignore

    arparam = jnp.atleast_1d(arparam)
    maparam = jnp.atleast_1d(maparam)

    complex_dtype = dtypes.to_complex_dtype(arparam.dtype)

    p = arparam.shape[0]
    q = maparam.shape[0] - 1
    sigma = maparam[0]

    # normalize beta_0 to 1
    maparam = maparam / sigma

    # init acf product terms
    num_left = jnp.zeros(p, dtype=complex_dtype)
    num_right = jnp.zeros(p, dtype=complex_dtype)
    denom = -2 * arroots.real + jnp.zeros_like(arroots) * 1j

    for k in range(q + 1):
        num_left += maparam[k] * jnp.power(arroots, k)
        num_right += maparam[k] * jnp.power(jnp.negative(arroots), k)

    root_idx = jnp.arange(p)
    for j in range(1, p):
        root_k = arroots[jnp.roll(root_idx, j)]
        denom *= (root_k - arroots) * (jnp.conj(root_k) + arroots)

    return sigma**2 * num_left * num_right / denom


@jax.jit
def _compute(alpha: JAXArray, beta: JAXArray, sigma: JAXArray) -> tuple[JAXArray, ...]:
    # Find acvf using Eqn. 4 in Kelly+14, giving the correct combination of
    # real/complex exponential kernels
    arroots = carma_roots(jnp.append(alpha, 1.0))
    acf = carma_acvf(arroots, alpha, beta * sigma)

    # Mask for real/complex exponential kernels
    _real_mask = jnp.abs(arroots.imag) < 10 * jnp.finfo(arroots.imag.dtype).eps
    _complex_mask = ~_real_mask
    complex_idx = jnp.cumsum(_complex_mask) * _complex_mask
    _complex_select = _complex_mask * complex_idx % 2

    # Construct the obsservation model => real + complex
    om_real = jnp.sqrt(jnp.abs(acf.real))

    a, b, c, d = (
        2 * acf.real,
        2 * acf.imag,
        -arroots.real,
        -arroots.imag,
    )
    max_d = jnp.finfo(a.dtype).max / 10
    c2 = jnp.square(c)
    d2 = jnp.square(d)
    s2 = c2 + d2
    denom = jnp.where(_real_mask, 1.0, 2 * c * s2)

    h2_2 = jnp.where(_real_mask, max_d, d2 * (a * c - b * d) / denom)
    h2 = jnp.sqrt(h2_2)

    denom = jnp.where(_real_mask, 1.0, d)
    a_d2_s2_h22 = jnp.where(_real_mask, max_d, a * d2 - s2 * h2_2)
    h1 = (c * h2 - jnp.sqrt(a_d2_s2_h22)) / denom

    # update h1, h2 => assign zero to real terms
    h1_final = jnp.where(_real_mask, 0.0, h1)
    h2_final = jnp.where(_real_mask, 0.0, h2)
    om_complex = jnp.array([h1_final, h2_final])

    return (
        arroots,
        acf,
        _real_mask,
        _complex_mask,
        _complex_select,
        om_real,
        om_complex,
    )

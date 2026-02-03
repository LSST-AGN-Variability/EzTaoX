"""
Scalable kernels exploiting the quasiseparable structure in the relevant matrices to
achieve a O(N) scaling. This module extends the `tinygp.kernels.quasisep` module.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import tinygp.kernels.quasisep as tkq
from jax._src import dtypes
from numpy.typing import NDArray
from tinygp.helpers import JAXArray
from tinygp.kernels import Kernel
from tinygp.kernels.quasisep import _prod_helper


class Quasisep(tkq.Quasisep):
    """
    An extension of the `tinygp.kernels.quasisep.Quasisep` kernel.

    `tinygp.kernels.quasisep.Quasisep` is the base class for all kernels that can be
    evaluated following an O(N) scaling. This extension adds a `power` method to return
    the power spectral density (PSD) of a quasiseparable kernel at an input frequency.
    """

    def __add__(self, other: Kernel | JAXArray) -> Kernel:
        if not isinstance(other, tkq.Quasisep):
            raise ValueError(
                "Quasisep kernels can only be added to other Quasisep kernels"
            )
        return Sum(self, other)

    def __radd__(self, other: Any) -> Kernel:
        # We'll hit this first branch when using the `sum` function
        if other == 0:
            return self
        if not isinstance(other, tkq.Quasisep):
            raise ValueError(
                "Quasisep kernels can only be added to other Quasisep kernels"
            )
        return Sum(other, self)

    def __mul__(self, other: Kernel | JAXArray) -> Kernel:
        if isinstance(other, tkq.Quasisep):
            return Product(self, other)
        if isinstance(other, Kernel) or jnp.ndim(other) != 0:
            raise ValueError(
                "Quasisep kernels can only be multiplied by scalars and other "
                "Quasisep kernels"
            )
        return Scale(kernel=self, scale=other)

    def __rmul__(self, other: Any) -> Kernel:
        if isinstance(other, tkq.Quasisep):
            return Product(other, self)
        if isinstance(other, Kernel) or jnp.ndim(other) != 0:
            raise ValueError(
                "Quasisep kernels can only be multiplied by scalars and other "
                "Quasisep kernels"
            )
        return Scale(kernel=self, scale=other)

    def power(
        self, f: float | JAXArray, df: float | JAXArray | None = None
    ) -> JAXArray:
        """Compute the power spectral density (PSD) at frequency `f`."""
        return NotImplementedError


class Sum(Quasisep, tkq.Sum):
    """A helper to represent the sum of two quasiseparable kernels"""

    def power(
        self, f: float | JAXArray, df: float | JAXArray | None = None
    ) -> JAXArray:
        """Compute the power spectral density (PSD) at frequency `f`."""
        return self.kernel1.power(f, df) + self.kernel2.power(f, df)


class Product(Quasisep, tkq.Product):
    """A helper to represent the product of two quasiseparable kernels"""

    def power(self, f: float | JAXArray, df: float | JAXArray) -> JAXArray:
        """Compute the power spectral density (PSD) at frequency `f`."""
        return NotImplementedError


class Scale(Quasisep, tkq.Scale):
    """The product of a scalar and a quasiseparable kernel"""

    def power(self, f: float | JAXArray, df: float | JAXArray) -> JAXArray:
        """Compute the power spectral density (PSD) at frequency `f`."""
        return self.kernel.power(f) * jnp.square(self.scale)


class Exp(Quasisep, tkq.Exp):
    """
    An extension of the `tinygp.kernels.quasisep.Exp` kernel, adding a power method.
    """

    def power(
        self, f: float | JAXArray, df: float | JAXArray | None = None
    ) -> JAXArray:
        """Compute the power spectral density (PSD) at frequency `f`."""
        a0 = 1 / self.scale
        sigma_hat2 = 2 * self.sigma**2 * a0
        return sigma_hat2 / (a0**2 + (2 * jnp.pi * f) ** 2)


class Cosine(Quasisep, tkq.Cosine):
    """
    An extension of the `tinygp.kernels.quasisep.Cosine` kernel, adding a power method.
    """

    psd_width: JAXArray | float = eqx.field(
        default_factory=lambda: 0.001 * jnp.ones(())
    )

    def power(
        self, f: float | JAXArray, df: float | JAXArray | None = None
    ) -> JAXArray:
        return (
            0.5
            * self.sigma**2
            * jnp.exp(-(((f - 1 / self.scale) / self.psd_width) ** 2))
        )


class Celerite(Quasisep, tkq.Celerite):
    """
    An extension of the `tinygp.kernels.quasisep.Celerite` kernel, adding a power method.
    """  # noqa: E501

    def power(
        self, f: float | JAXArray, df: float | JAXArray | None = None
    ) -> JAXArray:
        """Compute the power spectral density (PSD) at frequency `f`."""
        w = 2 * jnp.pi * f
        w2 = jnp.square(w)
        ac = self.a * self.c
        bd = self.b * self.d
        c2 = jnp.square(self.c)
        d2 = jnp.square(self.d)

        num = (ac + bd) * (c2 + d2) + (ac - bd) * w2
        denom = jnp.square(w2) + 2 * (c2 - d2) * w2 + jnp.square(c2 + d2)

        return jnp.sqrt(2 / jnp.pi) * num / denom


class Matern32(Quasisep, tkq.Matern32):
    """
    An extension of the `tinygp.kernels.quasisep.Matern32` kernel, adding a power method.
    """  # noqa: E501

    def power(
        self, f: float | JAXArray, df: float | JAXArray | None = None
    ) -> JAXArray:
        """Compute the power spectral density (PSD) at frequency `f`."""
        num = 4.0 * jnp.power(3, 3 / 2) * self.scale
        denom = jnp.square(3 + jnp.square(2.0 * jnp.pi * f * self.scale))
        return self.sigma**2 * num / denom


class Matern52(Quasisep, tkq.Matern52):
    """
    An extension of the `tinygp.kernels.quasisep.Matern52` kernel, adding a power method.
    """  # noqa: E501

    def power(
        self, f: float | JAXArray, df: float | JAXArray | None = None
    ) -> JAXArray:
        """Compute the power spectral density (PSD) at frequency `f`."""
        num = 4.0 * jnp.power(5, 5 / 2) * self.scale
        denom = 0.75 * (5 + jnp.square(2.0 * jnp.pi * f * self.scale)) ** 3
        return self.sigma**2 * num / denom


class SHO(Quasisep, tkq.SHO):
    """
    An extension of the `tinygp.kernels.quasisep.SHO` kernel, adding a power method.
    """  # noqa: E501

    def power(
        self, f: float | JAXArray, df: float | JAXArray | None = None
    ) -> JAXArray:
        """Compute the power spectral density (PSD) at frequency `f`."""
        s0 = self.sigma**2 / (self.quality * self.omega)
        omega = 2.0 * jnp.pi * f
        num = s0 * jnp.power(self.omega, 4)
        denom = jnp.square(jnp.square(omega) - jnp.square(self.omega)) + jnp.square(
            self.omega * omega / self.quality
        )

        return jnp.sqrt(2 / jnp.pi) * num / denom


class Lorentzian(Quasisep):
    r"""The Lorentzian kernel.

    The kernel takes the form:

    .. math::

        k(\tau) = \sigma^2\,\exp(-b\,\tau)\,cos(\omega\,\tau)

    for :math:`\tau = |x_i - x_j|` and :math:`b = \frac{\omega}{2\,Q}`.

    Args:
        omega: The parameter :math:`\omega`.
        quality: The parameter :math:`Q`.
        sigma (optional): The parameter :math:`\sigma`. Defaults to a value of
            1. Specifying the explicit value here provides a slight performance
            boost compared to independently multiplying the kernel with a
            prefactor.
    """

    omega: JAXArray | float
    quality: JAXArray | float
    sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))

    @eqx.filter_jit
    def get_scale(self) -> tuple[JAXArray | float, JAXArray | float]:
        return 2 * self.quality / self.omega, 2 * np.pi / self.omega

    def design_matrix(self) -> JAXArray:
        drw_scale, cos_scale = self.get_scale()
        f = 2 * np.pi / cos_scale
        F1 = jnp.array([[-1 / drw_scale]])
        F2 = jnp.array([[0, -f], [f, 0]])
        return _prod_helper(F1, jnp.eye(F2.shape[0])) + _prod_helper(
            jnp.eye(F1.shape[0]), F2
        )

    def stationary_covariance(self) -> JAXArray:
        drw_scale, cos_scale = self.get_scale()
        a1 = jnp.ones((1, 1))
        a2 = jnp.eye(2)
        return _prod_helper(a1, a2)

    def observation_model(self, X: JAXArray) -> JAXArray:
        del X
        a1 = jnp.array([self.sigma])
        a2 = jnp.array([1.0, 0.0])
        return _prod_helper(a1, a2)

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        drw_scale, cos_scale = self.get_scale()
        dt = X2 - X1
        f = 2 * np.pi / cos_scale
        cos = jnp.cos(f * dt)
        sin = jnp.sin(f * dt)

        a1 = jnp.exp(-dt[None, None] / drw_scale)
        a2 = jnp.array([[cos, sin], [-sin, cos]])

        return _prod_helper(a1, a2)

    def power(
        self, f: float | JAXArray, df: float | JAXArray | None = None
    ) -> JAXArray:
        f0 = self.omega / (2 * np.pi)
        num = jnp.square(self.sigma) * self.quality * f0
        denom = jnp.square(f0) + 4 * jnp.square(self.quality) * jnp.square(f - f0)
        pre_fix = np.sqrt(2 / np.pi) / (2 * np.pi)
        return pre_fix * num / denom


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
    def power(
        self, f: float | JAXArray, df: float | JAXArray | None = None
    ) -> JAXArray:
        arparams = jnp.append(jnp.array(self.alpha), 1.0)
        maparams = jnp.array(self.beta)

        complex_dtype = dtypes.to_complex_dtype(arparams.dtype)

        # init terms
        num_terms = jnp.zeros(1, dtype=complex_dtype)
        denom_terms = jnp.zeros(1, dtype=complex_dtype)

        for i, param in enumerate(maparams):
            num_terms += param * jnp.power(2 * jnp.pi * f * (1j), i)

        for k, param in enumerate(arparams):
            denom_terms += param * jnp.power(2 * jnp.pi * f * (1j), k)

        num = jnp.abs(jnp.power(num_terms, 2))
        denom = jnp.abs(jnp.power(denom_terms, 2))

        return (num / denom)[0]


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


class MultibandLowRank(tkq.Wrapper):
    """
    A multiband kernel implementating a low-rank Kronecker covariance structure.

    The specific form of the cross-band Kronecker covariance matrix is given by
    Equation 13 of `Gordon et al. (2020) <https://arxiv.org/pdf/2007.05799>`_.
    The implementation is inspired by this `tinygp` tutorial <https://tinygp.readthedocs.io/en/stable/tutorials/quasisep-custom.html#multivariate-quasiseparable-kernels>`_.

    Args:
        params: A dictionary of string and array pairs, which are used in the
            `observational_model` method to describe the cross-band covariance.
    """

    params: dict[str, JAXArray]

    def coord_to_sortable(self, X) -> JAXArray:
        return X[0]

    def observation_model(self, X) -> JAXArray:
        amplitudes = self.params["amplitudes"]
        return amplitudes[X[1]] * self.kernel.observation_model(
            self.coord_to_sortable(X)
        )

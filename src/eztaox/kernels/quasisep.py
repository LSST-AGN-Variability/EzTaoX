"""Quasiseparable kernels.

Scalable kernels exploiting the quasiseparable structure in the relevant
matrices to achieve a O(N) scaling.

This module extends the `tinygp.kernels.quasisep` module.
"""

from __future__ import annotations

from functools import partial
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
    """An extension of the `tinygp.kernels.quasisep.Quasisep` kernel.

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
    """A helper to represent the sum of two quasiseparable kernels."""

    def power(
        self, f: float | JAXArray, df: float | JAXArray | None = None
    ) -> JAXArray:
        """Compute the power spectral density (PSD) at frequency `f`."""
        return self.kernel1.power(f, df) + self.kernel2.power(f, df)


class Product(Quasisep, tkq.Product):
    """A helper to represent the product of two quasiseparable kernels."""

    def power(self, f: float | JAXArray, df: float | JAXArray) -> JAXArray:
        """Compute the power spectral density (PSD) at frequency `f`."""
        return NotImplementedError


class Scale(Quasisep, tkq.Scale):
    """The product of a scalar and a quasiseparable kernel."""

    def power(self, f: float | JAXArray, df: float | JAXArray) -> JAXArray:
        """Compute the power spectral density (PSD) at frequency `f`."""
        return self.kernel.power(f) * jnp.square(self.scale)


class Exp(Quasisep, tkq.Exp):
    """Extends the `tinygp.kernels.quasisep.Exp` kernel, adding a power method."""

    def power(
        self, f: float | JAXArray, df: float | JAXArray | None = None
    ) -> JAXArray:
        """Compute the power spectral density (PSD) at frequency `f`."""
        a0 = 1 / self.scale
        sigma_hat2 = 2 * self.sigma**2 * a0
        return sigma_hat2 / (a0**2 + (2 * jnp.pi * f) ** 2)


class Cosine(Quasisep, tkq.Cosine):
    """Extends the `tinygp.kernels.quasisep.Cosine` kernel, adding a power method."""

    psd_width: JAXArray | float = eqx.field(
        default_factory=lambda: 0.001 * jnp.ones(())
    )

    def power(
        self, f: float | JAXArray, df: float | JAXArray | None = None
    ) -> JAXArray:
        """Compute the power spectral density (PSD) at frequency `f`."""
        return (
            0.5
            * self.sigma**2
            * jnp.exp(-(((f - 1 / self.scale) / self.psd_width) ** 2))
        )


class Celerite(Quasisep, tkq.Celerite):
    """Extends the `tinygp.kernels.quasisep.Celerite` kernel, adding a power method."""

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
    """Extends the `tinygp.kernels.quasisep.Matern32` kernel, adding a power method."""

    def power(
        self, f: float | JAXArray, df: float | JAXArray | None = None
    ) -> JAXArray:
        """Compute the power spectral density (PSD) at frequency `f`."""
        num = 4.0 * jnp.power(3, 3 / 2) * self.scale
        denom = jnp.square(3 + jnp.square(2.0 * jnp.pi * f * self.scale))
        return self.sigma**2 * num / denom


class Matern52(Quasisep, tkq.Matern52):
    """Extends the `tinygp.kernels.quasisep.Matern52` kernel, adding a power method."""

    def power(
        self, f: float | JAXArray, df: float | JAXArray | None = None
    ) -> JAXArray:
        """Compute the power spectral density (PSD) at frequency `f`."""
        num = 4.0 * jnp.power(5, 5 / 2) * self.scale
        denom = 0.75 * (5 + jnp.square(2.0 * jnp.pi * f * self.scale)) ** 3
        return self.sigma**2 * num / denom


class SHO(Quasisep, tkq.SHO):
    """Extends the `tinygp.kernels.quasisep.SHO` kernel, adding a power method."""

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
        """Scale of the Lorentzian."""
        return 2 * self.quality / self.omega, 2 * np.pi / self.omega

    def design_matrix(self) -> JAXArray:  # noqa: D102
        # TODO: Write docstring.
        drw_scale, cos_scale = self.get_scale()
        f = 2 * np.pi / cos_scale
        F1 = jnp.array([[-1 / drw_scale]])
        F2 = jnp.array([[0, -f], [f, 0]])
        return _prod_helper(F1, jnp.eye(F2.shape[0])) + _prod_helper(
            jnp.eye(F1.shape[0]), F2
        )

    def stationary_covariance(self) -> JAXArray:
        """The variance of the kernel at :math:`t=0`."""
        drw_scale, cos_scale = self.get_scale()
        a1 = jnp.ones((1, 1))
        a2 = jnp.eye(2)
        return _prod_helper(a1, a2)

    def observation_model(self, X: JAXArray) -> JAXArray:  # noqa: D102
        # TODO: Write docstring.
        del X
        a1 = jnp.array([self.sigma])
        a2 = jnp.array([1.0, 0.0])
        return _prod_helper(a1, a2)

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:  # noqa: D102
        # TODO: Write docstring.
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
        """Compute the power spectral density (PSD) at frequency `f`."""
        f0 = self.omega / (2 * np.pi)
        num = jnp.square(self.sigma) * self.quality * f0
        denom = jnp.square(f0) + 4 * jnp.square(self.quality) * jnp.square(f - f0)
        pre_fix = np.sqrt(2 / np.pi) / (2 * np.pi)
        return pre_fix * num / denom


class CARMA(Quasisep):
    r"""A continuous-time autoregressive moving-average kernel.

    This kernel represents a CARMA(:math:`p, q`) process in companion-form
    state-space notation so it can be used with the quasiseparable solvers in
    `tinygp`.

    The autoregressive polynomial is parameterized in ascending power order as

    .. math::

        \alpha(D) = \alpha_0 + \alpha_1 D + \cdots + \alpha_{p-1} D^{p-1} + D^p,

    where the leading coefficient of :math:`D^p` is fixed to 1 and supplied
    implicitly. The moving-average polynomial is

    .. math::

        \beta(D) = \beta_0 + \beta_1 D + \cdots + \beta_q D^q.

    The corresponding power spectral density (PSD) is

    .. math::

        P(\omega) = \sigma^2\,\frac{|\sum_{q} \beta_q\,(i\,\omega)^q|^2}{|\sum_{p}
            \alpha_p\,(i\,\omega)^p|^2}

    following Equation 1 in `Kelly et al. (2014)
    <https://arxiv.org/abs/1402.5978>`_, where :math:`\alpha_p` and :math:`\beta_0`
    are set to 1. In this implementation, we absorb :math:`\sigma` into the
    definition of the :math:`\beta` parameters. That is,
    :math:`\beta_{\mathrm{new}} = \beta\,\sigma`.

    Args:
        alpha: Autoregressive coefficients in ascending power order, excluding
            the leading coefficient fixed to 1.
        beta: Moving-average coefficients :math:`[\beta_0, \ldots, \beta_q]`
            in ascending power order.
        sigma_w: Standard deviation of the white-noise driving term used when
            constructing the stationary state covariance.
    """

    alpha: JAXArray  # [a0, ..., ap-1]
    beta: JAXArray  # [b0, ..., bq]
    sigma_w: float = eqx.field(default=1.0, static=True)

    @classmethod
    def from_quads(
        cls,
        alpha_quads: JAXArray | NDArray,
        beta_quads: JAXArray | NDArray,
        beta_mult: JAXArray | NDArray,
    ) -> CARMA:
        r"""Construct a CARMA kernel using the roots of its characteristic polynomials.

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

    @property
    def arroots(self) -> JAXArray:
        """Return the autoregressive roots sorted by real part."""
        return carma_roots(jnp.append(self.alpha, 1.0))

    def _companion_eigenvectors(self, arroots) -> JAXArray:
        """Construct eigenvectors of the transposed companion matrix."""
        p = self.alpha.shape[0]
        complex_dtype = dtypes.to_complex_dtype(arroots.dtype)

        vecs = jnp.zeros((p, p), dtype=complex_dtype)
        vecs = vecs.at[-1, :].set(jnp.ones(p, dtype=complex_dtype))

        for row in range(p - 2, -1, -1):
            vecs = vecs.at[row, :].set(arroots * vecs[row + 1, :] + self.alpha[row + 1])

        return vecs

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def _padded_ma(beta, p):
        """Pad moving-average coefficients with zeros up to order ``p``."""
        # beta = [b0, ..., bq]
        beta = jnp.asarray(beta)
        h = jnp.zeros(p)
        h = h.at[: beta.shape[0]].set(beta)
        return h

    @jax.jit
    def _companion_transition(self, dt):
        """Evaluate the CAR companion-form transition matrix over a lag ``dt``."""
        dt = jnp.asarray(dt)
        p = self.alpha.shape[0]
        arroots = self.arroots

        if p == 1:
            return jnp.exp(-self.alpha[0] * dt)[None, None]

        # companion_matrix(alpha).T is diagonalizable by the AR roots when they are
        # distinct; this constructs the transition without calling matrix expm.
        exp_diag = jnp.exp(arroots * dt)
        vecs = self._companion_eigenvectors(arroots)
        vecs_inv = jnp.linalg.inv(vecs)
        transition = vecs @ (exp_diag[:, None] * vecs_inv)
        return transition.real

    def design_matrix(self):
        """Return the companion-form drift matrix for the latent CAR state."""
        p = self.alpha.shape[0]
        if p == 1:
            return jnp.array([[-self.alpha[0]]])

        F = jnp.zeros((p, p))
        F = F.at[jnp.arange(p - 1), jnp.arange(1, p)].set(1.0)
        F = F.at[-1, :].set(-self.alpha)
        return F

    def observation_model(self, X):
        """Return the observation vector that maps the state to the process value."""
        del X
        p = self.alpha.shape[0]
        return self._padded_ma(self.beta, p)

    def stationary_covariance(self):
        """Return the stationary covariance of the latent companion-form state."""
        return carma_root_stationary_covariance(self.arroots, self.sigma_w)

    def transition_matrix(self, X1, X2):
        """Return the state transition matrix between two one-dimensional inputs."""
        dt = X2 - X1
        return self._companion_transition(dt)

    @jax.jit
    def power(
        self, f: float | JAXArray, df: float | JAXArray | None = None
    ) -> JAXArray:
        """Compute the power spectral density (PSD) at frequency `f`."""
        del df
        arparams = jnp.append(jnp.asarray(self.alpha), 1.0)
        maparams = jnp.asarray(self.beta)

        complex_dtype = dtypes.to_complex_dtype(arparams.dtype)
        num_terms = jnp.zeros(1, dtype=complex_dtype)
        denom_terms = jnp.zeros(1, dtype=complex_dtype)

        for i, param in enumerate(maparams):
            num_terms += param * jnp.power(2 * jnp.pi * f * (1j), i)

        for k, param in enumerate(arparams):
            denom_terms += param * jnp.power(2 * jnp.pi * f * (1j), k)

        num = jnp.abs(jnp.power(num_terms, 2))
        denom = jnp.abs(jnp.power(denom_terms, 2))
        return num[0] / denom[0]


@jax.jit
def carma_root_stationary_covariance(
    arroots: JAXArray,
    sigma: JAXArray | float = 1.0,
) -> JAXArray:
    r"""Compute the CARMA state stationary covariance from AR roots.

    This implements the closed-form expression

    .. math::

        V_{ij} = -
        \,\sigma^2 \, \sum_{k=1}^{p}
        \frac{r_k^i(-r_k)^j}{
            2\mathrm{Re}(r_k)
            \prod_{l=1, l\ne k}^{p}(r_l-r_k)(r_l^*+r_k)
        }

    where :math:`r_k` are the autoregressive roots and :math:`i,j \in [0, p-1]`.

    Args:
        arroots: The roots of the autoregressive characteristic polynomial.
        sigma: The driving-noise amplitude :math:`\sigma`.

    Returns:
        The :math:`p \times p` matrix defined by the root-based covariance
        expression above.
    """
    arroots = jnp.asarray(arroots)
    sigma = jnp.asarray(sigma)
    complex_dtype = dtypes.to_complex_dtype(arroots.dtype)

    p = arroots.shape[0]
    idx = jnp.arange(p)
    powers = idx[:, None]

    root_diff = arroots[:, None] - arroots[None, :]
    conj_sum = jnp.conj(arroots)[:, None] + arroots[None, :]
    off_diag = ~jnp.eye(p, dtype=bool)
    denom_prod = jnp.prod(
        jnp.where(
            off_diag,
            root_diff * conj_sum,
            jnp.ones((p, p), dtype=complex_dtype),
        ),
        axis=0,
    )
    denom = 2.0 * jnp.real(arroots) * denom_prod

    # Rewrite the k-sum as a weighted matrix product to avoid allocating a
    # full (p, p, p) tensor of intermediate terms.
    left = jnp.power(arroots[None, :], powers) / denom[None, :]
    right = jnp.power((-arroots)[None, :], powers)
    cov = -(sigma**2) * (left @ right.T)
    cov = 0.5 * (cov + cov.T.conj())
    return cov.real


@jax.jit
def carma_roots(poly_coeffs: JAXArray) -> JAXArray:
    """Compute roots of a CARMA characteristic polynomial.

    Args:
        poly_coeffs: Polynomial coefficients in ascending power order, so the
            first element is the constant term.

    Returns:
        The polynomial roots sorted by their real part.
    """
    roots = jnp.roots(poly_coeffs[::-1], strip_zeros=False)
    return roots[jnp.argsort(roots.real)]


@jax.jit
def carma_quads2poly(quads_coeffs: JAXArray) -> JAXArray:
    """Expand a product of CARMA quadratic factors into a full polynomial.

    Args:
        quads_coeffs: Constant and linear coefficients of the quadratic factors
            used by the Kelly et al. parameterization. The last entry is the
            multiplier for the highest-order term of the reconstructed
            polynomial.

    Returns:
        Polynomial coefficients in ascending power order.
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
    """Factorize a CARMA polynomial into quadratic and linear factors.

    Args:
        poly_coeffs: Coefficients of the input characteristic polynomial. The
            first entry corresponds to the constant term.

    Returns:
        Constant and linear coefficients for the factorized quadratic blocks,
        followed by the multiplier for the highest-order term.
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
    r"""Compute exponential-basis coefficients of the CARMA autocovariance.

    Args:
        arroots: The roots of the autoregressive characteristic polynomial.
        arparam: Autoregressive coefficients in ascending power order.
        maparam: Moving-average coefficients :math:`[\beta_0, \ldots, \beta_q]`
            in ascending power order.

    Returns:
        The coefficients of the exponential expansion of the ACVF, with one
        coefficient per autoregressive root.
    """

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


class MultibandLowRank(tkq.Wrapper):
    """A multiband kernel implementating a low-rank Kronecker covariance structure.

    The specific form of the cross-band Kronecker covariance matrix is given by
    Equation 13 of `Gordon et al. (2020) <https://arxiv.org/pdf/2007.05799>`_.
    The implementation is inspired by `this tinygp tutorial <https://tinygp.readthedocs.io/en/stable/tutorials/quasisep-custom.html#multivariate-quasiseparable-kernels>`_.

    Args:
        params: A dictionary of string and array pairs, which are used in the
            `observational_model` method to describe the cross-band covariance.
    """

    params: dict[str, JAXArray]

    def coord_to_sortable(self, X) -> JAXArray:
        """Extract the time-sortable component of the coordinates."""
        return X[0]

    def observation_model(self, X) -> JAXArray:  # noqa: D102
        # TODO: Write docstring.
        amplitudes = self.params["amplitudes"]
        return amplitudes[X[1]] * self.kernel.observation_model(
            self.coord_to_sortable(X)
        )

from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.special import gamma
from jax.typing import ArrayLike
from sympy import Expr, S, Symbol, exp, integrate, log, oo, sqrt
from sympy.functions.special.gamma_functions import loggamma
from sympy.functions.special.polynomials import laguerre_poly

jax.config.update("jax_enable_x64", True)


def scaled_ho_basis_funcs_upto_n(
    r: ArrayLike, max_n: int, l: int, dtype: type = jnp.float64
):
    """All HO radial basis states up to `max_n` with b = ℏω = 1.

    Args:
    r (ArrayLike): Array of shape (M, ) containing points on which the basis states are defined.
    max_n (int): Maximum radial quantum number.
    l (int): Orbital angular momentum quantum number.
    dtype (type): The float precision to use. Default is jnp.float64.

    Returns:
    Array of shape (max_n + 1, len(r)) containing all the basis states from 0 to max_n.
    """

    x = r.astype(dtype)
    x2 = x**2
    exp_x2 = jnp.exp(-0.5 * x2)

    psi0 = (x**l) * jnp.sqrt(2 / gamma(l + 1.5)) * exp_x2
    psi1 = (x**l) * jnp.sqrt(2 / gamma(l + 2.5)) * (1.5 + l - x2) * exp_x2

    wfs = jnp.zeros((max_n + 1, len(r)), dtype=dtype)
    wfs = wfs.at[0].set(psi0)
    wfs = wfs.at[1].set(psi1)

    def init_body(n, wfs_):
        psi1_ = jnp.sqrt(n / (n + l + 0.5)) * (2 + (l - 0.5 - x2) / n) * wfs_[n - 1, :]
        psi2_ = (
            jnp.sqrt(n * (n - 1) / ((n + l) ** 2 - 0.25))
            * (1 + (l - 0.5) / n)
            * wfs_[n - 2, :]
        )
        return wfs_.at[n].set(psi1_ - psi2_)

    wfs = jax.lax.fori_loop(2, max_n + 1, init_body, wfs)
    return wfs


def symbolic_ho_basis_func(
    r: Symbol | float, n: Symbol | int, l: Symbol | int, b: Symbol | float
) -> Expr:
    """Symbolic representation of a basis state.

    Args:
    r (Symbol | float): Point at which basis function is to be evaluated.
    n (Symbol | int): Radial quantum number.
    l (Symbol | int): Orbital angular momentum quantum number.
    b (Symbol | float): Length parameter of the oscillator.

    Returns:
    The symbolic expression of the basis state.
    """
    norm = exp(
        S.Half
        * (log(2) + loggamma(n + S.One) - 3 * log(b) - loggamma(n + l + 3 * S.Half))
    )
    x = r / b
    x2 = x**2
    spatial = x**l * exp(-x2 * S.Half) * laguerre_poly(n, x2, l + S.Half)
    return norm * spatial


def symbolic_ho_matrix_element(
    f: Callable,
    bra_n: Symbol | int,
    ket_n: Symbol | int,
    bra_l: Symbol | int,
    ket_l: Symbol | int,
    b: Symbol | float,
) -> Expr:
    """Symbolic expression for the matrix element〈n'l'|f|nl〉.

    Args:
    f (Callable): Operator implement in the form of f(r).
    bra_n (Symbol | int): Radial quantum number of bra state.
    ket_n (Symbol | int): Radial quantum number of ket state.
    bra_l (Symbol | int): Orbital angular momentum quantum number of bra state.
    ket_l (Symbol | int): Orbital angular momentum quantum number of ket state.
    b (Symbol | float): Length parameter of oscillator.

    Returns:
    The symbolic or rational expression of the matrix element.
    """
    norm = exp(
        S.Half
        * (
            loggamma(bra_n + S.One)
            + loggamma(ket_n + S.One)
            - loggamma(bra_n + bra_l + 3 * S.Half)
            - loggamma(ket_n + ket_l + 3 * S.Half)
        )
    )

    x = Symbol("x")
    g = (
        x ** ((bra_l + ket_l + 1) * S.Half)
        * exp(-x)
        * laguerre_poly(bra_n, x, bra_l + S.Half)
        * laguerre_poly(ket_n, x, ket_l + S.Half)
        * f(b * sqrt(x))
    )
    return norm * integrate(g, (x, 0, oo))

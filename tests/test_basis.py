import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from rho.basis import scaled_ho_basis_funcs_upto_n

jax.config.update("jax_enable_x64", True)


def test_ho_wfs_orthonormality_array():
    xi, xf, npts = 0, 1, 18001
    x = jnp.linspace(xi, xf, npts, endpoint=False)
    r = x / (1 - x)
    dr = 1 / (1 - x) ** 2

    max_n = 200
    l = 10
    b = 2.0

    wfs = scaled_ho_basis_funcs_upto_n(r / b, max_n, l, dtype=np.float64)
    assert wfs.shape == (max_n + 1, npts)

    weights = jnp.ones(npts)
    weights = weights.at[0].set(0.5)
    weights = weights.at[-1].set(0.5)
    weights = weights / npts

    weighted_wfs = wfs * (weights * r**2 * dr / b**3)
    norms = weighted_wfs @ wfs.transpose()
    assert_almost_equal(np.asarray(norms), np.eye(max_n + 1), decimal=5)

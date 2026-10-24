"""
eta_sample.py  (JAX, single precision)

Latent-factor step of the Gibbs sampler for the cumulative shrinkage
process factor model (Legramanti, Durante & Dunson, 2020, Section 3.2,
Algorithm 1, step 3): sample the latent factors eta_i (i = 1, ..., n)
independently from their Gaussian full conditional

    eta_i | B, Omega, x_i ~ N_H(m_i, S),

    S   = (I_H + B^T Omega^{-1} B)^{-1},
    m_i = S B^T Omega^{-1} x_i.

This conditional is identical to that of the L_{1/2} factor model, so
the module is a port of GS_LH_JAX/eta_sample.py; only the notation (H
for the truncation index) differs.

Sampling scheme: perturb-and-solve (Rue & Held, 2005, GMRF sampling).
With C = B^T Omega^{-1/2}, Lambda = C C^T + I_H and z1 ~ N(0, I_p),
z2 ~ N(0, I_H),

    Lambda^{-1} (B^T Omega^{-1} x_i + C z1 + z2) ~ N_H(m_i, S),

an EXACT draw: the precision system Lambda is I + PSD by construction,
so it is solved by plain Cholesky with no fallback branch.  One
factorization produces all n draws simultaneously, and the data and
noise contributions are folded into a single matmul via
C (x_i/s + z1_i) = B^T Omega^{-1} x_i + C z1_i.

Orientation: the latent factors are returned as a (H, n) matrix whose
ith COLUMN is eta_i (transpose of the paper's n x H eta matrix); this
gives eta @ eta.T and eta @ X_{.j} without transposes in B_sample.py,
and B @ eta directly in Sigma2_sample.py.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

__all__ = ["eta_sample"]

_F32 = jnp.float32


def eta_sample(
    key: jax.Array,
    X: jax.Array,
    B: jax.Array,
    sigma2: jax.Array,
) -> jax.Array:
    r"""One exact draw of all latent factors eta_i | B, Omega, x_i.

    Parameters
    ----------
    key : jax.random.PRNGKey
    X : array-like, shape (n, p)
        Data matrix.
    B : array-like, shape (p, H)
        Factor loading matrix at the current iteration.
    sigma2 : array-like, shape (p,)
        DIAGONAL VARIANCES sigma_j^2 of Omega.  Floored at
        tiny ** 0.5 (i.e. sigma_j >= tiny ** 0.25) before the square
        root, so a perfect fit cannot overflow the divisions or the
        C @ C.T product; the perturbation binds only below
        ~1.1e-19 in float32.

    Returns
    -------
    jnp.ndarray, shape (H, n), float32
        Column i is one draw of eta_i | B, Omega, x_i; columns are
        independent across i.
    """
    X = jnp.asarray(X, dtype=_F32)
    B = jnp.asarray(B, dtype=_F32)
    sigma2 = jnp.asarray(sigma2, dtype=_F32)

    n = X.shape[0]
    # sigma_j >= tiny**0.25: one more squaring happens in C @ C.T, and
    # (B / tiny**0.25)^2 ~ 9e18 per entry, summed over p stays ~1e22,
    # far below the 3.4e38 float32 ceiling.
    s = jnp.sqrt(jnp.maximum(sigma2, jnp.finfo(_F32).tiny ** 0.5))

    C = B.T / s                              # B^T Omega^{-1/2},  (H, p)
    Lam = C @ C.T
    Lam = Lam + jnp.eye(B.shape[1], dtype=_F32)      # Lam = C C^T + I_H

    # Perturbed right-hand side folded into ONE matmul:
    #   rhs_i = C (x_i / s + z1_i) + z2_i
    #         = B^T Omega^{-1} x_i + C z1_i + z2_i,
    # whose noise covariance is C C^T + I = Lam, so the solved output
    # has covariance Lam^{-1} = S around the exact mean.
    key_z1, key_z2 = jax.random.split(key)
    z1 = jax.random.normal(key_z1, X.shape, dtype=_F32)
    E2 = z1 + X * (1.0 / s)[None, :]         # x_i / s + z1_i,   (n, p)
    rhs = C @ E2.T + jax.random.normal(key_z2, (B.shape[1], n), dtype=_F32)

    # Plain Cholesky: Lam = I + PSD by construction, so no fallback
    # branch.  The driver audits finiteness of the chain per chunk.
    L = jnp.linalg.cholesky(Lam)
    return jsla.cho_solve((L, True), rhs)

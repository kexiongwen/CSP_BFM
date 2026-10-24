"""
Sigma2_sample.py  (JAX, single precision)

Idiosyncratic-variance step of the Gibbs sampler for the cumulative
shrinkage process factor model (Legramanti, Durante & Dunson, 2020,
Section 3.2, Algorithm 1, step 2): sample the idiosyncratic error
variances sigma_j^2, j = 1, ..., p, independently from their
inverse-gamma full conditional

    sigma_j^2 | X_{.j}, B_j, eta
        ~ IG( a_sigma + n/2,
              b_sigma + (1/2) * || X_{.j} - eta^T B_j ||^2 ).

This conditional is identical to that of the L_{1/2} factor model; the
module is a simplified port of GS_LH_JAX/Sigma2_sample.py (the
chi-square Gamma approximation machinery is dropped: jax.random.gamma
is a pure XLA primitive, so the exact draw is always available and no
accuracy-for-speed tradeoff is needed).

Sampling scheme: reciprocal-gamma identity.  With residuals
RSS_j = || X_{.j} - eta^T B_j ||^2 and g_j ~ Ga(a_sigma + n/2, 1)
(shape-rate), independent across j,

    sigma_j^2 = (b_sigma + RSS_j / 2) / g_j
              ~ IG(a_sigma + n/2, b_sigma + RSS_j/2),

since c / g ~ IG(alpha, c) when g ~ Ga(alpha, 1).  The residual sum of
squares is formed directly (no ||X||^2 - 2 B^T eta X + ... identity),
avoiding catastrophic cancellation when the fit is good; b_sigma > 0
keeps every draw strictly positive.
"""

import jax
import jax.numpy as jnp

__all__ = ["sigma2_sample"]

_F32 = jnp.float32


def sigma2_sample(
    key: jax.Array,
    X: jax.Array,
    B: jax.Array,
    eta: jax.Array,
    a_sigma: float,
    b_sigma: float,
) -> jax.Array:
    r"""One exact draw of the diagonal variances sigma_j^2, j = 1..p.

    Parameters
    ----------
    key : jax.random.PRNGKey
    X : array-like, shape (n, p)
        Data matrix.
    B : array-like, shape (p, H)
        Loading matrix at the current iteration.
    eta : array-like, shape (H, n)
        Latent factors, column i is eta_i (output of eta_sample()).
    a_sigma, b_sigma : float
        Fixed inverse-gamma hyperparameters, a_sigma > 0, b_sigma > 0.

    Returns
    -------
    jnp.ndarray, shape (p,), float32
        The drawn variances sigma_j^2 (diagonal of Omega), independent
        across j and strictly positive.
    """
    X = jnp.asarray(X, dtype=_F32)
    B = jnp.asarray(B, dtype=_F32)
    eta = jnp.asarray(eta, dtype=_F32)

    # RSS_j = || X_{.j} - eta^T B_j ||^2, row j of the (p, n) residual
    rss = jnp.sum((X.T - B @ eta) ** 2, axis=1)

    # sigma_j^2 = rate_j / g_j,  g_j ~ Ga(a_sigma + n/2, 1) iid across j
    rate = b_sigma + 0.5 * rss
    g = jax.random.gamma(key, a_sigma + 0.5 * X.shape[0],
                         shape=rate.shape, dtype=_F32)
    return rate / g

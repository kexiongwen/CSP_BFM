"""
B_sample.py  (JAX, single precision)

Loading-matrix step of the Gibbs sampler for the cumulative shrinkage
process factor model (Legramanti, Durante & Dunson, 2020, Section 3.2,
Algorithm 1, step 1): sample each row B_j (j = 1, ..., p) of the
loading matrix B (p x H) independently from its Gaussian full
conditional

    B_j | X_{.j}, eta, theta, sigma_j^2 ~ N_H(Lam_j^{-1} b_j, Lam_j^{-1}),

    Lam_j = eta eta^T / sigma_j^2 + diag(1/theta_1, ..., 1/theta_H),
    b_j   = eta X_{.j} / sigma_j^2.

Conventions (shared with the whole pipeline):
    D      : (H,) column prior STANDARD DEVIATIONS sqrt(theta_h) --
             exactly the first row of the (p, H) matrix returned by
             shrinkage().  Under the CSP prior the prior variance is
             shared by all rows within a column (theta_h), so D is a
             vector, unlike the elementwise (p, K) scales of the
             L_{1/2} model.
    eta    : (H, n) latent factor matrix, column i is eta_i (as
             returned by eta_sample.py), matching eta eta^T / eta X_{.j}.
    sigma2 : (p,) diagonal variances of Omega.
    return : (p, H) matrix, row j is the drawn B_j.

Sampling scheme: whitened perturb-and-solve, batched over all p rows
(no Python loop).  Write W = diag(d) and G = eta eta^T.  G is formed
ONCE (O(n H^2)) and shared by all rows; the whitened system

    M_j = (d d^T) . (G / sigma_j^2) + I_H = W Lam_j W

is built by pure broadcasting -- O(p H^2) in memory and flops.  The
perturbed right-hand side

    phi_j = d * eta (X_{.j} + sigma_j z_j) / sigma_j^2 + eps_j,
    z_j ~ N(0, I_n), eps_j ~ N(0, I_H),

has noise covariance (d d^T) . (G / sigma_j^2) + I_H = M_j, and the
draw is mapped back by B_j = d * gamma_j with gamma_j = M_j^{-1}
phi_j.  Exactness:

    mean: W M^{-1} W b = W (W Lam W)^{-1} W b = Lam^{-1} b,
    cov : W M^{-1} M M^{-1} W = W (W Lam W)^{-1} W = Lam^{-1}.

Numerical merits:
    * the raw precision Lam_j has diagonal 1/theta_h, which is large
      on heavily shrunk columns (1/theta_infty ~ 20, more on small IG
      draws); whitening turns those into near-zero data terms in
      M_j = I + PSD;
    * d_h -> 0 decouples coordinate h (M_j[h, h] -> 1, gamma_h ~
      N(0, 1), B_jh = d_h gamma_h -> 0): tiny spike variances cannot
      produce inf/NaN;
    * M_j = I + PSD by construction, so the batched Cholesky needs no
      fallback branch (the driver audits finiteness per chunk).

Port of GS_LH_JAX/B_sample.py with the per-element scales replaced by
the column-shared CSP variances (which makes the whitening simpler:
a single (H, H) outer product instead of p of them).
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

__all__ = ["B_sample"]

_F32 = jnp.float32


def B_sample(
    key: jax.Array,
    X: jax.Array,
    D: jax.Array,
    eta: jax.Array,
    sigma2: jax.Array,
) -> jax.Array:
    r"""One exact draw of the loading matrix B given (eta, D, sigma2).

    Parameters
    ----------
    key : jax.random.PRNGKey
    X : array-like, shape (n, p)
        Data matrix.
    D : array-like, shape (H,)
        Column prior standard deviations sqrt(theta_h) (any single row
        of the (p, H) matrix returned by shrinkage(); the broadcast
        form is accepted too).  Must be strictly positive (guaranteed by
        the shrinkage step: spike columns carry sqrt(theta_inf), slab
        columns a positive IG draw).
    eta : array-like, shape (H, n)
        Latent factors, column i is eta_i (output of eta_sample()).
    sigma2 : array-like, shape (p,)
        Diagonal variances sigma_j^2 of Omega (output of
        Sigma2_sample.py).  Floored at tiny ** 0.5 before use, so a
        perfect fit cannot overflow the divisions.

    Returns
    -------
    jnp.ndarray, shape (p, H), float32
        The drawn loading matrix; rows are independent across j.
    """
    X = jnp.asarray(X, dtype=_F32)
    D = jnp.asarray(D, dtype=_F32)
    eta = jnp.asarray(eta, dtype=_F32)
    sigma2 = jnp.asarray(sigma2, dtype=_F32)

    H, n = eta.shape
    p = X.shape[1]
    d = D if D.ndim == 1 else D[0]                     # (H,)

    s2 = jnp.maximum(sigma2, jnp.finfo(_F32).tiny ** 0.5)

    # ---- whitened system M_j = (d d^T) . (G / sigma_j^2) + I_H ---------------
    # G = eta eta^T formed once and reused by all p rows.
    G = eta @ eta.T                                    # (H, H)
    M = (d[:, None] * d[None, :])[None, :, :] * (G / s2[:, None, None])
    M = M + jnp.eye(H, dtype=_F32)                     # M_j += I_H

    # ---- perturbed right-hand side -------------------------------------------
    # phi_j = d * eta (X_{.j} + sigma_j z_j) / sigma_j^2 + eps_j; the noise
    # d * (eta z_j) / sigma_j + eps_j has covariance M_j.
    key_z, key_e = jax.random.split(key)
    Z = jax.random.normal(key_z, (p, n), dtype=_F32)
    E = X + Z.T * jnp.sqrt(s2)[None, :]                # X_{.j} + sigma_j z_j
    phi = d[None, :] * ((eta @ E) / s2[None, :]).T \
        + jax.random.normal(key_e, (p, H), dtype=_F32)

    # ---- solve and map back: B_j = d * M_j^{-1} phi_j ------------------------
    # Plain Cholesky: M_j = I + PSD by construction, so no fallback branch.
    L = jnp.linalg.cholesky(M)                         # (p, H, H)
    gamma = jsla.cho_solve((L, True), phi[..., None]).squeeze(-1)
    return d[None, :] * gamma

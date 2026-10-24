"""Monte Carlo posterior moments of the model covariance matrix.

Given Gibbs draws (B_s, sigma_s^2), s = 1, ..., S, of the CSP factor
model (:func:`CSP_JAX.GS_CSP` / :class:`CSP_JAX.GibbsCSP`), where

    Sigma^(s) = B_s B_s^T + diag(sigma_s^2),

the functions below return the Monte Carlo sample mean, the entrywise
sample variance and arbitrary entry-pair sample covariances (ddof = 0).
These are consistent estimates of the true posterior moments (up to MCMC
error and autocorrelation).

Because the full p x p variance matrix costs O(S p^2 H) and becomes
expensive for large S (e.g. S = 10000 with p ~ 2000), it is guarded by a
compute budget and can be computed from a deterministic equidistant
subsample; the cheap diagonal variance and pair covariances always use
all draws.

Ported from the ``bfm`` package of the L_{1/2} paper (LH_BFM,
``bfm/_posterior.py``); the variational-moment half of that module is
not ported, since this package has no CAVI estimator.  The truncation
index is named H here (the paper's notation), not K.
"""

from __future__ import annotations

import operator

import numpy as np

__all__ = [
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_MAX_DRAWS",
    "DEFAULT_MAX_MOMENT_FLOPS",
    "suggest_max_draws",
    "montecarlo_sigma_mean",
    "montecarlo_sigma_variance",
    "montecarlo_sigma_diag_variance",
    "montecarlo_cov_sigma_pairs",
]

# Default cap on the number of entries of a returned p x p matrix.
# 2e7 float64 entries ~ 160 MB, i.e. p <= 4472.
DEFAULT_MAX_ENTRIES = 20_000_000

# Gibbs-draw Monte Carlo moments: default cap on the number of draws used for
# the full p x p variance matrix, and the default compute budget.  A full
# scan costs about 2 p^2 H multiply-adds per draw plus O(p^2) accumulation;
# the budget keeps the default property from silently running for minutes on
# a large S (e.g. S = 10000, p ~ 2000).
DEFAULT_MAX_DRAWS = 2_000
DEFAULT_MAX_MOMENT_FLOPS = 200_000_000_000.0

# Target bytes for one (chunk_draws, p, p) matrix in float64.
_MOMENT_CHUNK_BYTES = 128 * 2 ** 20

_F64 = np.float64


def _check_entries(p, max_entries):
    """Guard the allocation of a p x p float64 matrix."""
    if max_entries is not None and p * p > int(max_entries):
        raise ValueError(
            f"the p x p matrix has {p * p} entries (p = {p}), above the "
            f"configured max_entries = {int(max_entries)}; pass a larger "
            "max_entries to the _posterior function if the memory is "
            "available, or use montecarlo_cov_sigma_pairs for selected "
            "entry pairs"
        )


def _canonical_pair(pair, p):
    """Validate one (i, j) index pair and return (min, max)."""
    try:
        a, b = pair
    except (TypeError, ValueError):
        raise ValueError(f"each entry must be an (i, j) pair, got {pair!r}")
    try:
        i = operator.index(a)
        j = operator.index(b)
    except TypeError:
        raise ValueError(f"pair indices must be integers, got {pair!r}")
    if not (0 <= i < p and 0 <= j < p):
        raise ValueError(f"pair index out of range for p = {p}: {(i, j)}")
    return (i, j) if i <= j else (j, i)


def _prepare_draws(B, sigma2):
    """Validate the Gibbs draws and return them as arrays."""
    B = np.asarray(B)
    sigma2 = np.asarray(sigma2)
    if B.ndim != 3 or min(B.shape) < 1:
        raise ValueError(
            f"B draws must have shape (S, p, H) with S, p, H >= 1, "
            f"got {B.shape}"
        )
    S, p, H = B.shape
    if sigma2.shape != (S, p):
        raise ValueError(
            f"sigma2 draws must have shape ({S}, {p}), got {sigma2.shape}"
        )
    if not np.isfinite(B).all():
        raise ValueError("B draws must contain only finite values")
    if not np.isfinite(sigma2).all():
        raise ValueError("sigma2 draws must contain only finite values")
    if np.any(sigma2 <= 0):
        raise ValueError("sigma2 draws must be strictly positive")
    return B, sigma2


def _subsample_indices(S, max_draws):
    """Deterministic equidistant draw indices (all draws when max_draws >= S).

    ``max_draws`` is validated at the entry (None, or an integer >= 1),
    so the same illegal value raises identically whatever its relation
    to ``S``; only then does the >= S shortcut return all draws.
    """
    if max_draws is not None:
        if (isinstance(max_draws, (bool, np.bool_))
                or not isinstance(max_draws, (int, np.integer))):
            raise ValueError("max_draws must be an integer or None")
        max_draws = int(max_draws)
        if max_draws < 1:
            raise ValueError(f"max_draws must be >= 1, got {max_draws}")
    if max_draws is None or max_draws >= S:
        return np.arange(S, dtype=np.int64)
    idx = np.rint(np.linspace(0, S - 1, max_draws)).astype(np.int64)
    return np.unique(idx)


def _auto_chunk_draws(p, chunk_bytes=_MOMENT_CHUNK_BYTES):
    """Draw-block size keeping one (chunk, p, p) float64 matrix under budget."""
    return max(1, min(256, int(chunk_bytes / (8 * max(p * p, 1)))))


def _resolve_chunk(chunk_draws, p, S):
    if chunk_draws is None:
        chunk = _auto_chunk_draws(p)
    else:
        if (isinstance(chunk_draws, (bool, np.bool_))
                or not isinstance(chunk_draws, (int, np.integer))
                or int(chunk_draws) < 1):
            raise ValueError("chunk_draws must be a positive integer or None")
        chunk = int(chunk_draws)
    return max(1, min(chunk, S))


def suggest_max_draws(
    p,
    H,
    S,
    *,
    max_flops=DEFAULT_MAX_MOMENT_FLOPS,
    max_draws=DEFAULT_MAX_DRAWS,
):
    """Suggest how many draws to use for the full p x p variance matrix.

    The cost model is ``2 p^2 H`` multiply-adds per draw (forming
    ``B_s B_s^T``) plus ``O(p^2)`` accumulation.  All draws are suggested
    when the full scan fits ``max_flops``; otherwise the suggestion is
    ``min(max_draws, floor(max_flops / (2 p^2 H)))``, floored at two
    draws: ``montecarlo_sigma_variance`` rejects fewer (a sample variance
    is undefined) and one extra draw is cheap.  With ``S == 1`` the floor
    is one and the variance function raises.  ``max_draws=None`` only
    removes the hard cap.

    Parameters
    ----------
    p, H, S : int
        Dimension, truncation level and number of available draws.
    max_flops : float, default DEFAULT_MAX_MOMENT_FLOPS
        Approximate compute budget.
    max_draws : int or None, default DEFAULT_MAX_DRAWS
        Hard cap on the number of draws.

    Returns
    -------
    int
        Number of draws to use (``S`` when the full scan fits the budget).
    """
    for name, value in (("p", p), ("H", H), ("S", S)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 1):
            raise ValueError(f"{name} must be an integer >= 1, got {value}")
    p, H, S = int(p), int(H), int(S)
    if max_draws is None:
        cap = S
    else:
        if (isinstance(max_draws, (bool, np.bool_))
                or not isinstance(max_draws, (int, np.integer))
                or int(max_draws) < 1):
            raise ValueError("max_draws must be a positive integer or None")
        cap = min(int(max_draws), S)
    if not np.isfinite(max_flops) or max_flops <= 0:
        raise ValueError(f"max_flops must be positive and finite, got {max_flops}")
    flops_per_draw = 2.0 * p * p * H
    if flops_per_draw * S <= max_flops:
        return S
    # Floor the budget suggestion at two draws: a sample variance is
    # undefined below two and montecarlo_sigma_variance rejects it.
    budget_draws = max(2, int(max_flops // flops_per_draw))
    return int(max(min(cap, budget_draws), min(2, S)))


def montecarlo_sigma_mean(B, sigma2, *, chunk_draws=None):
    """Monte Carlo mean of Sigma over Gibbs draws.

    ``(1/S) sum_s (B_s B_s^T + diag(sigma_s^2))`` computed by streaming over
    draw blocks, so the full set of per-draw p x p matrices is never stored.

    Parameters
    ----------
    B : array-like, shape (S, p, H)
        Loading draws (``GibbsCSP.B_samples_``).
    sigma2 : array-like, shape (S, p)
        Noise-variance draws (``GibbsCSP.sigma2_samples_``).
    chunk_draws : int or None, default None
        Draws per block; ``None`` selects a block size from a memory budget.

    Returns
    -------
    numpy.ndarray, shape (p, p), float64
    """
    B, sigma2 = _prepare_draws(B, sigma2)
    S, p, H = B.shape
    chunk = _resolve_chunk(chunk_draws, p, S)
    total = np.zeros((p, p), dtype=_F64)
    for start in range(0, S, chunk):
        stop = min(start + chunk, S)
        Bc = B[start:stop]
        G = np.matmul(Bc, np.swapaxes(Bc, 1, 2))
        total += G.sum(axis=0, dtype=_F64)
    mean = total / S
    mean[np.diag_indices(p)] += sigma2.mean(axis=0, dtype=_F64)
    return mean


def montecarlo_sigma_variance(
    B,
    sigma2,
    *,
    max_draws=None,
    chunk_draws=None,
    max_entries=DEFAULT_MAX_ENTRIES,
):
    """Monte Carlo entrywise posterior variance of Sigma from Gibbs draws.

    Computes ``(1/S) sum_s Sigma_s^2 - ((1/S) sum_s Sigma_s)^2`` with
    ddof = 0, where ``Sigma_s = B_s B_s^T + diag(sigma_s^2)``, streaming over
    draw blocks.  Only the entrywise variance matrix is returned; the full
    p^2 x p^2 covariance of ``vec(Sigma)`` is never formed.

    Parameters
    ----------
    B, sigma2
        Gibbs draws, as in :func:`montecarlo_sigma_mean`.
    max_draws : int or None, default None
        Number of draws to use; ``None`` uses all draws.  A value below S
        selects a deterministic equidistant subsample (see
        :func:`_subsample_indices`), which reduces the Monte Carlo
        precision.  Fewer than two retained draws (including S == 1 or an
        explicit ``max_draws=1``) raise ValueError: a sample variance is
        undefined and a single draw would silently produce zeros.
    chunk_draws : int or None, default None
        Draws per block; ``None`` selects a block size from a memory budget.
    max_entries : int or None, default DEFAULT_MAX_ENTRIES
        Safety cap on p * p; ``None`` disables the cap.

    Returns
    -------
    var : numpy.ndarray, shape (p, p), float64
        Entrywise sample variance, symmetric and nonnegative.
    n_used : int
        Number of draws actually used.
    """
    B, sigma2 = _prepare_draws(B, sigma2)
    S, p, H = B.shape
    _check_entries(p, max_entries)

    indices = _subsample_indices(S, max_draws)
    n_used = int(indices.size)
    if n_used < 2:
        # With one draw E[X^2] - E[X]^2 is identically zero: returning a
        # silent all-zero "variance" would be indistinguishable from a
        # true result, so fail loudly instead.
        raise ValueError(
            f"a sample variance needs at least 2 draws, got n_used = "
            f"{n_used} (S = {S}, max_draws = {max_draws}); use a larger "
            "max_draws, or montecarlo_sigma_diag_variance / "
            "montecarlo_cov_sigma_pairs for the cheap summaries"
        )
    chunk = _resolve_chunk(chunk_draws, p, n_used)

    total = np.zeros((p, p), dtype=_F64)
    total_sq = np.zeros((p, p), dtype=_F64)
    idx = np.arange(p)
    for start in range(0, n_used, chunk):
        sel = indices[start:start + chunk]
        Bc = B[sel]
        G = np.matmul(Bc, np.swapaxes(Bc, 1, 2))
        G[:, idx, idx] += sigma2[sel]
        total += G.sum(axis=0, dtype=_F64)
        G64 = G.astype(_F64, copy=False)
        np.square(G64, out=G64)
        total_sq += G64.sum(axis=0, dtype=_F64)

    mean = total / n_used
    var = total_sq / n_used - mean * mean
    np.maximum(var, 0.0, out=var)
    return 0.5 * (var + var.T), n_used


def montecarlo_sigma_diag_variance(B, sigma2):
    """Monte Carlo posterior variance of the diagonal entries of Sigma.

    Cheap: O(S p H) instead of O(S p^2 H).  Uses all draws.

    Returns
    -------
    numpy.ndarray, shape (p,), float64
        ``Var(Sigma_jj)`` sample variance with ddof = 0.
    """
    B, sigma2 = _prepare_draws(B, sigma2)
    S, p, H = B.shape
    chunk = max(1, min(S, int(_MOMENT_CHUNK_BYTES / (4 * max(p, 1)))))

    total = np.zeros(p, dtype=_F64)
    total_sq = np.zeros(p, dtype=_F64)
    for start in range(0, S, chunk):
        stop = min(start + chunk, S)
        Bc = B[start:stop]
        d = np.einsum("spk,spk->sp", Bc, Bc) + sigma2[start:stop]
        d64 = d.astype(_F64, copy=False)
        total += d64.sum(axis=0)
        np.square(d64, out=d64)
        total_sq += d64.sum(axis=0)

    mean = total / S
    var = total_sq / S - mean * mean
    return np.maximum(var, 0.0)


def montecarlo_cov_sigma_pairs(B, sigma2, pairs):
    """Monte Carlo posterior covariance between selected entries of Sigma.

    Uses all draws; cost O(S H) per pair, independent of p^2.

    Parameters
    ----------
    B, sigma2
        Gibbs draws, as in :func:`montecarlo_sigma_mean`.
    pairs : iterable of ((i, j), (k, l))
        Zero-based entry indices; order within and between entries is
        irrelevant.

    Returns
    -------
    numpy.ndarray, shape (len(pairs),), float64
        Sample covariances with ddof = 0.
    """
    B, sigma2 = _prepare_draws(B, sigma2)
    S, p, H = B.shape

    canonical = []
    for entry in pairs:
        if len(entry) != 2:
            raise ValueError(
                f"each entry must be ((i, j), (k, l)), got {entry!r}"
            )
        canonical.append(
            _canonical_pair(entry[0], p) + _canonical_pair(entry[1], p)
        )

    out = np.empty(len(canonical), dtype=_F64)
    for idx, (i, j, k, l) in enumerate(canonical):
        X = np.einsum("sk,sk->s", B[:, i, :], B[:, j, :]).astype(_F64)
        Y = np.einsum("sk,sk->s", B[:, k, :], B[:, l, :]).astype(_F64)
        if i == j:
            X += sigma2[:, i]
        if k == l:
            Y += sigma2[:, k]
        out[idx] = (X * Y).mean() - X.mean() * Y.mean()
    return out

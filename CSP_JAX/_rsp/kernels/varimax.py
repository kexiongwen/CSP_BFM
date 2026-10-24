"""Batched varimax using the matrix-form iteration with `normalize = FALSE`,
default `eps = 1e-5`, and at most 1000 iterations by default. Returns rotated
loadings and rotation matrices for composition.

The default `polar="qdwh"` obtains the polar factor W of B through QR-based
dynamically weighted Halley iteration. Its stopping quantity is
d = <W, B>_F = tr(WᵀB), equal to the sum of singular values for a polar factor.
The `polar="svd"` path obtains this factor directly from an SVD.

QDWH results must have converged inner solves and finite orthogonality error
||WᵀW - I||_F <= orth_tol, computed at highest matmul precision. Failed draws
are recomputed with SVD outside the jitted batch; accepted draws retain their
QDWH results.

`return_n_iter=True` also returns per-draw iteration counts. The batched while
loop runs until its slowest active draw stops, while completed draws retain
their states; the counts describe individual draws, not batch execution cost.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .._numeric import power_of_two


def _staged_scale(x: jax.Array):
    """Return loadings scaled in two power-of-two stages for each draw.

    Uniform positive scaling preserves the varimax maximizers in exact
    arithmetic, while the intermediate B grows with the fourth power of
    input magnitude. Splitting the exponent keeps each scale factor within
    the dtype range without forming a potentially underflowing reciprocal.
    The factors are constructed from exponent bits (power_of_two), because
    device exp2 is inexact beyond its small exactly-representable window.
    Floating-point rounding can change the iteration path, particularly for
    degenerate or non-unique optima.
    """
    maxabs = jnp.max(jnp.abs(x))
    e = jnp.floor(jnp.log2(maxabs))
    ok = jnp.isfinite(e) & (maxabs > 0)
    e1 = jnp.where(ok, jnp.floor(e / 2), 0.0)
    e2 = jnp.where(ok, e - e1, 0.0)
    s1 = power_of_two(e1.astype(jnp.int32), x.dtype)
    s2 = power_of_two(e2.astype(jnp.int32), x.dtype)
    return (x / s1) / s2


def _varimax_body(xs, p, tt):
    """Return the matrix B whose polar factor defines the next rotation."""
    z = xs @ tt
    colsums_sq = jnp.sum(z**2, axis=0)  # colSums(z^2)
    # B = crossprod(x, z^3 - z %*% diag(colsums_sq) / p)
    b = xs.T @ (z**3 - z * (colsums_sq / p)[None, :])
    return b


def _varimax_single(x: jax.Array, eps: float, max_iter: int):
    """x: (p, q) with q >= 2. Returns (rotated loadings, rotmat TT, n_iter).

    A zero draw keeps the identity rotation without entering the loop:
    its d sequence stays zero, so the strict stopping rule would not fire.
    """
    p, q = x.shape
    xs = _staged_scale(x)
    active = jnp.max(jnp.abs(x)) > 0

    def cond(state):
        _tt, d, dpast, i = state
        # Stop when d < dpast * (1 + eps); never true on the first sweep
        # (dpast = 0 and d >= 0 -> 0 < 0 is FALSE).
        return active & (d >= dpast * (1.0 + eps)) & (i < max_iter)

    def body(state):
        tt, _d, _dpast, i = state
        b = _varimax_body(xs, p, tt)
        u, s, vh = jnp.linalg.svd(b, full_matrices=False)
        tt_new = u @ vh
        return tt_new, jnp.sum(s), _d, i + 1

    init = (jnp.eye(q, dtype=x.dtype), jnp.zeros((), x.dtype), jnp.zeros((), x.dtype), 0)
    tt, _d, _dpast, i = jax.lax.while_loop(cond, body, init)
    return x @ tt, tt, i


def _varimax_single_qdwh(x: jax.Array, eps: float, max_iter: int):
    """QDWH variant with the same stopping rule and scale normalization.
    x: (p, q) with q >= 2.

    A zero or non-finite computed norm of B keeps the current rotation.
    A zero B has no unique polar factor, and QDWH's zero result would not be
    an orthogonal rotation. No absolute small-norm cutoff is used: a small
    full-rank matrix can still require a rotation update.

    Per-draw QDWH convergence flags accumulate into `ok_conv`; a draw with any
    non-converged inner solve is marked for the host-side fallback."""
    from jax.lax.linalg import qdwh

    p, q = x.shape
    xs = _staged_scale(x)
    active = jnp.max(jnp.abs(x)) > 0  # zero draw: keep I without iterating

    def cond(state):
        _tt, d, dpast, i, _okc = state
        return active & (d >= dpast * (1.0 + eps)) & (i < max_iter)

    def body(state):
        tt, _d, _dpast, i, okc = state
        b = _varimax_body(xs, p, tt)
        u, _h, _n, conv = qdwh(b)
        b_norm = jnp.linalg.norm(b)
        good = jnp.isfinite(b_norm) & (b_norm > 0.0)
        tt_new = jnp.where(good, u, tt)
        d_new = jnp.sum(tt_new * b)  # d = <W, B>_F = tr(WᵀB) on the SCALED B
        okc = okc & (conv | (~good))  # skipped (degenerate) updates don't count
        return tt_new, d_new, _d, i + 1, okc

    init = (
        jnp.eye(q, dtype=x.dtype),
        jnp.zeros((), x.dtype),
        jnp.zeros((), x.dtype),
        0,
        jnp.asarray(True),
    )
    tt, _d, _dpast, i, okc = jax.lax.while_loop(cond, body, init)
    return x @ tt, tt, i, okc


def _default_orth_tol(dtype):
    return 1e-3 if dtype == jnp.float32 else 1e-9


def _varimax_impl(loadings, eps, max_iter, return_n_iter):
    t, p, q = loadings.shape
    if q < 2:
        eye = jnp.broadcast_to(jnp.eye(q, dtype=loadings.dtype), (t, q, q))
        if return_n_iter:
            return loadings, eye, jnp.zeros((t,), dtype=jnp.int32)
        return loadings, eye
    rotated, rotmats, n_iters = jax.vmap(lambda x: _varimax_single(x, eps, max_iter))(
        loadings
    )
    if return_n_iter:
        return rotated, rotmats, n_iters.astype(jnp.int32)
    return rotated, rotmats


def _varimax_qdwh_impl(loadings, eps, max_iter, orth_tol):
    """Batched QDWH varimax core, returning per-draw acceptance flags.

    Returns (rotated, rotmats, n_iters, bad): bad[t] iff the draw's rotation
    is non-finite, or non-orthogonal beyond orth_tol (checked at HIGHEST
    matmul precision), or any of its inner QDWH solves did not converge.
    Inner convergence alone is insufficient: a zero B can yield W=0,
    which is not orthogonal.
    """
    t, p, q = loadings.shape
    if q < 2:  # same identity shortcut as the SVD path
        eye = jnp.broadcast_to(jnp.eye(q, dtype=loadings.dtype), (t, q, q))
        return loadings, eye, jnp.zeros((t,), dtype=jnp.int32), jnp.zeros((t,), bool)
    rotated, rotmats, n_iters, ok_conv = jax.vmap(
        lambda x: _varimax_single_qdwh(x, eps, max_iter)
    )(loadings)
    eye = jnp.eye(q, dtype=loadings.dtype)
    with jax.default_matmul_precision("highest"):
        orth_err = jnp.linalg.norm(
            jnp.einsum("tji,tjk->tik", rotmats, rotmats) - eye, axis=(1, 2)
        )
    bad = (~jnp.isfinite(orth_err)) | (orth_err > orth_tol) | (~ok_conv)
    return rotated, rotmats, n_iters.astype(jnp.int32), bad


_VARIMAX_JIT = None
_VARIMAX_QDWH_JIT = None


def _varimax_svd_cached(loadings, eps, max_iter, return_n_iter):
    global _VARIMAX_JIT
    if _VARIMAX_JIT is None:
        _VARIMAX_JIT = jax.jit(
            _varimax_impl, static_argnames=("eps", "max_iter", "return_n_iter")
        )
    return _VARIMAX_JIT(loadings, eps=eps, max_iter=max_iter, return_n_iter=return_n_iter)


def _varimax_qdwh_cached(loadings, eps, max_iter, orth_tol):
    global _VARIMAX_QDWH_JIT
    if _VARIMAX_QDWH_JIT is None:
        _VARIMAX_QDWH_JIT = jax.jit(
            _varimax_qdwh_impl, static_argnames=("eps", "max_iter", "orth_tol")
        )
    return _VARIMAX_QDWH_JIT(loadings, eps=eps, max_iter=max_iter, orth_tol=orth_tol)


def _varimax_qdwh_host(loadings, eps, max_iter, return_n_iter, orth_tol):
    """Run QDWH, then replace failed draws with SVD results.

    Fallback runs outside the jitted batch so its input size can depend on
    the failed-draw mask.
    """
    rotated, rotmats, n_iters, bad = _varimax_qdwh_cached(loadings, eps, max_iter, orth_tol)
    if bool(jnp.any(bad)):
        idx = np.nonzero(np.asarray(bad))[0]
        sub = jnp.asarray(loadings)[idx]  # host gather; dynamic size is fine here
        r_svd, t_svd, n_svd = _varimax_svd_cached(sub, eps, max_iter, True)
        rotated = rotated.at[idx].set(r_svd)
        rotmats = rotmats.at[idx].set(t_svd)
        n_iters = n_iters.at[idx].set(n_svd)
    if return_n_iter:
        return rotated, rotmats, n_iters
    return rotated, rotmats


def varimax_batch(loadings: jax.Array, eps: float = 1e-5, max_iter: int = 1000,
                  return_n_iter: bool = False, polar: str = "qdwh",
                  orth_tol: float | None = None):
    """(T, p, q) -> ((T, p, q) rotated, (T, q, q) rotation matrices); with
    return_n_iter=True also the (T,) int32 per-draw iteration counts.

    q == 1 returns the input and identity rotations (no rotation is needed).

    polar: "qdwh" (default) uses jax.lax.linalg.qdwh with a degenerate-matrix
    guard and host-side SVD fallback; orth_tol defaults to 1e-3 in float32
    and 1e-9 in float64. "svd" uses the SVD-based matrix iteration.

    Each polar path reuses a cached jitted wrapper, with iteration settings
    captured as static arguments. QDWH fallback selection runs on the host.
    """
    loadings = jnp.asarray(loadings)
    if loadings.ndim != 3:
        raise ValueError(f"expected (T, p, q), got {loadings.shape}")
    if polar not in ("svd", "qdwh"):
        raise ValueError(f"unknown polar {polar!r}")
    if polar == "svd":
        return _varimax_svd_cached(loadings, eps, max_iter, return_n_iter)
    if orth_tol is None:
        orth_tol = _default_orth_tol(loadings.dtype)
    return _varimax_qdwh_host(loadings, eps, max_iter, return_n_iter, orth_tol)

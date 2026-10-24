"""On-demand conditional score sampling for retained Gibbs parameter draws.

Ported from the ``bfm`` package of the L_{1/2} paper (LH_BFM,
``bfm/_factor_scores.py``); the only change is the import path
(``eta_sample`` lives directly in this package here).

Score storage here is COLUMN-oriented: outputs are (S, H, n), column i
of draw s is the factor vector eta_i, and RSP alignment applies R^T to
each column (``swapaxes(R, -1, -2) @ scores``).  The
:func:`CSP_JAX._rsp.transform_scores` helper performs the same transform on
ROW-stored scores ((T, n, q), ``scores @ R``); the two conventions are
equivalent under transposition -- use this module for the estimator's
column-stored draws.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .eta_sample import eta_sample


@jax.jit
def _draw_score_batch(master_key, indices, Xc, loadings, variances):
    # Original draw indices make keys independent of batch boundaries.
    # map keeps the conditional sampler's workspace to one draw at a time.
    def draw(args):
        index, B, sigma2 = args
        key = jax.random.fold_in(master_key, index)
        return eta_sample(key, Xc, B, sigma2)

    return jax.lax.map(draw, (indices, loadings, variances))


def sample_factor_scores(
    X, loadings, variances, offset, *, return_eta, return_eta_aligned,
    seed, batch_size, transforms,
):
    """Validate and sample without modifying estimator-owned arrays."""
    for name, flag in (("return_eta", return_eta),
                       ("return_eta_aligned", return_eta_aligned)):
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean")
    if (isinstance(batch_size, (bool, np.bool_)) or
            not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")
    if seed is not None and (
        isinstance(seed, (bool, np.bool_)) or
            not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32
    ):
        raise ValueError("seed must be an integer in [0, 2**32) or None")
    if not return_eta and not return_eta_aligned:
        return {}

    loadings = np.asarray(loadings)
    variances = np.asarray(variances)
    if loadings.ndim != 3 or min(loadings.shape) < 1:
        raise ValueError("B_samples_ must have nonempty shape (S, p, H)")
    S, p, H = loadings.shape
    if variances.shape != (S, p):
        raise ValueError("sigma2_samples_ must have shape (S, p)")
    if (not np.isfinite(loadings).all() or not np.isfinite(variances).all()
            or np.any(variances <= 0)):
        raise ValueError("kept loadings must be finite and noise variances finite and positive")

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] != p:
        raise ValueError(f"X must have shape (n, {p}) with n >= 1, got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    offset = np.asarray(offset, dtype=np.float64)
    if offset.shape != (p,) or not np.isfinite(offset).all():
        raise ValueError("x_offset_ must be finite and have shape (p,)")
    # Center in float64 to avoid cancellation, then store the float32 result.
    with np.errstate(over="ignore", invalid="ignore"):
        Xc = (X - offset).astype(np.float32)
    if not np.isfinite(Xc).all():
        raise ValueError("centering X with x_offset_ produced non-finite values")

    aligned_dtype = None
    if return_eta_aligned:
        if transforms is None:
            raise RuntimeError("call align_factors() before requesting eta_sample_aligned")
        if transforms.shape != (S, H, H):
            raise ValueError("alignment transforms must have shape (S, H, H)")
        aligned_dtype = np.result_type(np.float32, transforms.dtype)

    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**32, dtype=np.uint32))
    # Give factor-score sampling a separate purpose tag from the fit stream.
    master_key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), 0x455441)
    Xc = jnp.asarray(Xc)
    output = {}
    shape = (S, H, X.shape[0])
    if return_eta:
        output["eta_sample"] = np.empty(shape, dtype=np.float32)
    if return_eta_aligned:
        output["eta_sample_aligned"] = np.empty(shape, dtype=aligned_dtype)

    for start in range(0, S, int(batch_size)):
        stop = min(start + int(batch_size), S)
        scores = np.asarray(_draw_score_batch(
            master_key, jnp.arange(start, stop, dtype=jnp.uint32), Xc,
            jnp.asarray(loadings[start:stop]), jnp.asarray(variances[start:stop]),
        ))
        if not np.isfinite(scores).all():
            raise FloatingPointError("conditional factor sampling produced non-finite scores")
        if return_eta:
            output["eta_sample"][start:stop] = scores
        if return_eta_aligned:
            rotations = np.asarray(transforms[start:stop])
            with np.errstate(over="ignore", invalid="ignore"):
                aligned = np.swapaxes(rotations, -1, -2) @ scores
            if not np.isfinite(aligned).all():
                raise FloatingPointError("factor alignment produced non-finite scores")
            output["eta_sample_aligned"][start:stop] = aligned
    return output

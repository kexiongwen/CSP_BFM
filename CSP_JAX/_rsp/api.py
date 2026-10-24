"""Public RSP API: validation, configuration, and alignment.

Alignment uses varimax rotation followed by exact sign/permutation steps.
Default precision is float32, with explicit input casting and no global x64
configuration change. precision="float64" enables x64 at the entry point,
never at import time.
"""
from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np

from ._numeric import safe_mean, scaled_norm, sum_squares
from .core.outer import make_step_fn
from .io import _resolve_pq, from_flat
from .kernels.assignment import DEFAULT_Q_ENUM
from .kernels.varimax import varimax_batch
from .types import RSPResult

logger = logging.getLogger("fs")

# Cache step wrappers by configuration so repeated calls reuse the same JIT
# function identity and its compilation cache.
_STEP_FN_CACHE: dict = {}


def _cached_step_fn(sp_path: str, backend: str, q_enum: int = DEFAULT_Q_ENUM,
                    max_chunk_elems: int = 1 << 26):
    key = (sp_path, backend, q_enum, max_chunk_elems)
    fn = _STEP_FN_CACHE.get(key)
    if fn is None:
        fn = make_step_fn(sp_path, backend, q_enum, max_chunk_elems)
        _STEP_FN_CACHE[key] = fn
    return fn


STATUS_CONVERGED = 1
STATUS_MAX_ITER = 2
STATUS_ANOMALY = 3  # exact objective increased beyond the numerical tolerance
STATUS_NON_FINITE = 4  # non-finite objective, intermediate, or returned state

#: relative canary tolerance for the (theoretically monotone) exact objective
CANARY_RTOL = 1e-9
#: relative threshold for the near-zero initial reference diagnostic
NEAR_ZERO_RTOL = 1e-8
#: multiplier on eps^2 * data_sq_scale for the absolute canary noise floor
CANARY_FLOOR_K = 100.0


def _canary_rtol(dtype, n_terms):
    """Relative tolerance for objective increases: max(1e-9, 10*eps*sqrt(N)).

    Scaling with the working dtype and reduction size allows ordinary
    floating-point jitter while detecting larger monotonicity violations.
    """
    return float(max(CANARY_RTOL, 10.0 * np.finfo(dtype).eps * np.sqrt(n_terms)))


def _canary_floor(dtype, data_sq_scale):
    """Absolute tolerance K*eps^2*data_sq_scale for objective increases.

    The input-scale bound allows for rotation and residual rounding when the
    true alignment loss is near zero and a relative loss tolerance is small.
    """
    return CANARY_FLOOR_K * float(np.finfo(dtype).eps) ** 2 * data_sq_scale


def _data_sq_scale(raw_j):
    """Host-side upper bound ||x||_F^2 <= (max|x|)^2 * nelems.

    Only the maximum magnitude is synchronized from the device; no device
    sum of squares is needed. Overestimating loosens the canary noise floor.
    """
    smax = float(jnp.max(jnp.abs(raw_j)))
    return smax * smax * float(raw_j.size)


def _configure_precision(precision="float32"):
    """float32 (default): explicit input cast; the global x64 config is NOT
    touched and the kernels preserve the input dtype. float64 (opt-in fidelity
    mode): enable x64 at the entry point and verify."""
    if precision == "float32":
        pass  # inputs are cast explicitly; kernels preserve the input dtype
    elif precision == "float64":
        jax.config.update("jax_enable_x64", True)
        probe = jnp.asarray(1.0)
        if probe.dtype != jnp.float64:
            raise RuntimeError("jax_enable_x64 did not yield float64; check the JAX install")
    else:
        raise ValueError(f"precision must be 'float32' or 'float64', got {precision!r}")


def _as_3d(lambda_mcmc, colnames=None, p=None, q=None, dtype=np.float32):
    cols = colnames
    if cols is None and hasattr(lambda_mcmc, "columns"):  # e.g. pandas DataFrame
        cols = list(lambda_mcmc.columns)
    if isinstance(lambda_mcmc, jax.Array):
        # Keep device inputs on device; the finiteness check synchronizes
        # only one boolean to the host.
        arr = jnp.asarray(lambda_mcmc, dtype=dtype)
        if arr.ndim == 3:
            if cols is not None or p is not None or q is not None:
                raise ValueError("3-D input takes no colnames/p/q")
        elif arr.ndim == 2:
            p, q = _resolve_pq(cols, p, q, arr.shape[1])
            arr = arr.reshape(arr.shape[0], p, q)
        else:
            raise ValueError(
                f"loadings must be (T, p, q) or flat (T, p*q), got shape {arr.shape}"
            )
        if not bool(jnp.all(jnp.isfinite(arr))):
            raise ValueError("loadings contain non-finite values")
    else:
        arr = np.asarray(lambda_mcmc, dtype=dtype)
        if arr.ndim == 3:
            if cols is not None or p is not None or q is not None:
                raise ValueError("3-D input takes no colnames/p/q")
        elif arr.ndim == 2:
            arr = from_flat(arr, colnames=cols, p=p, q=q, dtype=dtype)
        else:
            raise ValueError(
                f"loadings must be (T, p, q) or flat (T, p*q), got shape {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("loadings contain non-finite values")
    t_, p_, q_ = arr.shape
    if t_ < 1 or p_ < 1 or q_ < 1:
        raise ValueError(f"empty input (T, p, q) = {(t_, p_, q_)} is not supported")
    return arr


def _compose_transforms(rotmats, signs, perms, dtype):
    """Full per-draw orthogonal transform O(t) = R(t) @ Q_sp(t) via the
    signed-permutation structure: a column gather + sign scale, O(T q^2) and
    fully device-resident, instead of a dense matmul O(T q^3) plus a host
    round-trip for signs/perms. With rotmats=None the transform is Q_sp alone."""
    signs_j = jnp.asarray(signs).astype(dtype)
    perms_j = jnp.asarray(perms, dtype=jnp.int32)
    t, q = perms_j.shape
    if rotmats is None:
        base = jnp.broadcast_to(jnp.eye(q, dtype=dtype), (t, q, q))
    else:
        base = jnp.asarray(rotmats, dtype=dtype)
    gathered = jnp.take_along_axis(base, perms_j[:, None, :], axis=2)
    return gathered * signs_j[:, None, :]


def _representative_init(rotated, sp_path, backend, q_enum: int = DEFAULT_Q_ENUM,
                         max_chunk_elems: int = 1 << 26):
    """Representative-draw initialization (init='draw'/'auto').

    Align all draws to the draw with largest Frobenius norm, then use their
    aligned mean as the initial reference. A cached outer step performs this
    alignment without constructing a new compiled wrapper on each call.

    The zero check is elementwise; scaled norms avoid directly squaring tiny
    or large draws. Returns None for all-zero input, where the caller uses
    the zero mean.
    """
    norms = scaled_norm(rotated, axis=(1, 2))
    idx = jnp.argmax(norms)
    if not bool(jnp.any(rotated != 0)):
        return None
    rep = rotated[idx]
    _signs, _perms, _aligned, m_init, _f_pre, _f_post = _cached_step_fn(
        sp_path, backend, q_enum, max_chunk_elems
    )(rotated, rep)
    return m_init


def _prepare_input(lambda_mcmc, colnames, p, q, precision, max_iter, threshold):
    """Entry preamble: precision mode, max_iter/threshold validation,
    input conversion with device pass-through. Returns the (T, p, q) jax array
    in the working dtype."""
    _configure_precision(precision)
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("threshold must be positive and finite")
    raw = _as_3d(lambda_mcmc, colnames=colnames, p=p, q=q,
                 dtype=np.float32 if precision == "float32" else np.float64)
    return jnp.asarray(raw)


def _history_fields(hist_pre, hist_post, max_iter, dtype):
    """Build fixed-capacity loss histories and loss_final in the working dtype."""
    n_iter = len(hist_pre)
    loss_history = jnp.zeros((max_iter,), dtype=dtype).at[:n_iter].set(jnp.asarray(hist_pre, dtype=dtype))
    loss_history_post = jnp.zeros((max_iter,), dtype=dtype).at[:n_iter].set(jnp.asarray(hist_post, dtype=dtype))
    loss_final = jnp.asarray(hist_post[-1], dtype=dtype)
    return n_iter, loss_history, loss_history_post, loss_final


def _nonfinite_fields(**values):
    """Check the current outputs together, using one host synchronization.

    A finite pre-update loss does not establish that a new mean or
    post-update loss is finite. Initial loss is deliberately excluded:
    it may be out of range before a successful alignment removes a residual.
    """
    present = {name: value for name, value in values.items() if value is not None}
    if not present:
        return ()
    finite = np.asarray(jnp.stack([jnp.all(jnp.isfinite(v)) for v in present.values()]))
    return tuple(name for name, ok in zip(present, finite) if not ok)


def rsp_exact(
    lambda_mcmc,
    *,
    colnames=None,
    p=None,
    q=None,
    max_iter: int = 100,
    threshold: float = 1e-6,
    rotate: bool = True,
    sp_path: str = "elim",
    assignment_backend: str = "auto",
    assignment_q_enum: int = DEFAULT_Q_ENUM,
    sp_chunk_elems: int | None = None,
    polar: str = "qdwh",
    reference=None,
    init: str = "auto",
    return_transforms: bool = False,
    verbose: bool = False,
    precision: str = "float32",
) -> RSPResult:
    """Varimax rotation followed by exact sign/permutation alignment.

    lambda_mcmc: (T, p, q) array, or (T, p*q) flat with strict LambdaV
    `colnames`, or flat with explicit p and q.

    init: "auto" (default; representative-draw initialization, logging a
    near-zero mean when verbose), "draw" (the same rule without that log),
    or "mean" (mean of the rotated draws). The near-zero mean
    diagnostic is recorded for all modes. An explicit `reference` takes
    priority over `init` (init applies only when reference=None).

    precision: "float32" (default; explicit cast, x64 global untouched) or
    "float64" (opt-in fidelity mode, enables x64 at the entry).
    """
    if sp_path not in ("elim", "enum"):
        raise ValueError(f"unknown sp_path {sp_path!r}")
    if init not in ("auto", "mean", "draw"):
        raise ValueError(f"unknown init {init!r}")
    if not isinstance(assignment_q_enum, (int, np.integer)) or assignment_q_enum < 0:
        raise ValueError("assignment_q_enum must be a non-negative integer")
    if sp_chunk_elems is not None and (
        not isinstance(sp_chunk_elems, (int, np.integer)) or sp_chunk_elems < 1
    ):
        raise ValueError("sp_chunk_elems must be a positive integer or None")
    if polar not in ("svd", "qdwh"):
        raise ValueError(f"unknown polar {polar!r}")
    chunk_elems = (1 << 26) if sp_chunk_elems is None else int(sp_chunk_elems)

    raw_j = _prepare_input(lambda_mcmc, colnames, p, q, precision, max_iter, threshold)
    t, p_, q_ = raw_j.shape

    if rotate and q_ > 1:
        rotated, rotmats, vm_n_iter = varimax_batch(raw_j, polar=polar, return_n_iter=True)
    else:
        rotated = raw_j
        rotmats = None
        vm_n_iter = None

    mean_rot = safe_mean(rotated, axis=0)
    near_zero = bool(
        scaled_norm(mean_rot)
        <= NEAR_ZERO_RTOL * float(safe_mean(scaled_norm(rotated, axis=(1, 2))))
    )
    if reference is not None:
        m0 = jnp.asarray(reference, dtype=raw_j.dtype)
    elif init == "mean":
        m0 = mean_rot
    else:
        m0 = _representative_init(rotated, sp_path, assignment_backend, assignment_q_enum,
                                  chunk_elems)
        if m0 is None:  # all-zero input: use its zero mean
            m0 = mean_rot
        elif init == "auto" and verbose and near_zero:
            logger.info(
                "mean initialization cancels to near zero; using representative-draw initialization"
            )
    m0 = jnp.asarray(m0, dtype=raw_j.dtype)
    if m0.shape != (p_, q_):
        raise ValueError(f"reference must be ({p_}, {q_}), got {m0.shape}")
    if not bool(jnp.all(jnp.isfinite(m0))):
        raise ValueError("reference contains non-finite values")
    d0 = rotated - m0[None]
    f0 = sum_squares(d0)  # initial loss against M_0

    thr = threshold * t * p_ * q_
    canary = _canary_rtol(raw_j.dtype, t * p_ * q_)
    canary_floor = _canary_floor(raw_j.dtype, _data_sq_scale(raw_j))
    step = _cached_step_fn(sp_path, assignment_backend, assignment_q_enum, chunk_elems)

    m = m0
    f_prev = None
    status = STATUS_MAX_ITER
    stop_reason = "max_iter"
    nonfinite_fields = ()
    hist_pre = []
    hist_post = []
    signs = perms = aligned = None
    for it in range(1, max_iter + 1):
        signs, perms, aligned, m_new, f_pre, f_post = step(rotated, m)
        f_pre_f = float(f_pre)
        hist_pre.append(f_pre_f)
        hist_post.append(float(f_post))
        if verbose:
            logger.info("iteration %d: objective = %.6f", it, f_pre_f)
        nonfinite_fields = _nonfinite_fields(
            loss_pre=f_pre, loss_post=f_post, aligned_loadings=aligned,
            mean_loadings=m_new,
        )
        if nonfinite_fields:
            status = STATUS_NON_FINITE
            stop_reason = "nonfinite_objective"
            m = m_new  # the stopping sweep still updates the reference
            if verbose:
                logger.warning("objective is non-finite (%.6g); stopping", f_pre_f)
            break
        anomaly = f_prev is not None and f_pre_f - f_prev > max(
            canary * abs(f_prev), canary_floor
        )
        m = m_new  # the stopping sweep still updates the reference
        if anomaly:
            status = STATUS_ANOMALY
            stop_reason = "anomaly"
            if verbose:  # the status code records the anomaly even without logging
                logger.warning("exact objective increased (%.6f -> %.6f); stopping", f_prev, f_pre_f)
            break
        if f_prev is not None and f_prev - f_pre_f < thr:
            status = STATUS_CONVERGED
            stop_reason = "converged_threshold"
            break
        f_prev = f_pre_f

    n_iter, loss_history, loss_history_post, loss_final = _history_fields(
        hist_pre, hist_post, max_iter, raw_j.dtype
    )

    transforms = None
    if return_transforms:
        transforms = _compose_transforms(rotmats, signs, perms, raw_j.dtype)
    final_nonfinite = _nonfinite_fields(
        aligned_loadings=aligned, mean_loadings=m, loss_final=loss_final,
        transforms=transforms,
    )
    if final_nonfinite:
        nonfinite_fields = tuple(dict.fromkeys(nonfinite_fields + final_nonfinite))
        status, stop_reason = STATUS_NON_FINITE, "nonfinite_objective"

    return RSPResult(
        aligned_loadings=aligned,
        mean_loadings=m,
        signs=signs.astype(jnp.int8),
        permutations=perms.astype(jnp.int32),
        loss_initial=f0,
        loss_final=loss_final,
        loss_history=loss_history,
        n_iter=jnp.asarray(n_iter),
        converged=jnp.asarray(status == STATUS_CONVERGED),
        status_code=jnp.asarray(status),
        transforms=transforms,
        loss_history_post=loss_history_post,
        varimax_n_iter=vm_n_iter,
        diagnostics={
            "initial_reference_near_zero": near_zero,
            "sp_path": sp_path,
            "assignment_backend": assignment_backend,
            "rotated": bool(rotate and q_ > 1),
            "precision": precision,
            "stop_reason": stop_reason,
            "nonfinite_fields": nonfinite_fields,
        },
    )

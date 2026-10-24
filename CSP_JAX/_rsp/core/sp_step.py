"""SP (sign-permutation) step: per draw, fixed reference M, solve

    min over s in {-1,1}^q, nu in T_q of ||A P_nu D_s - M||_F^2

Two paths:

* "elim" (default): signs eliminated analytically per matched pair using the
  DIRECT squared residuals min(||a_k - m_j||^2, ||a_k + m_j||^2) (never
  -|G| with G = M' A -- inner-product rounding erases the residual-scale gap
  between assignments on large near-equal inputs; see sp_step_elim); one
  assignment per draw at ordinary ranges, with upper-bound refinement when
  a common scale erases positive costs. Sign ties select +1, never 0.
* "enum": enumerate all 2^q sign vectors in a fixed order (all +1; subsets of
  increasing size set to -1 in itertools.combinations order; all -1 last),
  solve one assignment per sign vector, take the first minimum.

Both return original-scale losses recomputed from the aligned residuals.
Pair costs retain a binary exponent separately from their floating-point
significand. This protects small columns next to large matching columns;
assignment still uses one common scale, never independently weighted rows
or columns. These are finite-precision computations, not an exact-arithmetic
guarantee for arbitrary dynamic range.
"""
from __future__ import annotations

import itertools
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from .._numeric import binary_exponent, scale_pow2, sum_squares
from ..kernels.assignment import DEFAULT_Q_ENUM, solve_assignment_batch

#: construction-time byte budget for the 2^q sign-enumeration table
_SIGN_TABLE_MAX_BYTES = 1 << 30


@lru_cache(maxsize=None)
def r_sign_enumeration(q: int) -> np.ndarray:
    """2^q sign vectors in the fixed enumeration order.

    Check the table's byte budget before allocation. Fill rows directly into
    a preallocated array to avoid a temporary list of all sign vectors.
    """
    nbytes = (1 << q) * q  # int8 entries
    if nbytes > _SIGN_TABLE_MAX_BYTES:
        raise ValueError(
            f"sign enumeration for q={q} needs a {nbytes / 2**30:.1f} GiB table; "
            "the enum path is validation-only, use path='elim'"
        )
    out = np.ones((1 << q, q), dtype=np.int8)
    idx = 1
    for size in range(1, q):
        for combo in itertools.combinations(range(q), size):
            out[idx, list(combo)] = -1
            idx += 1
    if q > 0:
        out[idx] = -1
    return out


def _less_cost(am, ae, bm, be):
    """Compare nonnegative costs represented by significand * 2**exponent."""
    return ((am == 0) & (bm != 0)) | (
        (am != 0) & (bm != 0) & ((ae < be) | ((ae == be) & (am < bm)))
    )


def _normalize_cost(value, exponent):
    e = binary_exponent(value)
    return scale_pow2(value, -e), jnp.where(value == 0, 0, exponent + e)


def _signed_pair_cost(a, m, sign):
    """Direct residual costs, with a separate exponent for each column pair.

    Subtract first so cancellation of a large identical component preserves
    a small remaining residual. Only overflowing subtractions use a common
    operand scale for that pair. Normalize the residual BEFORE squaring;
    normalizing the whole draw could erase small columns.
    """
    left = a[:, :, None, :]
    right = sign * m[None, :, :, None]
    residual = left - right
    finite = jnp.all(jnp.isfinite(residual), axis=1)

    def recover_overflow():
        largest = jnp.maximum(jnp.max(jnp.abs(left), axis=1),
                              jnp.max(jnp.abs(right), axis=1))
        e = binary_exponent(largest)
        scaled = scale_pow2(left, -e[:, None]) - scale_pow2(right, -e[:, None])
        return jnp.where(finite[:, None], residual, scaled), jnp.where(finite, 0, e)

    residual, base_e = jax.lax.cond(
        jnp.all(finite),
        lambda: (residual, jnp.zeros(finite.shape, dtype=jnp.int32)),
        recover_overflow,
    )
    e = binary_exponent(jnp.max(jnp.abs(residual), axis=1))
    normalized = scale_pow2(residual, -e[:, None])
    value = jnp.sum(normalized * normalized, axis=1)
    return _normalize_cost(value, 2 * (base_e + e))


def _pair_tables(a, m):
    positive_m, positive_e = _signed_pair_cost(a, m, 1)
    negative_m, negative_e = _signed_pair_cost(a, m, -1)
    negative = _less_cost(negative_m, negative_e, positive_m, positive_e)
    return positive_m, positive_e, negative_m, negative_e, negative


def _row_minimum(mantissa, exponent):
    key_e = jnp.where(mantissa == 0, jnp.iinfo(jnp.int32).min, exponent)
    minimum_e = jnp.min(key_e, axis=2, keepdims=True)
    minimum_m = jnp.min(jnp.where(key_e == minimum_e, mantissa, jnp.inf),
                        axis=2, keepdims=True)
    return minimum_m, jnp.where(minimum_m == 0, 0, minimum_e)


def _subtract_rows(mantissa, exponent, row_m, row_e):
    # The same row constant is used across every sign configuration in enum.
    # It therefore changes neither assignment nor sign-vector preferences.
    difference = jnp.maximum(mantissa - scale_pow2(row_m, row_e - exponent), 0)
    return _normalize_cost(difference, exponent)


def _scaled_costs(mantissa, exponent, common_e):
    """One common scale per draw; cap only edges above a feasible upper bound.

    The normalized feasible bound is below 2. An individual cost capped at
    4 cannot belong to a better assignment. No row/column reweighting occurs.
    """
    shift = jnp.where(mantissa == 0, 0, exponent - common_e[:, None, None])
    return jnp.minimum(scale_pow2(mantissa, jnp.minimum(shift, 2)), 4)


def _assignment_total(mantissa, exponent, perms):
    selected_m = jnp.take_along_axis(mantissa, perms[:, :, None], axis=2)[:, :, 0]
    selected_e = jnp.take_along_axis(exponent, perms[:, :, None], axis=2)[:, :, 0]
    common_e = jnp.max(jnp.where(selected_m > 0, selected_e,
                                jnp.iinfo(jnp.int32).min), axis=1)
    common_e = jnp.where(jnp.any(selected_m > 0, axis=1), common_e, 0)
    value = jnp.sum(scale_pow2(selected_m, selected_e - common_e[:, None]), axis=1)
    return _normalize_cost(value, common_e)


def _solve_costs_impl(mantissa, exponent, backend, q_enum):
    """Solve with retained original cost weights and range-aware refinement.

    Start at the largest edge's scale. If that loses smaller positive edges,
    the first feasible assignment supplies a tighter upper bound: rescale by
    that bound and cap edges that cannot improve it. Repeat only while the
    bound's exponent decreases. Ordinary-range batches need one solve.
    Floating-point sums and comparisons still have finite precision.
    """
    common_e = jnp.max(jnp.where(mantissa > 0, exponent,
                                jnp.iinfo(jnp.int32).min), axis=(1, 2))
    common_e = jnp.where(jnp.any(mantissa > 0, axis=(1, 2)), common_e, 0)
    perms = solve_assignment_batch(_scaled_costs(mantissa, exponent, common_e),
                                   backend=backend, q_enum=q_enum)
    total_m, total_e = _assignment_total(mantissa, exponent, perms)
    min_normal_e = jnp.finfo(mantissa.dtype).minexp

    def refinement_mask(state):
        _perms, tm, te, previous_e = state
        erased = jnp.any((mantissa > 0) &
                         (exponent - previous_e[:, None, None] < min_normal_e), axis=(1, 2))
        return (tm > 0) & (te < previous_e) & erased

    def needs_refinement(state):
        return jnp.any(refinement_mask(state))

    def refine(state):
        old_perms, old_m, old_e, previous_e = state
        active = refinement_mask(state)
        next_e = jnp.where(active, old_e, previous_e)
        new_perms = solve_assignment_batch(_scaled_costs(mantissa, exponent, next_e),
                                           backend=backend, q_enum=q_enum)
        new_m, new_e = _assignment_total(mantissa, exponent, new_perms)
        better = active & _less_cost(new_m, new_e, old_m, old_e)
        return (jnp.where(better[:, None], new_perms, old_perms),
                jnp.where(better, new_m, old_m), jnp.where(better, new_e, old_e), next_e)

    state = (perms, total_m, total_e, common_e)
    if backend == "scipy":
        while bool(needs_refinement(state)):
            state = refine(state)
    else:
        state = jax.lax.while_loop(needs_refinement, refine, state)
    return state[:3]


_SOLVE_DEVICE_COSTS = jax.jit(_solve_costs_impl, static_argnames=("backend", "q_enum"))


def _solve_costs(mantissa, exponent, backend, q_enum):
    # Enum calls this repeatedly with the same matrix shape. Reuse a single
    # compiled refinement loop instead of tracing a fresh closure per sign.
    fn = _solve_costs_impl if backend == "scipy" else _SOLVE_DEVICE_COSTS
    return fn(mantissa, exponent, backend=backend, q_enum=q_enum)


def apply_sign_perm(a: jax.Array, signs: jax.Array, perms: jax.Array) -> jax.Array:
    """B = A[:, perm] * sign (zero-based perms), batched over leading axis.

    a: (B, p, q); signs: (B, q) in {-1, +1}; perms: (B, q) bijections.
    """
    q = a.shape[-1]
    gathered = jnp.take_along_axis(a, perms[:, None, :].astype(jnp.int32), axis=2)
    return gathered * signs[:, None, :].astype(a.dtype)


def sp_step_elim(
    a: jax.Array,
    m: jax.Array,
    backend: str = "auto",
    max_chunk_elems: int = 1 << 26,
    q_enum: int = DEFAULT_Q_ENUM,
):
    """Elimination path (default). a: (B, p, q); m: (p, q).

    cost[j, k] = min(||a_k - m_j||^2, ||a_k + m_j||^2), both residuals computed
    DIRECTLY as squared differences. Do NOT use -|G| with G = M' A: the
    inner-product rounding erases the residual-scale gap between assignments
    on large near-equal inputs and can turn distinct assignments into ties.

    Memory: the (B, p, q, q) pairwise tensors are computed in chunks over B
    (two per chunk), with max_chunk_elems budgeting their combined element
    count. Inputs, outputs, and solver workspace require additional memory;
    this is not a bound on total device memory.
    Chunking partitions the batch without changing the per-draw problem.
    Returns (signs (B,q) int8, perms (B,q) int32, losses (B,)) where losses
    are full direct squared residuals at the ORIGINAL input magnitude (inf
    when the true loss exceeds the dtype range). `q_enum` is the "auto"
    exhaustive/optax crossover forwarded to the assignment solver.
    """
    a = jnp.asarray(a)
    m = jnp.asarray(m)
    if a.ndim != 3 or m.ndim != 2 or m.shape != a.shape[1:]:
        raise ValueError(
            f"sp_step_elim expects a (B, p, q) and m (p, q); got {a.shape} and {m.shape}"
        )
    b, p, q = a.shape

    def pair_costs(a_chunk):
        pm, pe, nm, ne, negative = _pair_tables(a_chunk, m)
        cm = jnp.where(negative, nm, pm)
        ce = jnp.where(negative, ne, pe)
        row_m, row_e = _row_minimum(cm, ce)
        cm, ce = _subtract_rows(cm, ce, row_m, row_e)
        return cm, ce, jnp.where(negative, -1, 1).astype(jnp.int8)

    chunk = max(1, min(b, max_chunk_elems // max(1, 2 * p * q * q)))
    pad = (-b) % chunk
    a_p = jnp.concatenate([a, jnp.zeros((pad, p, q), dtype=a.dtype)], axis=0) if pad else a
    n_chunks = a_p.shape[0] // chunk
    blocked = a_p.reshape(n_chunks, chunk, p, q)

    resolved = backend
    if resolved == "auto":
        resolved = "exhaustive" if q <= q_enum else "optax"

    if resolved != "scipy":
        # Construct costs and solve assignments within each chunk so pair
        # tables and residual tensors do not span the full batch. Recompute
        # original-scale losses afterward using the shared sum-of-squares
        # reduction; exactly aligned draws have zero residual.
        def chunk_step(a_chunk):
            cm, ce, sign_choice = pair_costs(a_chunk)
            perms, _tm, _te = _solve_costs(cm, ce, resolved, q_enum)
            signs = jnp.take_along_axis(sign_choice, perms[:, :, None], axis=2)[:, :, 0]
            return signs, perms

        signs, perms = jax.lax.map(chunk_step, blocked)
        signs = signs.reshape(n_chunks * chunk, q)[:b]
        perms = perms.reshape(n_chunks * chunk, q)[:b]
        aligned = apply_sign_perm(a, signs, perms)
        d = aligned - m[None]
        losses = sum_squares(d, axis=(1, 2))
        return signs, perms, losses

    # host scipy tier: two-phase (it cannot be traced inside lax.map)
    cm, ce, sign_choice = jax.lax.map(pair_costs, blocked)
    cm = cm.reshape(n_chunks * chunk, q, q)[:b]
    ce = ce.reshape(n_chunks * chunk, q, q)[:b]
    sign_choice = sign_choice.reshape(n_chunks * chunk, q, q)[:b]
    perms, _tm, _te = _solve_costs(cm, ce, "scipy", q_enum)
    signs = jnp.take_along_axis(sign_choice, perms[:, :, None], axis=2)[:, :, 0]
    aligned = apply_sign_perm(a, signs, perms)
    d = aligned - m[None]
    losses = sum_squares(d, axis=(1, 2))
    return signs, perms, losses


def sp_step_enum(a: jax.Array, m: jax.Array, backend: str = "auto", q_enum: int = DEFAULT_Q_ENUM):
    """Sign enumeration path. a: (B, p, q); m: (p, q).

    Costs are direct per-pair squared differences; the expanded
    norm form ||a||^2+||m||^2-2*s*G cancels catastrophically for large
    near-equal values.

    Both signed pair tables are constructed once, then the 2^q sign configs
    are processed sequentially. This path is intended for small q; the
    elimination path is the default.
    """
    a = jnp.asarray(a)
    m = jnp.asarray(m)
    if a.ndim != 3 or m.ndim != 2 or m.shape != a.shape[1:]:
        raise ValueError(
            f"sp_step_enum expects a (B, p, q) and m (p, q); got {a.shape} and {m.shape}"
        )
    q = a.shape[-1]
    b = a.shape[0]
    # Keep both sign costs in their original units. A shared row offset is
    # constant across ALL sign vectors, unlike a per-candidate normalization.
    pm, pe, nm, ne, negative = _pair_tables(a, m)
    row_m, row_e = _row_minimum(jnp.where(negative, nm, pm),
                               jnp.where(negative, ne, pe))
    signs_all = r_sign_enumeration(q)  # (S, q) int8, fixed enumeration order
    best_m = jnp.full((b,), jnp.inf, dtype=a.dtype)
    best_e = jnp.full((b,), jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    best_sign = jnp.ones((b, q), dtype=jnp.int8)
    best_perm = jnp.tile(jnp.arange(q, dtype=jnp.int32), (b, 1))
    for s in signs_all:
        positive = jnp.asarray(s)[None, :, None] == 1
        cm, ce = _subtract_rows(jnp.where(positive, pm, nm), jnp.where(positive, pe, ne),
                                row_m, row_e)
        perms, total_m, total_e = _solve_costs(cm, ce, backend, q_enum)
        better = _less_cost(total_m, total_e, best_m, best_e)  # strict: first minimum wins
        best_m = jnp.where(better, total_m, best_m)
        best_e = jnp.where(better, total_e, best_e)
        best_sign = jnp.where(better[:, None], jnp.asarray(s, dtype=jnp.int8)[None, :], best_sign)
        best_perm = jnp.where(better[:, None], perms, best_perm)
    aligned = apply_sign_perm(a, best_sign, best_perm)
    d = aligned - m[None]
    return best_sign, best_perm, sum_squares(d, axis=(1, 2))


def sp_step(a: jax.Array, m: jax.Array, path: str = "elim", backend: str = "auto",
            q_enum: int = DEFAULT_Q_ENUM, max_chunk_elems: int = 1 << 26):
    if path == "elim":
        return sp_step_elim(a, m, backend=backend, q_enum=q_enum,
                            max_chunk_elems=max_chunk_elems)
    if path == "enum":
        return sp_step_enum(a, m, backend=backend, q_enum=q_enum)
    raise ValueError(f"unknown SP path {path!r}")

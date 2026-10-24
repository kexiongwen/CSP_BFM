"""Batched linear assignment solvers.

Convention: given square cost matrices C (B, q, q), return per-matrix
permutations `perm` (B, q) such that `perm[j]` is the column assigned to row
`j` ("perm[rows] = cols", zero-based), minimizing sum_j C[j, perm[j]].

Backends:
  "exhaustive": all q! permutations, tie -> lexicographically smallest
      permutation (itertools order; jnp.argmin takes the first minimum).
      Chunked over the batch to bound the (chunk, q!) intermediate.
  "optax": optax.assignment.hungarian_algorithm, batched with vmap;
      tie-breaking is optax-internal, with no lexicographic guarantee.
  "scipy": scipy.optimize.linear_sum_assignment on the host (not jittable).

"auto" selects exhaustive for q <= q_enum, else optax; the default q_enum is 7.
"""
from __future__ import annotations

import itertools
from functools import lru_cache
from math import factorial

import jax
import jax.numpy as jnp
import numpy as np

from .._numeric import scale_pow2

DEFAULT_Q_ENUM = 7

#: construction-time byte budget for the exhaustive permutation table
_PERM_TABLE_MAX_BYTES = 1 << 30


@lru_cache(maxsize=None)
def _permutation_table(q: int) -> np.ndarray:
    # Check the factorial allocation before building the table. Streaming
    # into a preallocated array avoids a temporary list of permutation tuples.
    nbytes = factorial(q) * q * 4  # int32 entries
    if nbytes > _PERM_TABLE_MAX_BYTES:
        raise ValueError(
            f"exhaustive assignment for q={q} needs a {nbytes / 2**30:.1f} GiB "
            "permutation table; use backend='optax' instead"
        )
    flat = np.fromiter(
        itertools.chain.from_iterable(itertools.permutations(range(q))),
        dtype=np.int32,
        count=factorial(q) * q,
    )
    return flat.reshape(factorial(q), q)


def _exhaustive_chunk(costs: jax.Array, perms: jax.Array) -> jax.Array:
    """costs: (b, q, q); perms: (q!, q). Returns the best permutation per
    matrix: (b, q).

    A finite row offset is subtracted before summing. This offset is constant
    across feasible assignments, so in exact arithmetic the argmin and its
    lexicographic tie rule are unchanged. Removing the offset helps small
    differences between permutations survive float32 accumulation that would
    otherwise round distinct totals to the same value. An all-inf row keeps
    its entries (no inf-inf shift).

    General costs may also be negative. If subtraction would overflow, first
    apply a common positive binary scale to that matrix. Once row offsets
    are removed, a second scale protects the sum when necessary. Keeping
    ordinary row-offset removal first preserves small differences in rows
    whose large positive offsets cancel exactly."""
    q = costs.shape[-1]
    # The weak float literal promotes integer inputs to the configured float
    # dtype; low-precision floating inputs use at least f32 for safe scaling.
    finite_costs = jnp.where(jnp.isfinite(costs), costs, 0.0)
    working_dtype = jnp.promote_types(finite_costs.dtype, jnp.float32)
    costs = costs.astype(working_dtype)
    finite_costs = finite_costs.astype(working_dtype)
    offset = finite_costs.min(axis=2)  # (b, q)
    row_max = finite_costs.max(axis=2)
    max_value = jnp.asarray(jnp.finfo(costs.dtype).max, dtype=costs.dtype)
    subtraction_risk = jnp.any(
        (offset < 0) & (row_max > max_value + offset), axis=1
    )
    _, exponent = jnp.frexp(jnp.max(jnp.abs(finite_costs), axis=(1, 2)))
    shift = jnp.where(subtraction_risk, -exponent, 0)
    costs = scale_pow2(costs, shift[:, None, None])
    offset = jnp.where(jnp.isfinite(costs), costs, 0.0).min(axis=2)
    shifted = costs - offset[:, :, None]

    finite_shifted = jnp.where(jnp.isfinite(shifted), shifted, 0.0)
    largest = jnp.max(finite_shifted, axis=(1, 2))
    _, exponent = jnp.frexp(largest)
    # Leave a factor-two margin for rounded threshold arithmetic and sums.
    shift = jnp.where(largest > max_value / (2 * max(q, 1)), -exponent, 0)
    shifted = scale_pow2(shifted, shift[:, None, None])
    totals = jnp.zeros((costs.shape[0], perms.shape[0]), dtype=costs.dtype)
    for j in range(q):
        totals = totals + shifted[:, j, :][:, perms[:, j]]
    idx = jnp.argmin(totals, axis=1)
    return perms[idx]


def assignment_exhaustive(costs: jax.Array, max_perm_batch: int = 1 << 22) -> jax.Array:
    costs = jnp.asarray(costs)
    b, q, _ = costs.shape
    perms = jnp.asarray(_permutation_table(q))
    n_perm = perms.shape[0]
    chunk = max(1, min(b, max_perm_batch // n_perm))
    pad = (-b) % chunk
    if pad:
        pad_row = jnp.zeros((pad, q, q), dtype=costs.dtype)
        costs_p = jnp.concatenate([costs, pad_row], axis=0)
    else:
        costs_p = costs
    n_chunks = costs_p.shape[0] // chunk
    blocked = costs_p.reshape(n_chunks, chunk, q, q)
    out = jax.lax.map(lambda c: _exhaustive_chunk(c, perms), blocked)
    out = out.reshape(n_chunks * chunk, q)
    return out[:b]


def assignment_optax(costs: jax.Array) -> jax.Array:
    from optax.assignment import hungarian_algorithm

    costs = jnp.asarray(costs)
    b, q, _ = costs.shape
    rows, cols = jax.vmap(hungarian_algorithm)(costs)
    # normalize (row, col) pairs -> perm[rows] = cols
    batch_idx = jnp.arange(b)[:, None]
    perm = jnp.zeros((b, q), dtype=jnp.int32).at[batch_idx, rows].set(cols.astype(jnp.int32))
    return perm


def assignment_scipy(costs: jax.Array) -> jax.Array:
    """Solve assignments on the host with SciPy. Not jittable."""
    from scipy.optimize import linear_sum_assignment

    costs_np = np.asarray(costs, dtype=np.float64)
    out = np.empty((costs_np.shape[0], costs_np.shape[1]), dtype=np.int32)
    for i, c in enumerate(costs_np):
        rows, cols = linear_sum_assignment(c)
        perm = np.empty(c.shape[0], dtype=np.int32)
        perm[rows] = cols
        out[i] = perm
    return jnp.asarray(out)


def solve_assignment_batch(
    costs: jax.Array,
    backend: str = "auto",
    q_enum: int = DEFAULT_Q_ENUM,
) -> jax.Array:
    """Dispatch assignment solver. jit-safe: only static (shape) checks here;
    finiteness of the *values* is validated at the public API boundary.
    """
    costs = jnp.asarray(costs)
    if costs.ndim != 3 or costs.shape[-1] != costs.shape[-2]:
        raise ValueError(f"costs must be (B, q, q) with square last dims, got {costs.shape}")
    if backend == "auto":
        backend = "exhaustive" if costs.shape[-1] <= q_enum else "optax"
    if backend == "exhaustive":
        return assignment_exhaustive(costs)
    if backend == "optax":
        return assignment_optax(costs)
    if backend == "scipy":
        return assignment_scipy(costs)
    raise ValueError(f"unknown assignment backend {backend!r}")


__all__ = [
    "solve_assignment_batch",
    "assignment_exhaustive",
    "assignment_optax",
    "assignment_scipy",
    "DEFAULT_Q_ENUM",
]

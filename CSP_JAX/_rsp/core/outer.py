"""Outer alternation for the RSP algorithm.

One outer sweep = SP step for all draws against the current reference, then
reference update to the aligned mean.

`f_pre` is the squared residual against the reference used for matching,
before the mean update. `f_post` uses the updated reference.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .._numeric import safe_mean, sum_squares
from ..kernels.assignment import DEFAULT_Q_ENUM
from .sp_step import apply_sign_perm, sp_step


def outer_step(rotated: jax.Array, reference: jax.Array, sp_path: str, backend: str,
               q_enum: int = DEFAULT_Q_ENUM, max_chunk_elems: int = 1 << 26):
    """rotated: (T, p, q); reference: (p, q).

    Returns (signs, perms, aligned, new_reference, f_pre, f_post).
    `sp_path`/`backend`/`q_enum`/`max_chunk_elems` must be static (closure
    capture under jit).
    """
    signs, perms, losses = sp_step(rotated, reference, path=sp_path, backend=backend,
                                   q_enum=q_enum, max_chunk_elems=max_chunk_elems)
    aligned = apply_sign_perm(rotated, signs, perms)
    new_reference = safe_mean(aligned, axis=0)  # scale-safe mean reduction
    # Reduce all residuals together: summing per-draw subnormal losses can
    # flush them to zero even when the total is representable.
    f_pre = sum_squares(aligned - reference[None])
    d_post = aligned - new_reference[None]
    f_post = sum_squares(d_post)
    return signs, perms, aligned, new_reference, f_pre, f_post


def make_step_fn(sp_path: str, backend: str, q_enum: int = DEFAULT_Q_ENUM,
                 max_chunk_elems: int = 1 << 26):
    """Return the single-sweep function with static config bound.

    Jitted for device backends; the host-side "scipy" backend runs eagerly
    because its NumPy conversion cannot be traced by jax.jit.
    """
    step = lambda rot, ref: outer_step(  # noqa: E731
        rot, ref, sp_path, backend, q_enum, max_chunk_elems
    )
    return step if backend == "scipy" else jax.jit(step)

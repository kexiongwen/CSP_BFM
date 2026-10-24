"""Transform factor scores using accumulated orthogonal transforms."""
from __future__ import annotations

import jax.numpy as jnp


def transform_scores(scores, transforms):
    """Apply the accumulated orthogonal transforms to factor scores.

    scores: (T, n, q) or (n, q) row-stored factor vectors f (i.e. rows of the
    score matrix). transforms: (T, q, q) or (q, q) from RSPResult.transforms.

    Valid combinations: (n,q)+(q,q) -> (n,q); (T,n,q)+(T,q,q) or
    (T,n,q)+(q,q) -> (T,n,q); (n,q)+(T,q,q) broadcasts the single score
    matrix under every draw's transform -> (T,n,q).

    Row-stored convention: f' = f @ O. For column-vector scores F this is
    F' = Oᵀ F — the same operation on transposed storage. Prediction is
    invariant: (Λ O)(f O)ᵀ = Λ fᵀ.
    """
    scores = jnp.asarray(scores)
    transforms = jnp.asarray(transforms)
    if scores.ndim == 2 and transforms.ndim == 2:
        return scores @ transforms
    if scores.ndim == 3 and transforms.ndim == 3 and scores.shape[0] == transforms.shape[0]:
        return scores @ transforms
    if scores.ndim == 3 and transforms.ndim == 2:
        return scores @ transforms[None]
    if scores.ndim == 2 and transforms.ndim == 3:
        # one shared score matrix under every draw's transform -> (T, n, q)
        return scores[None, :, :] @ transforms
    raise ValueError(
        f"scores {scores.shape} and transforms {transforms.shape}: valid combos are "
        "(n,q)+(q,q), (T,n,q)+(T,q,q), (T,n,q)+(q,q), (n,q)+(T,q,q); per-draw "
        "transforms need a leading T axis in scores or a single shared (n, q) matrix"
    )

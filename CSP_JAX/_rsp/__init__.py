"""Rotation-Sign-Permutation (RSP) alignment of posterior loadings, in JAX.

Given an MCMC sample of loading matrices, :func:`rsp_exact` applies a
per-draw varimax rotation followed by exact sign-permutation alignment;
:func:`transform_scores` applies the resulting orthogonal transforms to
factor scores.

optax (the assignment solver for q > 7) and scipy (the "scipy" assignment
backend) are imported lazily inside the functions that need them.
"""
from .api import rsp_exact
from .chains import transform_scores
from .io import from_flat, lambda_colnames, to_flat
from .types import RSPResult

__all__ = [
    "rsp_exact",
    "transform_scores",
    "from_flat",
    "to_flat",
    "lambda_colnames",
    "RSPResult",
]

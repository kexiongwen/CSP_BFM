"""Flat-array conventions and strict input validation.

Flat layout: a (T, p*q) matrix whose columns are the entries of the
p x q loading matrix in variable-major order (C order), i.e.
`Lambda.reshape(T, p * q)`. Column names, when present, must be exactly
`LambdaV1_1 ... LambdaV1_q ... LambdaVp_1 ... LambdaVp_q`.
"""
from __future__ import annotations

import numpy as np


def lambda_colnames(p: int, q: int) -> list[str]:
    return [f"LambdaV{r + 1}_{j + 1}" for r in range(p) for j in range(q)]


def _positive_dim(value, name: str) -> int:
    """Require a positive integer dimension before reshaping input."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def parse_lambda_colnames(colnames: list[str], d: int) -> tuple[int, int]:
    """Infer (p, q) from strict LambdaV names.

    q is read from the last column name, p = d / q; every name must
    match the expected sequence exactly.
    """
    if len(colnames) != d:
        raise ValueError(f"got {len(colnames)} column names for {d} columns")
    if not colnames:
        raise ValueError(
            "colnames must be a non-empty LambdaV1_1 ... LambdaVp_q sequence"
        )
    last = colnames[-1]
    if not (isinstance(last, str) and last.startswith("LambdaV") and "_" in last):
        raise ValueError(
            "column names must be formatted as LambdaV1_1 ... LambdaVp_q"
        )
    try:
        q = int(last.rsplit("_", 1)[1])
    except ValueError as exc:
        raise ValueError(f"cannot parse q from last column name {last!r}") from exc
    if q < 1 or d % q != 0:
        raise ValueError(f"d={d} is not an integer multiple of q={q}")
    p = d // q
    expected = lambda_colnames(p, q)
    if list(colnames) != expected:
        raise ValueError(
            "column names do not match the expected LambdaV sequence "
            f"(first mismatch or wrong order; expected e.g. {expected[:3]} ... {expected[-1]})"
        )
    return p, q


def _resolve_pq(colnames, p, q, d):
    """Shared (p, q) resolution for flat (T, p*q) input: strict colnames or
    explicit p and q (no guessing)."""
    if colnames is not None:
        if p is not None or q is not None:
            raise ValueError("pass either colnames or explicit (p, q), not both")
        return parse_lambda_colnames(list(colnames), d)
    if p is None or q is None:
        raise ValueError("without colnames, explicit p and q are required")
    p = _positive_dim(p, "p")
    q = _positive_dim(q, "q")
    if p * q != d:
        raise ValueError(f"p*q = {p}*{q} != {d}")
    return p, q


def from_flat(flat, colnames=None, p: int | None = None, q: int | None = None, dtype=np.float64):
    """(T, p*q) flat array -> (T, p, q).

    With `colnames`, (p, q) are inferred and strictly validated. Without names,
    p and q must both be given explicitly (no guessing).
    """
    arr = np.asarray(flat, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"flat loadings must be 2-D (T, p*q), got shape {arr.shape}")
    t, d = arr.shape
    p, q = _resolve_pq(colnames, p, q, d)
    if not np.isfinite(arr).all():
        raise ValueError("loadings contain non-finite values")
    return arr.reshape(t, p, q)


def to_flat(arr) -> np.ndarray:
    """(T, p, q) -> (T, p*q) in variable-major order."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected (T, p, q), got shape {arr.shape}")
    return arr.reshape(arr.shape[0], -1)

"""ESS.py  -- effective-sample-size diagnostics for the Gibbs chain.

Computes per-entry effective sample sizes (ESS) of the kept draws
produced by :func:`CSP_JAX.GS_CSP` -- the loading matrix B
("B", shape (S, p, H)), the noise variances sigma2 ("sigma2", (S, p))
and, specific to the cumulative shrinkage process, the scalar
active-factor chain H* ("H_star", (S,)) -- via ArviZ
(:func:`arviz.ess`).

ArviZ is an OPTIONAL dependency of this package, imported lazily: every
other module here works without arviz except the functions of this
file, which raise ImportError with an install hint when it is missing.

Conventions
-----------
* Input arrays are the kept draws exactly as returned by GS_CSP
  (numpy float32 on the host; cast to float64 here).  Multi-chain
  arrays (C, S, ...) are accepted as-is.
* Inputs are normalized before any ArviZ computation: a single-chain
  input gets a chain axis prepended, and B, sigma2 and hs must share
  the same chain count C and draws per chain S; sigma2 must also
  describe the same p entries as B.  Mixing multi-chain and
  single-chain inputs is rejected with ValueError.
* ArviZ expects a (chain, draw, ...) layout.  The normalized draws
  are passed directly as ndarrays; ArviZ >= 1.3 is required for these
  n-dimensional inputs.  ESS then has the entry shape: (p, H) for B,
  (p,) for sigma2, and a plain float for the scalar H* chain.
* Empty chain or entry dimensions are rejected with ValueError
  instead of being allowed to produce NaN ESS values.
* The H* chain is a small non-negative integer; the rank-normalized
  bulk ESS is still a valid mixing summary for it.  A DEGENERATE
  constant chain (e.g. H* = 5 in every kept draw, as happens when the
  posterior of the dimension concentrates completely) reports ESS =
  n_draws: there is no autocorrelation to correct for.
* ``method`` and ``prob`` are passed through to :func:`arviz.ess`:
  "bulk" (default; rank-normalized bulk ESS, the recommended summary),
  "tail" and "quantile" (both require ``prob``, e.g. prob=0.9 / 0.05),
  "mean", "sd", "median", ...  ``relative=True`` returns ESS per draw
  (ESS divided by the total number of draws).
* Rule of thumb (Vehtari et al., 2021, doi:10.1214/20-BA1221): bulk
  ESS above ~400 per entry makes the Monte Carlo error of posterior
  means negligible for most purposes.

This module mirrors the ESS.py addition of the MGP_BFM package
(mgp._gs.ESS), with the H* chain as a CSP-specific extension.
"""

import numpy as np

__all__ = ["ess", "ess_summary"]


def _require_arviz():
    """Import arviz lazily, with an actionable error when missing."""
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError(
            "ESS diagnostics require arviz (e.g. pip install arviz)"
        ) from exc
    return az


def _as_chain_draw(a, name: str, entry_ndim: int) -> np.ndarray:
    """Validate kept draws and return the (chain, draw, ...) float64
    layout."""
    a = np.asarray(a, dtype=np.float64)   # also accepts the int H* chain
    if a.ndim == entry_ndim + 1:        # single chain: (S, ...) -> (1, S, ...)
        a = a[None]
    elif a.ndim != entry_ndim + 2:      # multi-chain: (C, S, ...)
        raise ValueError(
            f"{name} must have shape (S, ...) or (C, S, ...) with "
            f"{entry_ndim} entry dimension(s), got shape {a.shape}"
        )
    if a.shape[0] == 0:
        raise ValueError(
            f"{name} has no chains (shape {a.shape}); ESS is undefined"
        )
    if any(size == 0 for size in a.shape[2:]):
        raise ValueError(
            f"{name} has an empty entry dimension (shape {a.shape}); "
            "ESS is undefined"
        )
    if a.shape[1] < 4:
        raise ValueError(
            f"{name} has only {a.shape[1]} draws per chain; ESS needs at "
            "least 4 kept draws"
        )
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite draws; ESS is undefined")
    return a


def _validate_layout(B: np.ndarray, sigma2, hs) -> None:
    """Check that normalized draws share one (chain, draw) layout."""
    n_chains, n_draws = B.shape[:2]
    p = B.shape[2]
    for name, a in (("sigma2", sigma2), ("hs", hs)):
        if a is None:
            continue
        if a.shape[:2] != (n_chains, n_draws):
            raise ValueError(
                f"{name} has shape {a.shape} with {a.shape[0]} chain(s) and "
                f"{a.shape[1]} draw(s) per chain, but B has {n_chains} "
                f"chain(s) and {n_draws} draw(s) per chain; B, sigma2 and "
                "hs must share the same chain layout (do not mix multi-chain "
                "and single-chain inputs)"
            )
    if sigma2 is not None and sigma2.shape[2] != p:
        raise ValueError(
            f"sigma2 has p={sigma2.shape[2]} entries but B has p={p}; "
            "B and sigma2 must have the same number of entries"
        )


def _prepare_inputs(B, sigma2, hs):
    """Normalize all draws, then validate their shared layout."""
    Bc = _as_chain_draw(B, "B", entry_ndim=2)
    s2c = None if sigma2 is None else _as_chain_draw(
        sigma2, "sigma2", entry_ndim=1
    )
    hsc = None if hs is None else _as_chain_draw(hs, "hs", entry_ndim=0)
    _validate_layout(Bc, s2c, hsc)
    return Bc, s2c, hsc


def _ess_from_chain_draw(Bc, s2c, hsc, method, prob, relative):
    """Run ArviZ on draws with a validated shared layout."""
    az = _require_arviz()
    out = {
        "B": np.asarray(az.ess(Bc, method=method, prob=prob,
                               relative=relative))
    }
    if s2c is not None:
        out["sigma2"] = np.asarray(az.ess(
            s2c, method=method, prob=prob, relative=relative
        ))
    if hsc is not None:
        out["H_star"] = float(np.asarray(az.ess(
            hsc, method=method, prob=prob, relative=relative
        )))
    return out


def ess(B, sigma2=None, hs=None, method: str = "bulk", prob=None,
        relative: bool = False):
    """Per-entry effective sample size of the kept B / sigma2 / H* draws.

    Parameters
    ----------
    B : array-like, shape (S, p, H) or (C, S, p, H)
        Kept loading-matrix draws ("B" of the GS_CSP output).
    sigma2 : array-like, shape (S, p) or (C, S, p), optional
        Kept noise-variance draws ("sigma2" of the GS_CSP output).
    hs : array-like, shape (S,) or (C, S), optional
        Kept active-factor draws ("H_star" of the GS_CSP output).
    method : str, default "bulk"
        ESS kind, passed to :func:`arviz.ess` ("bulk", "tail", "mean",
        "sd", "median", "quantile", ...).  "tail" and "quantile"
        require ``prob``.
    prob : float, optional
        Probability for the "tail"/"quantile" methods (e.g. 0.9 / 0.05).
    relative : bool, default False
        If True, return ESS per draw (ESS / total draws).

    Returns
    -------
    dict
        {"B": (p, H) ndarray} plus {"sigma2": (p,) ndarray} when
        ``sigma2`` is given and {"H_star": float} when ``hs`` is given.

    Raises
    ------
    ValueError
        If an input has an unsupported shape, an empty chain or entry
        dimension, too few draws, non-finite values, or a chain/draw/p
        layout that does not match the other inputs.
    """
    Bc, s2c, hsc = _prepare_inputs(B, sigma2, hs)
    return _ess_from_chain_draw(Bc, s2c, hsc, method=method, prob=prob,
                                relative=relative)


def ess_summary(B, sigma2=None, hs=None, method: str = "bulk"):
    """Compact ESS summary over the entries of B (and sigma2 / H*).

    The worst entries drive the reliability of the chain, so the
    minimum is the number to watch (see the module docstring for the
    ~400 rule of thumb).

    Parameters
    ----------
    B, sigma2, hs : array-like
        Kept draws, as in :func:`ess`.

    Returns
    -------
    dict
        One entry per input array, each a dict with keys "min",
        "median", "mean", "max" (floats) and "n_draws" (total draws
        behind the ESS values, C * S for the shared layout).
    """
    Bc, s2c, hsc = _prepare_inputs(B, sigma2, hs)
    e = _ess_from_chain_draw(Bc, s2c, hsc, method=method, prob=None,
                             relative=False)
    n_draws = int(Bc.shape[0] * Bc.shape[1])
    return {
        key: {
            "min": float(np.min(a)),
            "median": float(np.median(a)),
            "mean": float(np.mean(a)),
            "max": float(np.max(a)),
            "n_draws": int(n_draws),
        }
        for key, a in ((k, np.atleast_1d(v)) for k, v in e.items())
    }

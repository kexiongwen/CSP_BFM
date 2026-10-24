"""Core result types.

Array dtypes follow the call's `precision` (float32 by default in this
package; "float64" opt-in enables x64 at the entry point, never at import).
Permutations are zero-based (T, q) bijections.

PyTree contract: numeric, run-dependent arrays live in leaf
fields (children); `diagnostics` and `metadata` are flattened into sorted
tuples of hashable scalars or scalar sequences as aux data, so two
independent results with equal configuration compare equal and can enter the
same jit-compiled function repeatedly. Unsupported values in those dicts
raise a TypeError with guidance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numbers

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


def _scalar_items(d: dict | None, name: str) -> tuple:
    items = []
    for k, v in (d or {}).items():
        if isinstance(v, bool) or v is None:
            items.append((k, v))
        elif isinstance(v, numbers.Integral):
            items.append((k, int(v)))
        elif isinstance(v, numbers.Real):
            items.append((k, float(v)))
        elif isinstance(v, str):
            items.append((k, v))
        elif isinstance(v, (list, tuple)) and all(
            x is None or isinstance(x, (str, bool, numbers.Integral, numbers.Real)) for x in v
        ):
            # sequences of scalars (e.g. variable names) are normalized to
            # comparable, hashable tuples. Check bool before numeric types
            # so boolean values keep their type through the round-trip.
            items.append((k, tuple(bool(x) if isinstance(x, bool)
                                   else int(x) if isinstance(x, numbers.Integral)
                                   else float(x) if isinstance(x, numbers.Real)
                                   else x for x in v)))
        else:
            raise TypeError(
                f"{name}[{k!r}] must be a hashable scalar or a sequence of scalars; "
                "run-dependent arrays belong to the leaf fields (loss_history_post, "
                "transforms, varimax_n_iter)"
            )
    return tuple(sorted(items, key=lambda kv: kv[0]))


@register_pytree_node_class
@dataclass
class RSPResult:
    """Result of an RSP alignment run.

    Losses are sums of squared Frobenius distances across draws, evaluated
    against the reference before or after each mean update.

    aligned_loadings:    (T, p, q) reordered draws.
    mean_loadings:       (p, q) mean of aligned draws.
    signs:               (T, q) int8 in {-1, +1}.
    permutations:        (T, q) int32 zero-based bijections.
    loss_initial:        f0 = sum_t ||A_t - M_0||_F^2 against the initial
                         reference; A_t are the optionally rotated inputs.
    loss_final:          sum_t ||aligned_t - mean_loadings||_F^2 after the
                         final mean update.
    loss_history:        (max_iter,) buffer; first n_iter entries: pre-update
                         objectives (recorded before the reference update).
    loss_history_post:   (max_iter,) buffer; objectives against each sweep's
                         updated mean.
    n_iter:              number of outer iterations actually run.
    converged:           stopping threshold was met (status_code == 1).
    status_code:         1 = converged, 2 = max_iter reached,
                         3 = anomaly, 4 = non-finite objective
                         (the original-scale loss is unrepresentable or a
                         non-finite intermediate appeared).
    transforms:          optional (T, q, q) full orthogonal transforms per draw;
                         None when return_transforms=False.
    varimax_n_iter:      optional (T,) int32 per-draw varimax iteration counts
                         (a count at the update limit indicates that the
                         iteration budget was exhausted; None when no
                         varimax pre-rotation ran). These counts do not
                         certify a local or global maximum of the varimax
                         criterion.
    diagnostics:         static configuration/scalars (hashable pairs in aux).
                         Includes `stop_reason` ("converged_threshold" /
                         "max_iter" / "anomaly" / "nonfinite_objective").
                         `nonfinite_fields` names quantities found non-finite
                         by the output checks (an empty tuple otherwise).
    metadata:            MCMC metadata held by the outer object (chain name,
                         variable names, start/thin, ...); kernels never depend
                         on it. Populated by the caller.
    """

    aligned_loadings: jnp.ndarray
    mean_loadings: jnp.ndarray
    signs: jnp.ndarray
    permutations: jnp.ndarray
    loss_initial: jnp.ndarray
    loss_final: jnp.ndarray
    loss_history: jnp.ndarray
    n_iter: jnp.ndarray
    converged: jnp.ndarray
    status_code: jnp.ndarray
    transforms: jnp.ndarray | None = None
    loss_history_post: jnp.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    varimax_n_iter: jnp.ndarray | None = None

    @property
    def loss_history_valid(self) -> jnp.ndarray:
        """The recorded pre-update objectives: loss_history[:n_iter]."""
        return self.loss_history[: int(self.n_iter)]

    @property
    def loss_history_post_valid(self) -> jnp.ndarray | None:
        """The recorded post-update objectives: loss_history_post[:n_iter]
        (None when not recorded)."""
        if self.loss_history_post is None:
            return None
        return self.loss_history_post[: int(self.n_iter)]

    def __repr__(self):
        try:
            t, p, q = self.aligned_loadings.shape
            return (
                f"RSPResult(T={t}, p={p}, q={q}, n_iter={int(self.n_iter)}, "
                f"status_code={int(self.status_code)}, converged={bool(self.converged)}, "
                f"loss_initial={float(self.loss_initial):.6g}, "
                f"loss_final={float(self.loss_final):.6g})"
            )
        except Exception:  # e.g. tracers under jit: fall back to a shapeless repr
            return "RSPResult(...)"

    def tree_flatten(self):
        children = (
            self.aligned_loadings,
            self.mean_loadings,
            self.signs,
            self.permutations,
            self.loss_initial,
            self.loss_final,
            self.loss_history,
            self.n_iter,
            self.converged,
            self.status_code,
            self.transforms,
            self.loss_history_post,
            self.varimax_n_iter,
        )
        aux = (
            _scalar_items(self.diagnostics, "diagnostics"),
            _scalar_items(self.metadata, "metadata"),
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        diag_items, meta_items = aux_data
        (aligned, mean, signs, perms, li, lf, lh, n_it, conv, sc,
         tr, lhp, vni) = children
        return cls(
            aligned_loadings=aligned,
            mean_loadings=mean,
            signs=signs,
            permutations=perms,
            loss_initial=li,
            loss_final=lf,
            loss_history=lh,
            n_iter=n_it,
            converged=conv,
            status_code=sc,
            transforms=tr,
            loss_history_post=lhp,
            varimax_n_iter=vni,
            diagnostics=dict(diag_items),
            metadata=dict(meta_items),
        )

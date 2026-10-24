"""CSP_JAX: JAX Gibbs sampler for the Bayesian cumulative shrinkage
process factor model (Legramanti, Durante & Dunson, 2020).

Functional interface (the sampler driver and its blocks) plus a
sklearn-style estimator layer:

* :func:`GS_CSP` -- the sampler driver (Algorithm 1 of the paper);
* :class:`GibbsCSP` -- estimator wrapper with unified fitted accessors
  (``B_hat_`` / ``sigma2_hat_`` / ``Sigma_`` / ``n_factors``), RSP
  factor-switching post-processing (:meth:`GibbsCSP.align_factors`,
  provided by :mod:`CSP_JAX._rsp`), on-demand factor scores
  (:meth:`GibbsCSP.sample_factors`) and Monte Carlo posterior moments
  of the model covariance.

Runtime dependencies: JAX/jaxlib and NumPy.  arviz (ESS diagnostics),
optax (RSP assignment backend for H > 7) and scipy (the "scipy"
assignment backend) are optional and imported lazily.
"""

from .base import BaseCSP
from .ESS import ess, ess_summary
from .gibbs import GibbsCSP
from .GS_CSP import GS_CSP, Initialization
from .shrinkage import make_log_t_constant, shrinkage, z_logits

__all__ = ["GS_CSP", "Initialization", "shrinkage", "z_logits",
           "make_log_t_constant", "ess", "ess_summary",
           "BaseCSP", "GibbsCSP"]

"""Gibbs sampler estimator for the CSP Bayesian factor model.

sklearn-style wrapper of the JAX Gibbs sampler driver
:func:`CSP_JAX.GS_CSP` (Algorithm 1 of Legramanti, Durante & Dunson,
2020, "Bayesian cumulative shrinkage for infinite factorizations").  All
numerical semantics (float32, chunking, audits) are those of the
underlying driver; this class only adds the estimator interface and the
shared preprocessing of :class:`CSP_JAX.base.BaseCSP`.

The class-layer design is ported from ``GibbsBFM`` of the ``bfm``
package (LH_BFM), with the truncation index named H and the CSP-specific
active-factor chain H* exposed as ``H_star_samples_``.
"""

import warnings

import numpy as np

from ._posterior import (
    montecarlo_cov_sigma_pairs,
    montecarlo_sigma_diag_variance,
    montecarlo_sigma_mean,
    montecarlo_sigma_variance,
    suggest_max_draws,
)
from .base import BaseCSP
from .GS_CSP import GS_CSP

__all__ = ["GibbsCSP"]


class GibbsCSP(BaseCSP):
    """Gibbs sampler for the CSP Bayesian factor model (exact MCMC).

    Parameters
    ----------
    H : int
        Truncation level of the loading matrix (number of columns).
        At most ``H - 1`` factors can be active, since the Hth column is
        spiked by construction.
    alpha : float, default 4.0
        CSP concentration parameter: the prior expected number of
        active factors in the infinite process, before truncation.
        The paper used 5.
    a_theta, b_theta : float
        IG(a_theta, b_theta) slab hyperparameters of the spike-and-slab
        prior on theta_h; the defaults (2, 2) satisfy the paper's
        Theorem 2 condition a_theta > 1.
    theta_inf : float
        Spike variance theta_infty (NOT a standard deviation).  Paper
        default 0.05.
    a_sigma, b_sigma : float
        Inverse-gamma hyperparameters of the noise variances.
        Defaults (1, 1); the paper used (1, 0.3)
        (Bhattacharya & Dunson, 2011).
    center : bool, default True
        Subtract the column means of X before fitting.
    n_iter, burn_in, thin : int
        Total sweeps, discarded initial sweeps, and keep-every-thin.
    seed : int or None
        Seed for jax.random.PRNGKey; None draws a random seed.  The same
        seed (on the same backend) reproduces the same chain bit-exactly.
    verbose : bool, default True
        Print progress about every 10% of the sweeps.

    Attributes
    ----------
    B_samples_ : (S, p, H) numpy array
        Kept loading-matrix draws, S = (n_iter - burn_in) // thin.
        These RAW draws are never modified by :meth:`align_factors`;
        ``ess()`` also continues to operate on them.
    sigma2_samples_ : (S, p) numpy array
        Kept noise-variance draws.
    H_star_samples_ : (S,) numpy array
        Kept number of active factors per draw (columns with z_h > h).
        Its posterior mean estimates E(H* | y), the model dimension --
        the paper's headline feature.  Always computed from the raw
        chain; alignment does not change it.  Note that z is drawn
        BEFORE the loading update within a sweep, so
        ``H_star_samples_[s]`` and ``B_samples_[s]`` differ by half a
        sweep: every marginal summary of either chain is exact, but the
        two arrays are not a synchronous joint posterior draw -- do not
        pair them entry by entry.
    seed_ : int
        The seed actually used.
    runtime_sec_ : float
        Wall-clock seconds of the scan loop (excludes AOT compilation).
    B_hat_, sigma2_hat_, Sigma_ : numpy arrays
        Posterior-mean point estimates (see
        :class:`CSP_JAX.base.BaseCSP`); after :meth:`align_factors`,
        ``B_hat_``/``Sigma_`` reflect the ALIGNED draws.
    Sigma_mc_ : (p, p) numpy array
        Monte Carlo posterior mean of Sigma = B B^T + Omega,
        (1/S) sum_s (B_s B_s^T + diag(sigma2_s)); invariant to per-draw
        orthogonal transforms, hence identical before and after
        :meth:`align_factors`.  Aliased as ``Sigma_mean_``.
    Sigma_mean_ : (p, p) numpy array
        Alias of :attr:`Sigma_mc_` (Monte Carlo posterior mean of Sigma).
    Sigma_var_, Sigma_sd_ : (p, p) numpy arrays
        Monte Carlo entrywise posterior variance and standard deviation
        of Sigma, using ddof = 0.  ``Sigma_var_`` may use a deterministic
        equidistant subsample of the draws when the full scan exceeds the
        default compute budget; see ``Sigma_draws_used_`` and
        :meth:`posterior_sigma_var`.
    Sigma_diag_var_ : (p,) numpy array
        Entrywise posterior variance of the diagonal of Sigma; cheap
        (O(S p H)) and always computed from all draws.
    Sigma_draws_used_ : int or None
        Number of draws used by the most recent ``Sigma_var_``
        computation (``None`` before it is computed).
    rsp_result_, B_aligned_
        Produced by :meth:`align_factors` (see
        :meth:`CSP_JAX.base.BaseCSP.align_factors`); removed by the
        next successful ``fit`` call.
    raw_output_ : dict
        The unmodified return value of the underlying driver, including
        ``eta_last``, ``D_last``, ``V_last`` and ``H_star_last`` for
        diagnostics.  ``D_last``/``V_last`` are never aligned (the
        column-wise shrinkage scales and stick weights have no
        counterpart under the varimax rotation).

    Examples
    --------
    Fit, select the dimension, and resolve factor switching::

        gs = GibbsCSP(H=10, n_iter=2000, burn_in=1000, seed=2025).fit(X)
        gs.n_factors()              # posterior median of H* (CSP rule)
        gs.h_star_summary()         # mean / median / 95% CI of H*
        gs.align_factors()          # RSP: varimax + sign/permutation
        gs.B_hat_                   # mean of the ALIGNED draws
        gs.rsp_result_.diagnostics  # status, stop reason, losses

    Return factor scores only when needed::

        scores = gs.sample_factors(X, return_eta=True,
                                   return_eta_aligned=True, seed=123)
        scores["eta_sample"]          # (S, H, n), paired to B_samples_
        scores["eta_sample_aligned"]  # (S, H, n), paired to B_aligned_

    Notes
    -----
    ``Sigma_`` is the plug-in estimate ``B_hat B_hat^T +
    diag(sigma2_hat_)``; ``Sigma_mc_``/``Sigma_mean_`` is the Monte Carlo
    posterior mean of ``Sigma = B B^T + Omega``.  ``Sigma_var_``,
    ``Sigma_sd_``, ``Sigma_diag_var_`` and :meth:`cov_Sigma` estimate the
    posterior covariance of Sigma from the draws with ddof = 0; they are
    noisy Monte Carlo estimates (and the draws are autocorrelated), not
    exact moments.  The full p x p variance scan costs O(S p^2 H); when
    it exceeds the default compute budget, ``Sigma_var_`` uses a
    deterministic equidistant subsample (with a warning) and records the
    count in ``Sigma_draws_used_``.  Use :meth:`posterior_sigma_var` to
    control this explicitly, or the cheap ``Sigma_diag_var_`` /
    :meth:`cov_Sigma` for selected entries on large problems.  As for
    the other covariance summaries, the moments are computed from the
    raw ``B_samples_`` and ``sigma2_samples_`` and are invariant to
    :meth:`align_factors`.
    """

    def __init__(
        self,
        H,
        alpha=4.0,
        a_theta=2.0,
        b_theta=2.0,
        theta_inf=0.05,
        a_sigma=1.0,
        b_sigma=1.0,
        center=True,
        n_iter=2000,
        burn_in=1000,
        thin=1,
        seed=None,
        verbose=True,
    ):
        super().__init__(
            H, alpha=alpha, a_theta=a_theta, b_theta=b_theta,
            theta_inf=theta_inf, a_sigma=a_sigma, b_sigma=b_sigma,
            center=center,
        )
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.thin = thin
        self.seed = seed
        self.verbose = verbose

    def _fit_arrays(self, Xc):
        # Posterior-moment cache (Sigma_mc_/Sigma_var_/...); a new fit must
        # not reuse moments of the previous fit.
        self._sigma_cache = {}
        self.Sigma_draws_used_ = None
        out = GS_CSP(
            Xc,
            self.H,
            alpha=self.alpha,
            a_theta=self.a_theta,
            b_theta=self.b_theta,
            theta_inf=self.theta_inf,
            a_sigma=self.a_sigma,
            b_sigma=self.b_sigma,
            n_iter=self.n_iter,
            burn_in=self.burn_in,
            thin=self.thin,
            seed=self.seed,
            verbose=self.verbose,
        )
        self.raw_output_ = out
        self.B_samples_ = out["B"]
        self.sigma2_samples_ = out["sigma2"]
        self.H_star_samples_ = out["H_star"]
        self.seed_ = out["seed"]
        self.runtime_sec_ = out["runtime_sec"]
        self._B_hat = self.B_samples_.mean(axis=0)
        self._sigma2_hat = self.sigma2_samples_.mean(axis=0)

    def sample_factors(
        self,
        X,
        *,
        return_eta=True,
        return_eta_aligned=False,
        seed=None,
        batch_size=32,
    ):
        """Return optional factor-score draws paired to all kept samples.

        For every kept ``(B_samples_[s], sigma2_samples_[s])``, draw a
        fresh ``eta_s`` from its conditional distribution given X.  For
        the original training data, these are joint posterior samples,
        not the historical intermediate scores from the Gibbs sweeps.
        They are correctly paired even when thinning omitted the final
        sweep.  This method does not change or store any fitted
        attributes.

        Parameters
        ----------
        X : array-like, shape (n, p)
            Uncentered observations in the fitted feature order.  The
            saved ``x_offset_`` is subtracted; do not center X again.
            Pass the original training data for posterior factor scores.
            New rows produce conditional scores under the already-fitted
            parameter draws, without updating the loading posterior.
        return_eta : bool, default True
            Include raw scores under the key ``"eta_sample"``.
        return_eta_aligned : bool, default False
            Include ``"eta_sample_aligned"`` using the same raw draws
            and the latest successful :meth:`align_factors` transforms.
            Requires alignment first.  Setting both flags to False
            returns an empty dictionary without sampling.
        seed : int or None, default None
            Unsigned 32-bit seed for this separate sampling stream.
            None selects a fresh random seed.  A fixed seed gives the
            same draws on the same backend, independently of the output
            flags and batching (up to floating-point rounding).
        batch_size : int, default 32
            Maximum number of retained draws processed per batch.
            Requested output arrays are allocated on the host; when only
            aligned scores are requested, raw scores are held one batch
            at a time.

        Returns
        -------
        dict
            Only the requested keys, each a NumPy array of shape
            ``(S, H, n)``.  Raw scores are float32; aligned scores use
            the common dtype of the scores and saved RSP transforms.
            For each s, ``B_aligned_[s] @ eta_sample_aligned[s]`` equals
            ``B_samples_[s] @ eta_sample[s]`` up to numerical error.
            Scores are COLUMN-stored ((H, n) per draw, RSP transform
            applied as R^T); the row-stored equivalent is
            :func:`CSP_JAX._rsp.transform_scores`.
        """
        self._check_fitted()
        from ._factor_scores import sample_factor_scores

        result = getattr(self, "rsp_result_", None)
        transforms = None if result is None else result.transforms
        return sample_factor_scores(
            X, self.B_samples_, self.sigma2_samples_, self.x_offset_,
            return_eta=return_eta,
            return_eta_aligned=return_eta_aligned,
            seed=seed,
            batch_size=batch_size,
            transforms=transforms,
        )

    @property
    def Sigma_mc_(self):
        """(p, p) Monte Carlo posterior mean of Sigma = B B^T + Omega.

        Computed from the raw draws as
        ``(1/S) sum_s (B_s B_s^T + diag(sigma2_s))``, equivalently
        ``(1/S) sum_s B_s B_s^T + diag(sigma2_hat_)``; cached until the
        next successful fit.  Each per-draw covariance is invariant to
        that draw's own orthogonal transform, so this quantity is
        identical before and after :meth:`align_factors` (it is computed
        from the raw ``B_samples_``).  Contrast with the plug-in
        :attr:`CSP_JAX.base.BaseCSP.Sigma_` = B_hat B_hat^T + diag(
        sigma2_hat_), whose value DOES change under alignment (e.g. two
        draws B and -B have mean 0 raw but mean B after sign alignment):
        ``Sigma_mc_`` summarizes the posterior mean of the covariance,
        while ``Sigma_`` is the covariance of the (aligned or raw)
        loading point estimate.  Aliased as :attr:`Sigma_mean_`.
        """
        self._check_fitted()
        cache = self._sigma_cache
        if "mc" not in cache:
            cache["mc"] = montecarlo_sigma_mean(
                self.B_samples_, self.sigma2_samples_
            )
        return cache["mc"]

    @property
    def Sigma_mean_(self):
        """(p, p) Monte Carlo posterior mean of Sigma; alias of Sigma_mc_."""
        return self.Sigma_mc_

    @property
    def Sigma_var_(self):
        """(p, p) Monte Carlo entrywise posterior variance of Sigma.

        ``Sigma_var_[i, j]`` estimates ``Var(Sigma_ij)`` from the draws
        with ddof = 0, where ``Sigma = B B^T + Omega`` and each draw
        contributes ``Sigma_s = B_s B_s^T + diag(sigma2_s)``.  Cached
        until the next successful fit.  The computation costs
        O(S p^2 H); when the full scan exceeds the default compute
        budget, a deterministic equidistant subsample is used and a
        ``UserWarning`` is emitted; the actual number of draws is stored
        in :attr:`Sigma_draws_used_`.  Use :meth:`posterior_sigma_var`
        to control the draw count explicitly, or
        :attr:`Sigma_diag_var_`/:meth:`cov_Sigma` for cheap summaries.
        """
        self._check_fitted()
        cache = self._sigma_cache
        if "var" not in cache:
            S, p, H = self.B_samples_.shape
            n_used = suggest_max_draws(p, H, S)
            if n_used < S:
                warnings.warn(
                    f"Sigma_var_: using {n_used} of {S} draws (deterministic "
                    "equidistant subsample) because the full p x p variance "
                    "scan exceeds the default compute budget; the Monte "
                    "Carlo error is correspondingly larger.  Call "
                    "posterior_sigma_var(max_draws=None) for all draws, or "
                    "use cov_Sigma(pairs)/Sigma_diag_var_ for cheap entry "
                    "summaries.",
                    UserWarning,
                    stacklevel=2,
                )
            var, used = montecarlo_sigma_variance(
                self.B_samples_, self.sigma2_samples_, max_draws=n_used
            )
            cache["var"] = var
            cache.pop("sd", None)
            self.Sigma_draws_used_ = used
        return cache["var"]

    @property
    def Sigma_sd_(self):
        """(p, p) Monte Carlo entrywise posterior standard deviation of Sigma."""
        self._check_fitted()
        cache = self._sigma_cache
        if "sd" not in cache:
            cache["sd"] = np.sqrt(self.Sigma_var_)
        return cache["sd"]

    @property
    def Sigma_diag_var_(self):
        """(p,) Monte Carlo posterior variance of the diagonal of Sigma.

        Cheap (O(S p H)); always uses all draws and is not subject to
        the full-matrix compute budget.
        """
        self._check_fitted()
        cache = self._sigma_cache
        if "diag_var" not in cache:
            cache["diag_var"] = montecarlo_sigma_diag_variance(
                self.B_samples_, self.sigma2_samples_
            )
        return cache["diag_var"]

    def cov_Sigma(self, pairs):
        """Monte Carlo posterior covariance between selected entries of Sigma.

        Parameters
        ----------
        pairs : iterable of ((i, j), (k, l))
            Zero-based entry indices; order within each pair and between
            the two entries does not matter.

        Returns
        -------
        numpy.ndarray, shape (len(pairs),), float64
            Sample covariances with ddof = 0.  Uses all draws; the cost
            is O(S H) per pair, independent of p^2, so this is the
            recommended summary when p is large.  Requires no
            subsampling.
        """
        self._check_fitted()
        return montecarlo_cov_sigma_pairs(
            self.B_samples_, self.sigma2_samples_, pairs
        )

    def posterior_sigma_var(self, max_draws=None, chunk_draws=None):
        """Compute the (p, p) Monte Carlo variance matrix with explicit controls.

        Parameters
        ----------
        max_draws : int or None, default None
            Number of draws to use; ``None`` uses all draws.  A value
            below the number of kept draws selects a deterministic
            equidistant subsample, with correspondingly larger Monte
            Carlo error.
        chunk_draws : int or None, default None
            Draws per processing block; ``None`` selects a block size
            from a memory budget.  Does not change the result.

        Returns
        -------
        numpy.ndarray, shape (p, p), float64
            Entrywise posterior variance with ddof = 0.  Unlike the
            ``Sigma_var_`` property this method does not apply the
            default compute budget (``None`` means all draws; an
            explicit ``max_draws`` selects the deterministic equidistant
            subsample).  With ``max_draws=None`` the result is also
            cached for ``Sigma_var_`` and ``Sigma_draws_used_`` is
            updated to the number of draws used; a subsampled call
            leaves the cached property result and that attribute
            untouched.
        """
        self._check_fitted()
        var, used = montecarlo_sigma_variance(
            self.B_samples_, self.sigma2_samples_,
            max_draws=max_draws, chunk_draws=chunk_draws,
        )
        if max_draws is None:
            self._sigma_cache["var"] = var
            self._sigma_cache.pop("sd", None)
            self.Sigma_draws_used_ = used
        return var

    def ess(self, method="bulk", prob=None, relative=False):
        """Per-entry effective sample size of the kept draws (ArviZ).

        Returns a dict {"B": (p, H), "sigma2": (p,), "H_star": float} of
        ESS values for the kept loading-matrix, noise-variance and
        active-factor-count draws.  ``method`` and ``prob`` are passed
        to :func:`arviz.ess` ("bulk" default; "tail"/"quantile" require
        ``prob``); ``relative=True`` returns ESS per draw.  Requires the
        optional arviz dependency.

        Single-chain ESS only (no R-hat/MCSE); for those, run several
        chains with different seeds and call :func:`arviz.rhat` /
        :func:`arviz.mcse` on the stacked draws -- see
        :mod:`CSP_JAX.ESS`.
        """
        self._check_fitted()
        from .ESS import ess

        return ess(self.B_samples_, self.sigma2_samples_,
                   hs=self.H_star_samples_, method=method,
                   prob=prob, relative=relative)

    def ess_summary(self, method="bulk"):
        """Compact ESS summary (min/median/mean/max per array).

        The minimum is the number to watch; bulk ESS above ~400 per
        entry is the usual rule of thumb for reliable posterior means.
        Requires the optional arviz dependency.
        """
        self._check_fitted()
        from .ESS import ess_summary

        return ess_summary(self.B_samples_, self.sigma2_samples_,
                           hs=self.H_star_samples_, method=method)

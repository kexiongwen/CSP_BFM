"""Shared base class for the CSP estimators.

:class:`BaseCSP` holds the hyperparameters of the cumulative shrinkage
process (CSP) prior and implements the common preprocessing (validation,
float32 coercion, column centering), the common fitted accessors
(``B_hat_``, ``sigma2_hat_``, ``Sigma_``, ``n_factors``), and the RSP
post-processing entry point (:meth:`align_factors`).

The model is the Gaussian factor model of Legramanti, Durante & Dunson
(2020), "Bayesian cumulative shrinkage for infinite factorizations":

    x_i = B eta_i + eps_i,
    eta_i ~ N_H(0, I_H),
    eps_i ~ N_p(0, Omega),  Omega = diag(sigma_1^2, ..., sigma_p^2),

with the CSP spike-and-slab prior on the column variances theta_h of B
(paper eq. 6) and inverse-gamma priors on the sigma_j^2.

The class-layer design mirrors the ``bfm`` package of the L_{1/2} paper
(LH_BFM); unlike that package there is only one inference algorithm here
(the Gibbs sampler, :class:`CSP_JAX.GibbsCSP`), so the base class exists
mainly to keep the estimator interface stable if a second algorithm is
added later.
"""

import numbers
import warnings

import numpy as np

__all__ = ["BaseCSP"]


def _validated_level(level):
    """Validate a credibility level and return it as a Python float.

    Accepts Python and numpy real scalars (an ``isinstance`` check
    against ``float`` alone misses ``np.float32``); booleans, strings
    and other non-numeric inputs are rejected, as are values outside
    (0, 1).
    """
    if (isinstance(level, (bool, np.bool_))
            or not isinstance(level, numbers.Real)):
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    level = float(level)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    return level


class BaseCSP:
    """Shared hyperparameters and preprocessing for the CSP estimators.

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
        IG(a_theta, b_theta) slab of the spike-and-slab prior on
        theta_h.  ``a_theta > 1`` is required by the paper's Theorem 2.
        Paper default a_theta = b_theta = 2.
    theta_inf : float
        Spike variance theta_infty (NOT a standard deviation).  Small
        but strictly positive; paper default 0.05.
    a_sigma, b_sigma : float
        Inverse-gamma hyperparameters of sigma_j^2 ~ IG(a_sigma,
        b_sigma); both must be > 0.  Defaults (1, 1); the paper used
        (1, 0.3).
    center : bool, default True
        Subtract the column means of X before fitting (the model has
        zero mean).  The column means are stored in ``x_offset_``.

    Notes
    -----
    This class only stores parameters and provides shared machinery;
    use :class:`CSP_JAX.GibbsCSP` for fitting.  Detailed validation of
    the algorithm parameters is left to the underlying driver
    (:func:`CSP_JAX.GS_CSP`) so that the wrapper never drifts from
    its rules.
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
    ):
        self.H = H
        self.alpha = alpha
        self.a_theta = a_theta
        self.b_theta = b_theta
        self.theta_inf = theta_inf
        self.a_sigma = a_sigma
        self.b_sigma = b_sigma
        self.center = center

    # ------------------------------------------------------------------
    # fitting (template method)
    # ------------------------------------------------------------------
    # Alignment outputs produced by align_factors(); removed by fit() AFTER
    # the new fit succeeds, so a fit that fails (invalid X, numerical
    # error) keeps the previous state -- raw or aligned -- fully
    # consistent instead of half-cleared.
    _ALIGN_ATTRS = ("rsp_result_", "B_aligned_")
    # Gibbs summarizes aligned draws by updating B_hat_ to the aligned mean.
    _ALIGN_UPDATES_B_HAT = True

    def fit(self, X):
        """Fit the model to the data matrix X and return ``self``.

        X is validated, coerced to float32 and column-centered (unless
        ``center=False``); the concrete algorithm then runs in
        :meth:`_fit_arrays`.  Results of a previous :meth:`align_factors`
        call are discarded only after the new fit succeeds: a fit that
        raises leaves the estimator exactly as it was.
        """
        # Fitting replaces attributes, so preserving their references is
        # enough to restore the previous fit without copying large draws.
        previous_state = self.__dict__.copy()
        try:
            Xc = self._prepare_X(X)
            self.n_samples_, self.n_features_in_ = Xc.shape
            self._fit_arrays(Xc)
            for attr in self._ALIGN_ATTRS:
                if hasattr(self, attr):
                    delattr(self, attr)
        except BaseException:
            # Restore preprocessing and partially assigned results even
            # when a long-running fit is interrupted, then re-raise.
            self.__dict__.clear()
            self.__dict__.update(previous_state)
            raise
        return self

    def _fit_arrays(self, Xc):
        """Run the concrete algorithm on the prepared data matrix.

        Replace fitted attributes rather than modifying existing arrays
        or containers in place, so :meth:`fit` can restore them on failure.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # shared preprocessing
    # ------------------------------------------------------------------
    def _prepare_X(self, X):
        """Validate, center in float64 and return the data as float32.

        The input and ``x_offset_`` stay in float64 while the column
        means are subtracted, so the centering is not rounded at float32
        precision; the centered matrix is cast to float32 afterwards and
        the cast is checked for finiteness (overflow to inf raises).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-dimensional, got shape {tuple(X.shape)}"
            )
        if not np.isfinite(X).all():
            raise ValueError(
                "X must contain only finite values (found NaN or inf)"
            )
        if self.center:
            self.x_offset_ = X.mean(axis=0)
            X = X - self.x_offset_
        else:
            self.x_offset_ = np.zeros(X.shape[1], dtype=np.float64)
        X = X.astype(np.float32)
        if not np.isfinite(X).all():
            raise ValueError(
                "X must be representable in float32 (found non-finite "
                "values after conversion)"
            )
        return X

    # ------------------------------------------------------------------
    # fitted accessors
    # ------------------------------------------------------------------
    def _check_fitted(self):
        if not hasattr(self, "_B_hat"):
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet; call fit(X) first"
            )

    @property
    def B_hat_(self):
        """(p, H) point estimate of the loading matrix.

        Posterior mean over the kept draws.  After :meth:`align_factors`,
        this is the posterior mean of the ALIGNED draws; the mean of the
        raw draws remains available as ``B_samples_.mean(axis=0)``.
        """
        self._check_fitted()
        return self._B_hat

    @property
    def sigma2_hat_(self):
        """(p,) point estimate of the idiosyncratic variances.

        Posterior mean over the kept draws.
        """
        self._check_fitted()
        return self._sigma2_hat

    @property
    def Sigma_(self):
        """(p, p) estimated covariance B_hat B_hat^T + diag(sigma2_hat_)."""
        self._check_fitted()
        return self._B_hat @ self._B_hat.T + np.diag(self._sigma2_hat)

    # ------------------------------------------------------------------
    # factor-dimension selection
    # ------------------------------------------------------------------
    def n_factors(self, level=0.95, *, method="h_star", interval="hpd",
                  use_aligned=False):
        """Estimate the number of active factor columns.

        Two rules are available:

        * ``method="h_star"`` (default): the CSP dimension posterior.
          The number of active factors H* is sampled every sweep (columns
          with z_h > h); this rule returns the posterior median of the
          kept H* draws.  Use :meth:`h_star_summary` for the posterior
          mean and credible interval.  This is the paper's headline
          dimension-selection mechanism and is computed from the raw
          chain; it does not depend on alignment (the varimax rotation
          mixes spike columns into active ones, so H* is only meaningful
          in the prior's coordinate system).  With an even number of
          draws the median can be a half-integer (e.g. 4.5); it is then
          FLOORED to the returned int (4) by ``int()`` — use
          :meth:`h_star_summary` when the exact median matters.
        * ``method="credible"``: the entry-wise marginal credible-interval
          rule (as in the LH_BFM package): an entry B_jk counts as zero
          when 0 lies inside its smallest (highest-density) marginal
          credible interval at credibility ``level``, and a column is
          active when at least one entry is nonzero.  Builds the
          intervals empirically from the kept draws (``B_samples_``, or
          ``B_aligned_`` with ``use_aligned=True``).

        Parameters
        ----------
        level : float, default 0.95
            Credibility level in (0, 1).  Only used by
            ``method="credible"`` (the marginal entry intervals); the
            "h_star" rule reports the posterior median, which does not
            depend on ``level`` — use :meth:`h_star_summary` (which does
            take ``level``) for the credible interval of H*.
        method : {"h_star", "credible"}, default "h_star"
        interval : {"hpd", "quantile"}, default "hpd"
            Interval construction for the "credible" rule: "hpd" is the
            shortest interval covering ``level`` of the draws (robust to
            multimodal marginals, e.g. residual sign switching in the
            chain); "quantile" is the equal-tailed interval.
        use_aligned : bool, default False
            "credible" rule only: use ``B_aligned_`` instead of the raw
            ``B_samples_`` (requires a successful :meth:`align_factors`
            first).  Note that the marginal intervals are invariant to
            sign flips and column permutations but NOT to the varimax
            rotation, so the two variants answer slightly different
            questions.

        Returns
        -------
        int
        """
        self._check_fitted()
        level = _validated_level(level)
        if method == "h_star":
            draws = getattr(self, "H_star_samples_", None)
            if draws is None:
                raise RuntimeError(
                    f"{type(self).__name__} has no H* draws; the h_star "
                    "rule requires a fitted Gibbs chain"
                )
            return int(np.median(np.asarray(draws)))
        if method == "credible":
            return int(
                self.active_entries(level=level, interval=interval,
                                    use_aligned=use_aligned)
                .any(axis=0).sum()
            )
        raise ValueError(f"method must be 'h_star' or 'credible', got {method!r}")

    def h_star_summary(self, level=0.95):
        """Posterior summary of the active-factor count H*.

        Parameters
        ----------
        level : float, default 0.95
            Credibility level in (0, 1) for the equal-tailed credible
            interval.

        Returns
        -------
        dict
            {"mean", "median", "ci", "n_draws"} where ``ci`` is the
            equal-tailed (level) credible interval ``(lo, hi)`` of the
            kept H* draws.  The posterior of H* is typically highly
            concentrated at the true dimension (paper, Section 4), in
            which case the interval collapses to a single integer.
        """
        self._check_fitted()
        level = _validated_level(level)
        draws = getattr(self, "H_star_samples_", None)
        if draws is None:
            raise RuntimeError(
                f"{type(self).__name__} has no H* draws; the h_star "
                "summary requires a fitted Gibbs chain"
            )
        hs = np.asarray(draws, dtype=np.float64)
        a = 1.0 - level
        lo, hi = np.quantile(hs, [a / 2.0, 1.0 - a / 2.0])
        return {
            "mean": float(hs.mean()),
            "median": float(np.median(hs)),
            "ci": (float(lo), float(hi)),
            "n_draws": int(hs.size),
        }

    def active_entries(self, level=0.95, *, interval="hpd", use_aligned=False):
        """(p, H) bool mask of active loading entries.

        Entry (j, h) is active iff 0 lies outside its smallest marginal
        credible interval at credibility ``level`` (see
        :meth:`n_factors` with ``method="credible"``).
        """
        self._check_fitted()
        level = _validated_level(level)
        if interval not in ("hpd", "quantile"):
            raise ValueError(
                f"interval must be 'hpd' or 'quantile', got {interval!r}"
            )
        return self._credible_active(level, interval, use_aligned)

    def _credible_draws(self, use_aligned):
        """The draw array to build empirical intervals from."""
        self._check_fitted()
        draws = getattr(self, "B_samples_", None)
        if draws is None:
            raise RuntimeError(
                f"{type(self).__name__} has no posterior draws for the "
                "credible-interval rule"
            )
        if use_aligned:
            aligned = getattr(self, "B_aligned_", None)
            if aligned is None:
                raise RuntimeError(
                    "use_aligned=True requires a successful "
                    "align_factors() call first"
                )
            return aligned
        return draws

    def _credible_active(self, level, interval, use_aligned):
        """Empirical marginal intervals over the draws."""
        draws = np.asarray(self._credible_draws(use_aligned), dtype=np.float64)
        S = draws.shape[0]
        if S < 2:
            raise ValueError(
                f"the credible rule needs at least 2 draws, got S = {S}"
            )
        if interval == "quantile":
            a = 1.0 - level
            lo, hi = np.quantile(draws, [a / 2.0, 1.0 - a / 2.0], axis=0)
        else:
            # HPD: shortest window covering m = ceil(level * S) sorted
            # draws (a window of m consecutive points is s[i+m-1] - s[i])
            s = np.sort(draws, axis=0)
            m = int(np.ceil(level * S))
            if m >= S:
                lo, hi = s[0], s[-1]
            else:
                w = s[m - 1:] - s[: S - m + 1]             # (S - m + 1, p, H)
                i = np.argmin(w, axis=0)                   # (p, H)
                lo = np.take_along_axis(s, i[None], axis=0)[0]
                hi = np.take_along_axis(s, (i + m - 1)[None], axis=0)[0]
        return (lo > 0) | (hi < 0)

    # ------------------------------------------------------------------
    # factor switching (RSP post-processing)
    # ------------------------------------------------------------------
    def align_factors(
        self,
        *,
        rotate=True,
        sp_path="elim",
        init="auto",
        reference=None,
        threshold=1e-6,
        max_iter=100,
        polar="qdwh",
        assignment_backend="auto",
        precision="float32",
        verbose=False,
    ):
        """Align the kept Gibbs draws using RSP.

        Runs the Rotation-Sign-Permutation algorithm implemented in
        :mod:`CSP_JAX._rsp` (Papastamoulis & Ntzoufras 2022; Rey-Saez &
        Revuelta 2026) on the retained loading draws and updates
        ``B_hat_`` to their aligned mean.  Consequently the plug-in
        ``Sigma_`` can change, while ``Sigma_mc_`` and the raw draws used
        by ``ess()`` remain unchanged.  The H* chain
        (``H_star_samples_``) is also unaffected: it is defined in the
        prior's coordinate system and is only meaningful before
        alignment.

        Repeated calls always start from the original draws and replace
        the previous alignment outputs.  A successful new ``fit``
        discards them; a failed fit or alignment preserves them.

        Parameters
        ----------
        rotate, sp_path, init, reference, threshold, max_iter, polar,
        assignment_backend, precision, verbose
            Passed through to :func:`CSP_JAX._rsp.rsp_exact` (see its
            docstring for details).
            ``return_transforms`` is forced on for consistent score
            transforms in :meth:`GibbsCSP.sample_factors`.
            ``assignment_backend="auto"`` needs optax for H > 7 (install
            with ``pip install -e ".[rsp]"``).  The "scipy" backend
            needs scipy, which is NOT part of the ``rsp`` extra; install
            it separately (``pip install scipy``).

            ``precision="float64"`` turns the PROCESS-WIDE JAX x64 mode
            ON inside the RSP code
            (``jax.config.update("jax_enable_x64", True)``) and does NOT
            restore it afterwards: every JAX array created later in this
            process defaults to float64.  Keep the default "float32"
            unless the extra fidelity is required, or run such an
            alignment in a subprocess.

        Returns
        -------
        self
            New attributes:

            - ``rsp_result_``: the full :class:`CSP_JAX._rsp.RSPResult`
              (aligned draws, signs, permutations, loss histories,
              diagnostics, per-draw transforms);
            - ``B_aligned_``: (S, p, H) aligned loadings.

            Factor scores are returned on request by
            :meth:`GibbsCSP.sample_factors`.

        Raises
        ------
        RuntimeError
            If the estimator is unfitted or has no supported alignment input.
        FloatingPointError
            If RSP ended with status_code 4: NO estimator attribute
            changes, the previous estimates (raw or from a previous
            successful alignment) are kept, and the failed run is
            attached to the exception as ``rsp_result`` for diagnostics.
        UserWarning
            If the run ended with status_code 3 (the exact objective
              increased beyond the numerical tolerance); the finite
              aligned results are still accepted.
        """
        draws = self._alignment_loadings()
        try:
            from ._rsp import rsp_exact
        except ImportError as exc:  # pragma: no cover - internal import
            raise ImportError(
                "the RSP module CSP_JAX._rsp failed to import: "
                f"{exc}"
            ) from exc

        try:
            result = rsp_exact(
                draws,
                rotate=rotate,
                sp_path=sp_path,
                init=init,
                reference=reference,
                threshold=threshold,
                max_iter=max_iter,
                polar=polar,
                assignment_backend=assignment_backend,
                precision=precision,
                return_transforms=True,
                verbose=verbose,
            )
        except ImportError as exc:
            raise ImportError(
                f"the RSP alignment backend is missing a dependency "
                f"({exc}); install optax with: pip install -e \".[rsp]\" "
                "(optax is needed for H > 7); the \"scipy\" assignment "
                "backend needs scipy instead, which is not part of the "
                "rsp extra: pip install scipy"
            ) from exc

        status = int(result.status_code)
        if status == 4:
            # Zero estimator state change on failure: the failed run is
            # attached to the exception (exc.rsp_result) for diagnostics
            # and NO attribute is written.  A failed run must not
            # create alignment outputs or overwrite a successful one.
            exc = FloatingPointError(
                "rsp_exact ended with status_code 4 (non-finite "
                "objective, intermediate, or returned state); no "
                "estimator state was changed and the previous estimates "
                "are kept (the failed run is attached as exc.rsp_result)"
            )
            exc.rsp_result = result
            raise exc
        # Build and validate every output before publishing any of them.
        updates = self._alignment_outputs(result)
        if status == 3:
            warnings.warn(
                "rsp_exact ended with status_code 3 (the exact alignment "
                "objective increased beyond the numerical tolerance); "
                "accepting the finite aligned results -- inspect "
                "rsp_result_.diagnostics",
                UserWarning,
                stacklevel=2,
            )

        self.__dict__.update(updates)
        return self

    def _alignment_loadings(self):
        """Return original loadings with a leading alignment-sample axis."""
        self._check_fitted()
        draws = getattr(self, "B_samples_", None)
        if draws is None:
            raise RuntimeError(f"{type(self).__name__} has no alignment input")
        return draws

    def _alignment_outputs(self, result):
        """Prepare alignment attributes without changing estimator state."""
        outputs = {
            "rsp_result_": result,
            "B_aligned_": np.asarray(result.aligned_loadings),
        }
        if self._ALIGN_UPDATES_B_HAT:
            outputs["_B_hat"] = np.asarray(result.mean_loadings)
        return outputs

"""
GS_CSP.py  (JAX, single precision)

Gibbs sampler driver for the Bayesian factor model with the cumulative
shrinkage process prior (Legramanti, Durante & Dunson, 2020, "Bayesian
cumulative shrinkage for infinite factorizations", Sections 3.1-3.2).

Model:
    x_i = B eta_i + eps_i,
    eta_i ~ N_H(0, I_H),
    eps_i ~ N_p(0, Omega),  Omega = diag(sigma_1^2, ..., sigma_p^2).

Prior (eq. 6 of the paper):
    (lambda_jh | theta_h) ~ N(0, theta_h),
    (theta_h | pi_h) ~ (1 - pi_h) IG(a_theta, b_theta) + pi_h delta_theta_inf,
    pi_h = sum_{l<=h} omega_l,  omega_l = v_l prod_{m<l} (1 - v_m),
    v_l ~ Be(1, alpha) iid,  v_H = 1 (truncation),  sigma_j^2 ~ IG(a_sigma, b_sigma).

Orchestrates one full sweep of the sampler by calling, in order
(each step is an exact full conditional -- Algorithm 1 reordered into
a fixed cyclic scan, which leaves the invariant distribution
unchanged):

    1. eta_sample    : eta_i | B, Omega, x_i              (Alg. 1, step 3)
    2. shrinkage     : z_h, v_l, theta_h | B, V           (Alg. 1, steps 4-7)
    3. sigma2_sample : sigma_j^2 | X_{.j}, B_j, eta       (Alg. 1, step 2)
    4. B_sample      : B_j | X_{.j}, eta, theta, sigma_j^2 (Alg. 1, step 1)

Chain structure: the Markov state carried between sweeps is
(B, sigma2, V) -- the CSP stick V is genuine sampler state, unlike
the L_{1/2} shrinkage block, which regenerates fully from B.  The
eta / z / theta blocks are marginal-then-conditional draws given the
state and are not carried.  z is integrated out of the state via eq.
(8) and theta is a deterministic function of (z, B).  Because z is
drawn against the PRE-update B while the kept B is the post-update
draw, the recorded H* chain and B chain are offset by half a sweep:
each marginal law is the exact posterior, but (B[s], H*[s]) pairs are
not synchronous joint posterior draws -- summarize them separately.

This is Algorithm 1 of the paper (fixed truncation H).  Algorithm 2
(adaptive tuning of H) is not ported; with the paper's default alpha
= 5, redundant columns are spiked automatically and a conservative H
is typically enough (see Section 3.3 and Table 1 of the paper).

JAX engineering (ported from GS_LH_JAX):
* Single precision (float32) throughout: X is coerced on entry and
  every module derives its dtype from it.
* Randomness is explicit: `seed` seeds one jax.random.PRNGKey split
  into one key per sweep up front (plus an initialization key and one
  extra child for the final partial sweep); the same seed reproduces
  the same chain bit-exactly.
* The chain is executed as jax.lax.scan over the sweep keys, in
  chunks of at most _CHUNK sweeps (bounded by a device-memory cap):
  each chunk is ONE compiled XLA While loop with no Python dispatch
  or host synchronization inside; between chunks the draws are copied
  to the host (progress reports, finiteness audit).  Two compiled
  runners are used -- burn-in (advances only) and collecting -- and
  are compiled ahead of time via lower().compile(), so no compilation
  is counted in runtime_sec.  Compilation is cached per distinct
  chunk length across calls.
* Posterior draws are stored on the host and returned as numpy
  arrays.  Every chunk is audited for finiteness: all sampled systems
  are I + PSD by construction, so a non-finite state signals a
  numerical anomaly outside the guarded range and fails loudly.

Implementation: JAX-based, functional style.
"""

import os
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from .B_sample import B_sample
from .eta_sample import eta_sample
from .shrinkage import make_log_t_constant, shrinkage
from .Sigma2_sample import sigma2_sample

__all__ = ["GS_CSP", "Initialization"]

_F32 = jnp.float32
_EPS = jnp.finfo(_F32).eps

# Sweeps per lax.scan chunk (see _chunk_sizes): bounds device-side
# draw storage per chunk and gives one progress report / finiteness
# audit per chunk.
_CHUNK = 500
# Cap on device-side collected draws per chunk, in bytes.
_CHUNK_BYTES = 256 * 2 ** 20


def Initialization(X: jax.Array, r: int) -> tuple[jax.Array, float]:
    r"""Principal-factor initialization for (B, sigma2).

    Eigen-decomposition of the sample covariance (X is assumed
    column-centered, as the model has zero mean): with
    M = X.T / sqrt(n - 1), M M^T = X^T X / (n - 1) = Sigma_hat, and
    eigenvalues s_i^2 of Sigma_hat (descending),

        sigma2_hat = mean of the trailing eigenvalues {s_i^2}_{i > r},
        B0         = U_r diag( sqrt(s_1^2 - sigma2_hat, ...) ),

    so that B0 B0^T is the rank-r PSD approximation of
    Sigma_hat - sigma2_hat I (the factor model decomposes
    Sigma = B B^T + Omega at initialization).

    Implementation notes:
    * the top-r eigenpairs come from one jnp.linalg.eigh of the p x p
      covariance -- deterministic and exact, and this runs ONCE per
      chain;
    * the trailing sum uses the exact trace identity
      sum_{i > r} s_i^2 = tr(Sigma_hat) - sum_{i <= r} s_i^2, with
      denominator min(n - 1, p) - r: column-centered X has rank at
      most min(n - 1, p), so this is exactly the number of trailing
      eigenvalues that can be nonzero;
    * s_i^2 - sigma2_hat is floored at 0: top-r components below the
      noise floor initialize to a zero column instead of NaN.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Column-centered data matrix.
    r : int
        Number of leading factors, 1 <= r <= min(n, p) - 2.  For the
        CSP model take r <= H - 1: at most H - 1 factors can be
        active, as the Hth column is spiked by construction.

    Returns
    -------
    B0 : jnp.ndarray, shape (p, r), float32
        Initial loading matrix.
    sigma2_hat : float
        Noise-floor variance estimate (broadcast to (p,) by the
        caller for the initial Omega diagonal).
    """
    X = jnp.asarray(X, dtype=_F32)
    n, p = X.shape
    if not 1 <= r <= min(n, p) - 2:
        raise ValueError(
            f"r must be in [1, min(n, p) - 2] = [1, {min(n, p) - 2}], got {r}"
        )

    M = X.T / jnp.sqrt(jnp.asarray(n - 1, dtype=_F32))  # (p, n)
    C = M @ M.T                                          # sample cov (p, p)
    evals, evecs = jnp.linalg.eigh(C)                    # ascending order

    # top-r eigenpairs, descending
    s2 = evals[::-1][:r]
    U = evecs[:, ::-1][:, :r]

    # mean trailing eigenvalue (trace identity for the discarded
    # spectrum; exactly min(n - 1, p) - r trailing eigenvalues can be
    # nonzero for column-centered X)
    tail = jnp.trace(C) - s2.sum()
    sigma2_hat = jnp.maximum(tail / (min(n - 1, p) - r), 0.0)

    B0 = U * jnp.sqrt(jnp.maximum(s2 - sigma2_hat, 0.0))[None, :]
    # The eigen-computation above squares X in float32 with no scaling:
    # finite input with |x| ~ 1e19 makes x^2 unrepresentable and yields a
    # non-finite s2 / sigma2_hat (rarely, eigh itself can also fail to
    # converge).  Surface that HERE as an actionable ValueError instead of
    # letting the non-finite B0 trip the driver's per-chunk audit, whose
    # message targets internal numerical anomalies.
    if not (bool(jnp.isfinite(s2).all()) and bool(jnp.isfinite(sigma2_hat))):
        raise ValueError(
            "non-finite principal-factor initialization: the float32 sample "
            "covariance of X overflowed (|X| too large; x^2 must stay within "
            "float32 range) or the eigensolver failed.  Rescale the data "
            "(the factor model is scale-covariant: B scales linearly, "
            "sigma2 quadratically) and retry"
        )
    return B0, float(sigma2_hat)


def _sweep(state, key, X, alpha, a_theta, b_theta, theta_inf, constant,
           a_sigma, b_sigma):
    """One full Gibbs sweep: eta -> shrinkage -> sigma2 -> B.

    Scalar hyperparameters are traced, so changing them does not
    trigger recompilation.  This stays UN-jitted: it is compiled
    inline into the scan runners below.
    """
    B, sigma2, V = state
    k_eta, k_sh, k_s2, k_B = jax.random.split(key, 4)
    # 1. latent factors
    eta = eta_sample(k_eta, X, B, sigma2)
    # 2. CSP shrinkage: z, v, theta (D = sqrt(theta_h) prior stds)
    D, V, Z = shrinkage(k_sh, B, V, alpha, a_theta, b_theta, theta_inf,
                        constant)
    # 3. idiosyncratic variances
    sigma2 = sigma2_sample(k_s2, X, B, eta, a_sigma, b_sigma)
    # 4. loading matrix
    B = B_sample(k_B, X, D, eta, sigma2)
    # active factors; z was drawn given the pre-update B, so H* lags the
    # returned B by half a sweep (marginal law is exact -- module docs)
    h_star = (Z > jnp.arange(Z.shape[0])).sum()
    return (B, sigma2, V), (B, sigma2, h_star)


def _run_burn_impl(state, keys, X, alpha, a_theta, b_theta, theta_inf,
                   constant, a_sigma, b_sigma):
    """Advance the chain over a chunk of burn-in keys, collecting
    nothing.  Returns only the final state."""
    def f(state, k):
        state, _ = _sweep(state, k, X, alpha, a_theta, b_theta, theta_inf,
                          constant, a_sigma, b_sigma)
        return state, None

    state, _ = jax.lax.scan(f, state, keys)
    return state


def _run_keep_impl(state, keys, X, alpha, a_theta, b_theta, theta_inf,
                   constant, a_sigma, b_sigma):
    """Advance the chain over a chunk of keys, collecting
    (B, sigma2, H_star) of every sweep."""
    def f(state, k):
        state, draws = _sweep(state, k, X, alpha, a_theta, b_theta,
                              theta_inf, constant, a_sigma, b_sigma)
        return state, draws

    return jax.lax.scan(f, state, keys)


_run_burn = jax.jit(_run_burn_impl)
_run_keep = jax.jit(_run_keep_impl)


def _chunk_sizes(total: int, p: int, H: int) -> list[int]:
    """Split `total` sweeps into chunk lengths bounded by _CHUNK and by
    the per-chunk device memory cap _CHUNK_BYTES (the collecting
    runner stores p*H + p floats and one int per sweep)."""
    per_sweep = (p * H + p) * 4 + 4
    chunk = max(1, min(_CHUNK, _CHUNK_BYTES // max(per_sweep, 1)))
    sizes, left = [], total
    while left > 0:
        sizes.append(min(chunk, left))
        left -= sizes[-1]
    return sizes


def _audit_host(arrs: tuple[np.ndarray, ...], upto: int) -> None:
    """Finiteness audit of one host-side chunk.

    All sampled systems are I + PSD by construction and the stick V is
    clamped to (0, 1) by shrinkage, so a non-finite state signals a
    numerical anomaly OUTSIDE the guarded range; fail loudly rather
    than silently contaminating the chain.
    """
    if not all(np.isfinite(a).all() for a in arrs):
        raise FloatingPointError(
            f"non-finite state (B or sigma2) detected in the chunk ending "
            f"at sweep {upto}; the chain is numerically contaminated and "
            "must be discarded. This should not happen (all systems are "
            "I + PSD by construction) -- please report the data and "
            "hyperparameters."
        )


def GS_CSP(
    X: jax.Array,
    H: int,
    alpha: float = 4.0,
    a_theta: float = 2.0,
    b_theta: float = 2.0,
    theta_inf: float = 0.05,
    a_sigma: float = 1.0,
    b_sigma: float = 1.0,
    n_iter: int = 2000,
    burn_in: int = 1000,
    thin: int = 1,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    r"""Gibbs sampler for the Bayesian factor model with the cumulative
    shrinkage process prior (JAX, float32).

    Runs n_iter sweeps of Algorithm 1 on the model truncated at H
    columns (v_H = 1), each sweep updating (all exact full
    conditionals):

        1. eta    ~ pi(eta | B, sigma2, X)                [eta_sample]
        2. z, v, theta ~ pi(z, v, theta | B, V)           [shrinkage;
                    V is chain state; D = sqrt(theta_h)]
        3. sigma2 ~ pi(sigma2 | B, eta, X)                [sigma2_sample]
        4. B      ~ pi(B | eta, sigma2, D, X)             [B_sample]

    Parameters
    ----------
    X : array-like, shape (n, p)
        Column-centered data matrix, coerced to float32.  Must be
        finite: NaN and inf entries (including float64 values that
        overflow float32) raise ValueError.  Extreme but finite scales
        also fail loudly: the float32 sample covariance in the
        initialization requires x^2 representable (|X| beyond roughly
        1e18 overflows); a non-finite initialization raises ValueError
        asking for a rescale (see Initialization).
    H : int
        Truncation index of the loading matrix (number of columns);
        integer >= 2.  At most H - 1 factors can be active, since the
        Hth column is spiked by construction; the paper's conservative
        choice is H = p + 1, but substantially fewer columns are
        typically enough (Section 3.3); H > p + 1 issues a
        RuntimeWarning.  E(pi_H) via eq. (2) of the paper can be
        monitored to check that H is large enough.
    alpha : float, default 4.0
        CSP concentration parameter (must be > 0): the prior expected
        number of active factors in the infinite process, before
        truncation.  The paper used 5.
    a_theta, b_theta : float
        IG(a_theta, b_theta) slab of the spike-and-slab prior on
        theta_h.  a_theta > 1 is required by the paper's Theorem 2
        (finite E(theta_h), full prior support for Omega); violating
        it issues a RuntimeWarning but the sampler on the H-truncated
        model remains valid.  Paper default a_theta = b_theta = 2.
    theta_inf : float
        Spike variance theta_infty (NOT a standard deviation): the
        marginal of the loadings on a spiked column is
        N(0, theta_inf).  Small but strictly positive; paper default
        0.05.
    a_sigma, b_sigma : float
        Inverse-gamma hyperparameters of sigma_j^2; both > 0.
        Defaults (1, 1); the paper used (1, 0.3)
        (Bhattacharya & Dunson, 2011).
    n_iter, burn_in, thin : int
        Total sweeps, discarded initial sweeps, and keep-every-thin.
    seed : int or None
        Seed for jax.random.PRNGKey.  None draws a random seed.  The
        same seed (on the same backend) reproduces the same chain
        bit-exactly, including eta_last / D_last / V_last.
    verbose : bool
        Print progress about every 10% of the sweeps (reports are
        emitted at chunk boundaries, so small runs print less often).

    Returns
    -------
    dict with keys
        "B"       : (S, p, H) kept loading matrices, numpy float32 on
                    the host, S = (n_iter - burn_in) // thin;
        "sigma2"  : (S, p) kept variance vectors, likewise;
        "H_star"  : (S,) number of active factors per kept draw
                    (sum of z_h > h), drawn against the pre-update B of
                    the sweep while the kept B is the post-update draw
                    (half a sweep apart: use marginal summaries only --
                    see the chain-structure note in the module
                    docstring); the posterior mean of this chain
                    estimates E(H* | y), the model dimension;
        "eta_last": (H, n) latent factors drawn from the FINAL state
                    (one extra partial sweep), for diagnostics or warm
                    starts;
        "D_last"  : (p, H) prior stds sqrt(theta_h) from the final
                    state;
        "V_last"  : (H,) final stick (placeholder 0 last);
        "H_star_last" : int, active factors at the final state;
        "seed"    : the seed actually used;
        "runtime_sec" : wall-clock seconds of the scan loop (excludes
                    the ahead-of-time JIT compilation).

    Initialization: principal-factor initialization (see
    Initialization) with r = min(H - 1, min(n, p) - 2) leading factors;
    remaining columns of B start at zero, every sigma_j^2 starts at
    the noise-floor estimate, and the stick starts at
    v_l ~ Be(1, alpha) iid (l = 1, ..., H - 1), matching the prior.

    Numerical safety: all sampled systems are I + PSD by construction;
    the stick is clamped to (eps, 1 - eps) inside shrinkage; every
    chunk is audited for finiteness on the host and a non-finite
    state raises FloatingPointError.

    Memory note: the B chain costs S*p*H*4 bytes on the host -- use
    thinning for large p.
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    key = jax.random.PRNGKey(seed)

    # single precision throughout: every module derives dtype from X
    X = jnp.asarray(X, dtype=_F32)
    # reject non-finite data up front (also catches float64 entries that
    # overflow to +-inf under the float32 coercion): a contaminated chain
    # would otherwise only surface at the per-chunk finiteness audit with
    # a message meant for internal numerical anomalies
    if not bool(jnp.isfinite(X).all()):
        raise ValueError(
            "X contains NaN or infinite values; the factor model requires "
            "finite data"
        )

    # ---- hyperparameter validation ----------------------------------------
    # integral floats (H = 10.0) are coerced; fractional, inf or NaN H
    # are rejected (is_integer() is False for non-finite values, so the
    # int() conversion below can never raise OverflowError).
    if not float(H).is_integer() or H < 2:
        raise ValueError(f"H must be an integer >= 2, got {H}")
    H = int(H)
    for name, val in (("n_iter", n_iter), ("burn_in", burn_in),
                      ("thin", thin)):
        if not float(val).is_integer():
            raise ValueError(f"{name} must be an integer, got {val}")
    n_iter, burn_in, thin = int(n_iter), int(burn_in), int(thin)
    for name, val in (("alpha", alpha), ("a_theta", a_theta),
                      ("b_theta", b_theta), ("theta_inf", theta_inf),
                      ("a_sigma", a_sigma), ("b_sigma", b_sigma)):
        if not np.isfinite(val) or not val > 0:  # rejects NaN and +-inf
            raise ValueError(
                f"{name} must be finite and > 0, got {name} = {val}")
    if a_theta <= 1.0:
        warnings.warn(
            f"a_theta = {a_theta} violates the condition a_theta > 1 of "
            "the paper's Theorem 2 (E(theta_h) < inf, full prior support "
            "for Omega): the asymptotic theory need not apply, although "
            "the sampler on the H-truncated model remains valid",
            RuntimeWarning,
        )

    if not 0 <= burn_in < n_iter:
        raise ValueError(f"need 0 <= burn_in < n_iter, got {burn_in}/{n_iter}")
    if thin < 1:
        raise ValueError(f"thin must be >= 1, got {thin}")
    if (n_iter - burn_in) // thin == 0:
        raise ValueError(
            f"no draws would be kept: (n_iter - burn_in) // thin = "
            f"({n_iter} - {burn_in}) // {thin} = 0; reduce thin or burn_in"
        )

    n, p = X.shape
    if min(n, p) < 3:
        raise ValueError(
            f"need min(n, p) >= 3 for initialization, got (n, p) = ({n}, {p})"
        )
    if H > p + 1:
        warnings.warn(
            f"H = {H} exceeds p + 1 = {p + 1}: at most H - 1 factors can be "
            "active and the paper (Section 3.3) recommends a conservative "
            "H <= p + 1; larger truncations add spike columns that cost "
            "computation without adding representational power",
            RuntimeWarning,
            stacklevel=2,
        )

    # host-side float64 t log-normalizer for the z-step (see shrinkage)
    constant = make_log_t_constant(p, a_theta, b_theta)

    # one key per sweep, split up front: same seed -> same chain.  Two
    # extra children are reserved: the stick initialization and the
    # eta_last / D_last partial sweep at the end of the run; children
    # of one split are distinct by construction, so they never collide
    # with in-chain sweep keys.
    keys_all = jax.random.split(key, n_iter + 2)
    keys, key_init, key_extra = (keys_all[:n_iter], keys_all[n_iter],
                                 keys_all[n_iter + 1])

    # ---- initialize ---------------------------------------------------------
    # at most H - 1 factors can be active under the truncated CSP
    r = min(H - 1, min(n, p) - 2)
    B0, s2_0 = Initialization(X, r)
    B = jnp.zeros((p, H), dtype=_F32).at[:, :r].set(B0)
    sigma2 = jnp.full((p,), s2_0, dtype=_F32)
    # stick: v_l ~ Be(1, alpha) iid (l = 1..H-1), placeholder 0 last
    v0 = jax.random.beta(key_init, 1.0, alpha, shape=(H - 1,), dtype=_F32)
    V = jnp.concatenate([jnp.clip(v0, _EPS, 1.0 - _EPS),
                         jnp.zeros((1,), _F32)])

    n_keep = (n_iter - burn_in) // thin
    args = (X, alpha, a_theta, b_theta, theta_inf, constant, a_sigma, b_sigma)
    state = (B, sigma2, V)

    # ---- ahead-of-time compilation (not timed) ------------------------------
    # One compilation per DISTINCT chunk length per runner: lower() traces
    # only shapes, so WHICH key slice is fed to it is irrelevant -- iterate
    # the distinct lengths in sorted order and point each at the leading
    # slice of its phase (compile() warms the jit cache for that length;
    # the run loops below then hit it with their real slices).
    burn_sizes = _chunk_sizes(burn_in, p, H)
    keep_sizes = _chunk_sizes(n_iter - burn_in, p, H)
    for c in sorted(set(burn_sizes)):
        _run_burn.lower(state, keys[:c], *args).compile()
    for c in sorted(set(keep_sizes)):
        _run_keep.lower(state, keys[burn_in:burn_in + c], *args).compile()

    # ---- run ------------------------------------------------------------------
    B_keep = np.empty((n_keep, p, H), dtype=np.float32)   # stored on host
    s2_keep = np.empty((n_keep, p), dtype=np.float32)
    hs_keep = np.empty((n_keep,), dtype=np.int32)

    t_start = time.perf_counter()
    report = max(1, n_iter // 10)
    last_report = 0

    # burn-in: advance without collecting (audit the carried state per chunk)
    start = 0
    for c in burn_sizes:
        state = _run_burn(state, keys[start:start + c], *args)
        start += c
        _audit_host((np.asarray(state[0]), np.asarray(state[1])), start)
        if verbose and (start - last_report >= report or start == n_iter):
            print(f"GS_CSP: sweep {start}/{n_iter}", flush=True)
            last_report = start

    # kept phase: collect every sweep per chunk, thin on the host
    kept = 0
    for c in keep_sizes:
        state, (Bc, s2c, hsc) = _run_keep(state, keys[start:start + c], *args)
        Bc, s2c = np.asarray(Bc), np.asarray(s2c)
        _audit_host((Bc, s2c), start + c)
        # global sweep indices (after burn-in) covered by this chunk
        idx = np.arange(start - burn_in, start - burn_in + c)
        sel = (idx + 1) % thin == 0                     # keep-every-thin
        take = idx[sel]
        B_keep[kept:kept + take.size] = Bc[sel]
        s2_keep[kept:kept + take.size] = s2c[sel]
        hs_keep[kept:kept + take.size] = np.asarray(hsc)[sel]
        kept += take.size
        start += c
        if verbose and (start - last_report >= report or start == n_iter):
            print(f"GS_CSP: sweep {start}/{n_iter}", flush=True)
            last_report = start

    runtime = time.perf_counter() - t_start
    if kept != n_keep:
        raise RuntimeError(
            f"kept {kept} draws but expected {n_keep}: the thinning "
            "bookkeeping is inconsistent -- please report this bug"
        )

    # eta_last / D_last / V_last: one extra partial sweep from the final
    # state (steps 1-2 only; same marginal law as the last in-chain
    # draws).  key_extra is the reserved child of the up-front split, so
    # these draws are independent of every sweep key and the whole run
    # stays reproducible from `seed` alone.
    B_f, s2_f, V_f = state
    k_eta, k_sh = jax.random.split(key_extra)
    eta_last = eta_sample(k_eta, X, B_f, s2_f)
    D_last, V_last, Z_last = shrinkage(k_sh, B_f, V_f, alpha, a_theta,
                                       b_theta, theta_inf, constant)
    h_star_last = int((np.asarray(Z_last) > np.arange(H)).sum())

    return {
        "B": B_keep,
        "sigma2": s2_keep,
        "H_star": hs_keep,
        "eta_last": np.asarray(eta_last),
        "D_last": np.asarray(D_last),
        "V_last": np.asarray(V_last),
        "H_star_last": h_star_last,
        "seed": seed,
        "runtime_sec": runtime,
    }

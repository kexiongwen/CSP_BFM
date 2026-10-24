"""
shrinkage.py  (JAX, single precision)

Shrinkage-parameter steps of the Gibbs sampler for the Bayesian factor
model with the cumulative shrinkage process (CSP) prior
(Legramanti, Durante & Dunson, 2020, "Bayesian cumulative shrinkage for
infinite factorizations", Section 3.2, Algorithm 1, steps 4-7).

Given the loading matrix B (p x H) and the stick-breaking weights
encoded by V, one call performs, in order:

    z_h | B, omega ~ Cat[ pr(z_h = l) proportional to
                          omega_l * N_p(B_.h; 0, theta_inf I)   l <= h
                          omega_l * t_{2a}(B_.h; 0, (b/a) I)    l > h ]
                                                                    (eq. 8)
    v_l | z ~ Be(1 + sum_h 1(z_h = l), alpha + sum_h 1(z_h > l)),
                                                              l = 1..H-1
    theta_h | z, B = theta_inf                         if z_h <= h  (spike)
                 ~ IG(a + p/2, b + 0.5 sum_j B_jh^2)    if z_h >  h  (slab)

Conventions (shared with the GS_LH_JAX reference pipeline):
    B      : (p, H) loading matrix; H is the truncation index of the
             paper (v_H = 1, at most H-1 active factors).
    V      : (H,) chain STATE carried across sweeps -- unlike the
             L_{1/2} prior, the CSP stick is not regenerated from B.
             V[:-1] holds v_1..v_{H-1}; V[-1] is a structural
             placeholder for v_H = 1 whose value is IGNORED on entry
             (omega is computed from V[:-1] alone) and set to 0 on
             return.  v_l for l < H are kept inside (eps, 1 - eps).
    Z      : (H,) 0-based categorical draws, Z[h] = z_h - 1.  With
             h_idx = arange(H): column h is ACTIVE iff Z[h] > h_idx[h]
             (paper's z_h > h), and H* = sum(Z > h_idx).
    D      : (p, H) PRIOR STANDARD DEVIATIONS of the loadings:
             D[:, h] = sqrt(theta_h), i.e. exactly the `w` scales the
             GS_LH_JAX B_sample consumes (prior precision 1/D^2 --
             the paper's diag(theta)^{-1} in the B_j conditional).
    theta_inf : the paper's spike VARIANCE theta_infty (marginal
             B_jh | spike ~ N(0, theta_inf)); NOT a standard deviation.

Numerical design (everything must stay finite in float32):
    * the z-step never exponentiates: jax.random.categorical consumes
      the (H, H) log-weight matrix directly (Gumbel-max), so logits of
      magnitude ~1e6 or -inf entries cannot overflow;
    * log omega is assembled from an EXCLUSIVE cumulative sum of
      log1p(-v), never as (inclusive cumsum) - log(1 - v_l), which
      yields inf - inf = NaN when some v_l rounds to exactly 1;
    * v is clamped to [eps, 1 - eps] (eps = float32 machine epsilon)
      before every log, so omega is strictly positive and finite --
      the clamp is statistically invisible (Beta posterior draws
      within eps of {0, 1} have probability < 1e-14);
    * the t marginal uses log1p(sum B^2 / (2 b)) -- the sum of squares
      of a shrunk column underflows the log's argument towards 0 and
      plain log(1 + x) would return plain 0 there;
    * the t log-normalizer `constant` (see make_log_t_constant) is
      computed on the HOST in float64: it contains lgamma(a + p/2),
      whose float32 error (~1e-3 at p ~ 1000) would otherwise shift
      the spike/slab odds;
    * theta enters as its SQUARE ROOT only: the raw precision 1/theta
      reaches 1/theta_inf ~ 20 on the spike and much more on tiny IG
      draws; the ill-conditioning is delegated to B_sample's whitened
      solver, which is built for it (M = I + PSD by construction).

Implementation: JAX-based, functional style; the module stays
un-jitted and is compiled inline into the driver's scan body.
"""

import math

import jax
import jax.numpy as jnp

__all__ = ["shrinkage", "z_logits", "make_log_t_constant"]

_F32 = jnp.float32
_EPS = jnp.finfo(_F32).eps


def make_log_t_constant(p: int, a_theta: float, b_theta: float) -> float:
    r"""Host-side (float64) log-normalizer of the slab marginal.

    The p-variate Student-t with nu = 2*a_theta degrees of freedom and
    scale matrix (b_theta / a_theta) I_p has log-density

        log t(x) = log G(a + p/2) - log G(a)
                   - (p/2) log(2 pi b)                        [constant]
                   - 0.5 (p + 2 a) log(1 + x'x / (2 b)),       [variable]

    where the a_theta of (nu pi)^{p/2} |Sigma|^{1/2} cancels against
    the scale: (2 a pi)^{p/2} (b/a)^{p/2} = (2 pi b)^{p/2}.  Pass the
    returned float as `constant` to z_logits / shrinkage.  It depends
    on p: recompute it whenever the problem size changes.
    """
    return (math.lgamma(a_theta + 0.5 * p) - math.lgamma(a_theta)
            - 0.5 * p * math.log(2.0 * math.pi * b_theta))


def _log_omega(V: jax.Array) -> jax.Array:
    r"""log omega_l, l = 1..H, from the stick V (length H).

    omega_1 = v_1,  omega_l = v_l prod_{m<l} (1 - v_m)  (l < H),
    omega_H = prod_{m<H} (1 - v_m)  (the remaining stick, since the
    paper fixes v_H = 1).  Only V[:-1] is read: the placeholder V[-1]
    never enters, so its value on entry is irrelevant.
    """
    v = jnp.clip(V[:-1], _EPS, 1.0 - _EPS)
    lom = jnp.log1p(-v)                       # log(1 - v_m), m = 1..H-1
    c = jnp.cumsum(lom)                       # inclusive prefix sums
    c_excl = jnp.concatenate([jnp.zeros((1,), _F32), c[:-1]])
    # exclusive prefix for l <= H-1; total for l = H.  Assembling from
    # the EXCLUSIVE sum (never c - log(1 - v_l)) keeps every entry
    # finite even if some v_l rounds to exactly 1.
    return jnp.concatenate([jnp.log(v) + c_excl, c[-1:]])


def z_logits(
    B: jax.Array,
    V: jax.Array,
    a_theta: float,
    b_theta: float,
    theta_inf: float,
    constant: float,
) -> jax.Array:
    r"""(H, H) unnormalized log-weights of the categorical for z (eq. 8).

    Row h (one per column of B) holds, over candidate values l:

        logits[h, l] = log omega_l + log d_h(l),

        log d_h(l) = log N_p(B_.h; 0, theta_inf I)   if l <= h,
                     log t_{2a}(B_.h; 0, (b/a) I)    if l >  h,

    with BOTH marginal log-densities indexed by the ROW h: eq. (8)
    evaluates the spike and slab marginals at lambda_h, the hth column
    of B, for every candidate l (they differ only through omega_l).
    Feed the result to jax.random.categorical(., axis=1); no
    normalization or stabilization is needed on the caller's side.
    """
    B = jnp.asarray(B, dtype=_F32)
    V = jnp.asarray(V, dtype=_F32)
    p, H = B.shape
    h_idx = jnp.arange(H)

    ssq = jnp.sum(B * B, axis=0)               # sum_j B_jh^2,  (H,)

    # spike marginal: N_p(B_.h; 0, theta_inf I) in closed form
    log_n = (-0.5 * p * jnp.log(2.0 * jnp.pi * theta_inf)
             - ssq / (2.0 * theta_inf))
    # slab marginal: variable part of the p-variate t (constant is
    # host-precomputed in float64, see make_log_t_constant)
    log_t = (constant
             - 0.5 * (p + 2.0 * a_theta) * jnp.log1p(ssq / (2.0 * b_theta)))

    dens = jnp.where(h_idx[None, :] <= h_idx[:, None],
                     log_n[:, None], log_t[:, None])       # (H, H)
    return dens + _log_omega(V)[None, :]


def shrinkage(
    key: jax.Array,
    B: jax.Array,
    V: jax.Array,
    alpha: float,
    a_theta: float,
    b_theta: float,
    theta_inf: float,
    constant: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""One draw of (D, V_new, Z) -- Algorithm 1 steps 4-7.

    Parameters
    ----------
    key : jax.random.PRNGKey
    B : array-like, shape (p, H)
        Loading matrix at the current sweep (H >= 2; the H = 1 model
        has no random stick and is rejected by the driver).
    V : array-like, shape (H,)
        Stick state from the previous sweep; see module docstring.
        V[:-1] in (0, 1) (clamped to [eps, 1 - eps] internally),
        V[-1] ignored.
    alpha : float
        CSP concentration; prior expected number of active factors.
        Enters only the Beta rate of the v update.
    a_theta, b_theta : float
        IG(a_theta, b_theta) slab of the spike-and-slab prior on
        theta_h.  a_theta > 1 is required by the paper's Theorem 2
        (finite E(theta_h)); check at the driver level.
    theta_inf : float
        Spike VARIANCE theta_infty (paper's default 0.05).
    constant : float
        Host-precomputed t log-normalizer, see make_log_t_constant.

    Returns
    -------
    D : jnp.ndarray, (p, H), float32
        Prior standard deviations sqrt(theta_h) per column, broadcast
        across rows (feed to B_sample as the `D` scales).
    V_new : jnp.ndarray, (H,), float32
        Fresh stick: v_l ~ Be(1 + #{z=l}, alpha + #{z>l}) for
        l = 1..H-1, clamped to [eps, 1 - eps]; placeholder 0 last.
        This is CHAIN STATE: pass it back on the next sweep.
    Z : jnp.ndarray, (H,), int32
        0-based z draws.  Column h is active iff Z[h] > h; the number
        of active factors is H* = sum(Z > arange(H)) -- collect it per
        sweep for the posterior E(H* | y).
    """
    B = jnp.asarray(B, dtype=_F32)
    V = jnp.asarray(V, dtype=_F32)
    p, H = B.shape
    h_idx = jnp.arange(H)

    key_z, key_v, key_t = jax.random.split(key, 3)

    # ---- step 4: z_h | B, omega  (eq. 8) ------------------------------------
    Z = jax.random.categorical(key_z,
                               z_logits(B, V, a_theta, b_theta, theta_inf,
                                        constant), axis=1)      # 0-based

    # ---- step 5: v_l | z  (conjugate Beta), l = 1..H-1 ----------------------
    counts = (Z[:, None] == h_idx[None, :]).sum(0)   # #{z_h = l},   (H,)
    gt = H - jnp.cumsum(counts)                      # #{z_h > l},   (H,)
    v_new = jax.random.beta(key_v, 1.0 + counts[:-1], alpha + gt[:-1],
                            dtype=_F32)
    v_new = jnp.clip(v_new, _EPS, 1.0 - _EPS)
    V_new = jnp.concatenate([v_new, jnp.zeros((1,), _F32)])

    # ---- step 7: theta_h | z, B -- spike at theta_inf, slab IG --------------
    # All H columns are drawn (fixed shapes under jit); the where() then
    # keeps sqrt(theta_inf) on the spiked ones.  theta = rate / g with
    # g ~ Ga(shape, 1) is the reciprocal-gamma identity for the IG draw.
    active = Z > h_idx                               # slab: z_h > h
    rate = b_theta + 0.5 * jnp.sum(B * B, axis=0)
    theta = rate / jax.random.gamma(key_t, a_theta + 0.5 * p,
                                    shape=(H,), dtype=_F32)
    col_std = jnp.where(active, jnp.sqrt(theta), jnp.sqrt(theta_inf))

    return jnp.broadcast_to(col_std, (p, H)), V_new, Z

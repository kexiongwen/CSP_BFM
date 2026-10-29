<h1 align="center">CSP_JAX</h1>

<p align="center">
  Bayesian factor analysis with a <b>cumulative shrinkage process</b> prior, in JAX.<br/>
  <sub>JAX implementation of the Gibbs sampler from Legramanti, Durante &amp; Dunson (2020), <i>Biometrika</i> 107(3), 745–752.</sub>
</p>

<p align="center">
  <a href="https://doi.org/10.1093/biomet/asaa008"><img alt="Paper: Biometrika 2020" src="https://img.shields.io/badge/paper-Biometrika%202020-blue"></a>
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue">
  <img alt="JAX 0.4.36+" src="https://img.shields.io/badge/JAX-%E2%89%A50.4.36-orange">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green">
</p>

CSP_JAX fits the Gaussian factor model with an **increasing-shrinkage prior** on
the factor variances. Redundant factors are progressively pushed to zero by a
stick-breaking spike-and-slab prior, so the number of factors is learned from
the data rather than fixed in advance — the paper's headline feature. The package
provides exact Gibbs sampling, a small sklearn-style estimator, and optional
Rotation-Sign-Permutation (RSP) post-processing for factor switching.

## Highlights

- **Automatic dimension selection** — `H* = Σ_h 1(z_h > h)` is sampled every
  sweep and summarized by `n_factors()` / `h_star_summary()`.
- **Exact Gibbs sampler** — every block is an exact full conditional (whitened
  perturb-and-solve for the factors and loadings, conjugate updates elsewhere);
  no Metropolis–Hastings tuning.
- **Bit-exact reproducibility** — the same `seed` reproduces the chain
  bit-for-bit; chunking and thinning do not change the draws.
- **sklearn-style estimator** — fitted accessors, credible-interval dimension
  rule, RSP alignment, on-demand factor scores, Monte Carlo moments of `Σ`.
- **Optional diagnostics** — per-entry ESS via ArviZ.

## Installation

```bash
pip install -e .            # core: jax>=0.4.36, numpy
pip install -e ".[ess]"     # + arviz      (ESS diagnostics)
pip install -e ".[rsp]"     # + optax      (RSP alignment when H > 7)
```

Python ≥ 3.12. No build step is required; the package also works directly from
the repository root.

## Quick start

### Functional interface

```python
import numpy as np
from CSP_JAX import GS_CSP

rng = np.random.default_rng(0)
B_true = np.zeros((60, 3))
B_true[:20, 0] = 1.0
B_true[20:40, 1] = 1.5
B_true[40:, 2] = 0.8
X = rng.standard_normal((400, 3)) @ B_true.T + 0.5**0.5 * rng.standard_normal((400, 60))
X -= X.mean(axis=0)                       # the model has zero mean

out = GS_CSP(X, H=10, n_iter=1500, burn_in=500, seed=2025, verbose=False)
print(out["H_star"].mean())               # ≈ 3 — the true number of factors
```

The driver returns `"B" (S, p, H)`, `"sigma2" (S, p)` and `"H_star" (S,)`,
plus final-state diagnostics (`"eta_last"`, `"D_last"`, `"V_last"`,
`"H_star_last"`, `"seed"`, `"runtime_sec"`).

### Estimator interface

```python
from CSP_JAX import GibbsCSP

gs = GibbsCSP(H=10, n_iter=2000, burn_in=1000, seed=2025).fit(X)  # centers internally
gs.n_factors()                 # posterior median of H*  (the CSP rule)
gs.h_star_summary()            # mean / median / 95% credible interval of H*
gs.B_hat_, gs.Sigma_           # posterior-mean loadings, plug-in covariance
gs.Sigma_mc_, gs.Sigma_var_    # Monte Carlo posterior moments of Sigma
```

## Model

For observations $x_i \in \mathbb{R}^p$ and latent factors $\eta_i \in \mathbb{R}^H$,

$$
x_i = \Lambda\eta_i + \varepsilon_i,\qquad
\eta_i \sim \mathcal{N}_H(0, I_H),\qquad
\varepsilon_i \sim \mathcal{N}_p(0,\Omega),\qquad
\Omega = \operatorname{diag}(\sigma_1^2,\dots,\sigma_p^2),
$$

with the cumulative shrinkage process prior (paper eq. 6) on the column
variances of $\Lambda=(\lambda_{jh})$:

$$
(\theta_h \mid \pi_h) \sim (1-\pi_h)\operatorname{IG}(a_\theta,b_\theta) +
  \pi_h\,\delta_{\theta_\infty},
\qquad
\pi_h=\sum_{l\le h}\omega_l,
\qquad
\omega_l=v_l\prod_{m<l}(1-v_m),
\qquad
v_h \sim \operatorname{Be}(1,\alpha),
$$

truncated at $H$ columns ($v_H=1$), with
$(\lambda_{jh}\mid\theta_h)\sim\mathcal{N}(0,\theta_h)$ and
$\sigma_j^2\sim\operatorname{IG}(a_\sigma,b_\sigma)$.
Because $\pi_h$ increases with $h$, later factors are shrunk harder; the
truncation leaves at most $H-1$ active factors and `H*` counts them.
$\alpha$ equals the prior expected number of active factors in the infinite
process, before truncation. Current implementation defaults are
$\alpha=4$, $a_\theta=b_\theta=2$, $\theta_\infty=0.05$,
$(a_\sigma,b_\sigma)=(1,1)$. The paper used $\alpha=5$ and
$(a_\sigma,b_\sigma)=(1,0.3)$, with the same remaining prior parameters.

> [!NOTE]
> - `a_theta > 1` is required for the paper's Theorem 2; otherwise a
>   `RuntimeWarning` is issued (the sampler on the truncated model remains valid).
> - `H = p + 1` is the conservative truncation; larger `H` issues a
>   `RuntimeWarning` (paper §3.3). Monitor `E(π_H) = 1 - (α/(1+α))^H` to check
>   that `H` is large enough.
> - Algorithm 2 (adaptive `H`) is not ported.

## Factor switching (RSP)

Raw MCMC draws can differ by orthogonal transforms (rotation, sign, permutation),
so their elementwise mean may obscure the loading structure. `align_factors()`
post-processes the retained draws with `CSP_JAX._rsp` — a per-draw varimax
rotation followed by exact sign-permutation alignment (Papastamoulis &
Ntzoufras 2022; Rey-Sáez & Revuelta 2026):

```python
gs.align_factors()             # varimax + signed-permutation alignment
gs.B_aligned_                  # (S, p, H) aligned draws; B_samples_ stays raw
gs.rsp_result_                 # losses, signs, permutations, transforms, diagnostics
scores = gs.sample_factors(X, return_eta=True, return_eta_aligned=True, seed=123)
```

> [!IMPORTANT]
> - `B_hat_` becomes the aligned posterior mean; `B_samples_` and `ess()` keep
>   using the raw draws.
> - `H_star_samples_` is unchanged and only meaningful **before** alignment: the
>   varimax rotation mixes spike columns into active ones.
> - `Sigma_` (plug-in) changes under alignment; `Sigma_mc_` (per-draw invariant)
>   does not.
> - RSP `status_code 4` raises `FloatingPointError` with no estimator state
>   change; `status_code 3` warns and accepts the finite results.
> - The varimax tolerance (`eps=1e-5`) monitors relative stagnation of the
>   gradient's nuclear norm. The default `init="auto"` uses representative-draw
>   initialization to mitigate mean cancellation. Neither the stopping tolerance
>   nor this initialization guarantees a global optimum or bounds loading errors.

## Numerical design

- float32 end-to-end. All sampled systems are `I + PSD` by construction and
  solved by plain Cholesky; the factor/loading blocks use whitened
  perturb-and-solve.
- The `z` step works in log space (no exponentiation); `log ω` is assembled from
  an exclusive cumsum of clamped sticks; the Student-t normalizer is computed
  host-side in float64.
- Randomness is explicit: keys are split per sweep, so the same `seed` replays
  the chain exactly. The chain runs as chunked `lax.scan` with ahead-of-time
  compilation; every chunk is host-audited for finite values.
- `fit()` is transactional (a failure restores the previous estimator state);
  a failed RSP run leaves estimates untouched.

## Package layout

```
CSP_JAX/
├── GS_CSP.py            # sampler driver + principal-factor initialization
├── eta_sample.py B_sample.py Sigma2_sample.py shrinkage.py   # Gibbs blocks
├── base.py gibbs.py     # BaseCSP / GibbsCSP estimator
├── ESS.py _posterior.py _factor_scores.py
└── _rsp/                # RSP alignment (varimax + exact sign-permutation)
PBMC/                    # PBMC3k preprocessing and CSP analysis workflow
demos/                   # two executed notebooks
```

## Demos

- [`demos/demo1.ipynb`](demos/demo1.ipynb) — five block factors, `n=100 ≪ p=1956`,
  homoscedastic noise: `E(H*|y)=4` at `H=50` (the 45 spike columns absorb the
  weakest factor; with the conservative `H=10` the design recovers 5 — the
  paper §3.3 trade-off).
- [`demos/demo2.ipynb`](demos/demo2.ipynb) — sparse random loadings,
  `n=100 ≪ p=1000`, heteroscedastic noise: `E(H*|y)=5` at `H=50`.

## PBMC3k analysis

The fixed PBMC workflow is `X_scaled -> GibbsCSP(center=False) -> RSP ->
conditional-mean scores -> loading-column L2 norm weighting -> all 50 columns
-> Leiden (resolution 0.40) -> 9 clusters`. Cluster plots use display labels
1–9; raw labels and marker files remain zero-based. The native `H*` posterior
is summarized separately in the original CSP coordinates, not after varimax
rotation. See `PBMC/plan.md` for the fixed workflow and current diagnostics.

`PBMC/data_processing.py` reproduces the Seurat/Scanpy preprocessing, including
`seurat_v3` HVG selection on counts with `n_top_genes=2000`,
`min_mean=0.0125`, `max_mean=3`, and `min_disp=0.5`, followed by regression of
`total_counts` and `pct_counts_mt` and `scale(max_value=10)`. Only the 2000
HVGs are densified for the `2638 x 2000` model input. The full AnnData retains
`lognorm` for Wilcoxon marker analysis.

`PBMC/PCA.py` builds the independent PCA reference from the same `X_scaled`:
50 components, the first 40 for the neighbor graph, and Leiden resolution 0.7.
PCA is a reference, not cell-type truth. ARI and AMI measure clustering
agreement only; the output field named `nmi` contains AMI for compatibility.

The fixed CSP settings are `H=50`, `alpha=5`, `a_theta=b_theta=2`,
`theta_inf=0.05`, `a_sigma=b_sigma=1`, `n_iter=20000`, `burn_in=10000`,
`thin=10`, and `seed=2026`. RSP is required. Conditional means are accumulated
in batches using each draw's RSP transform and multiplied by the aligned
`B_hat_` column norms. All 50 weighted columns are used for the Euclidean
neighbor graph and Leiden clustering. The current nine-cluster result is fixed
as an analysis choice, not evidence for a true cell-type count.

```bash
# Fixed preprocessing
python PBMC/data_processing.py PBMC/data/pbmc3k_raw.h5ad \
  --output PBMC/data/notebook_scaled_preprocessed.h5ad \
  --qc-figure-dir PBMC/figures/notebook_scaled_qc

# PCA reference baseline
python PBMC/PCA.py PBMC/data/pbmc3k_raw.h5ad \
  --output PBMC/data/notebook_scaled_pca_baseline.h5ad \
  --figure-dir PBMC/figures/notebook_scaled_pca

# Formal CSP chain (1000 retained draws)
python PBMC/CSP_GS.py PBMC/data/notebook_scaled_pca_baseline.h5ad \
  --output PBMC/data/notebook_scaled_csp_norm_d50_r0p4_k9_n20000_b10000_t10_h50.h5ad \
  --figure-dir PBMC/figures/notebook_scaled_csp_norm_d50_r0p4_k9_n20000_b10000_t10_h50

# Summary report
python -m jupyter nbconvert --to notebook --execute --inplace PBMC/PBMC.ipynb
```

Outputs include the downstream H5AD, a separate model NPZ with raw draws,
`H*`, RSP transforms, all 50 score columns, loading norms and diagnostics,
figures, and top-5 marker tables. `PBMC/PBMC.ipynb` summarizes the saved formal
results without hard-coding cluster counts or agreement metrics.

The saved `validation_status` is `review_required` when RSP is incomplete, ESS
is low, or every retained draw has `H* = H - 1`. Saturation means that the
truncation and native dimension posterior are not validated; it is not evidence
that the model found the correct number of factors. Do not select `theta_inf`,
`H`, or `alpha` by clustering performance.

PBMC scripts require Scanpy, AnnData, SciPy, pandas, scikit-learn, scikit-misc,
python-igraph, umap-learn, and matplotlib in addition to the `ess` and `rsp`
optional dependencies; these analysis dependencies are not required by the
core package.

## Citation

```bibtex
@article{legramanti2020bayesian,
  title   = {Bayesian cumulative shrinkage for infinite factorizations},
  author  = {Legramanti, Sirio and Durante, Daniele and Dunson, David B.},
  journal = {Biometrika},
  volume  = {107},
  number  = {3},
  pages   = {745--752},
  year    = {2020},
  doi     = {10.1093/biomet/asaa008}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).

"""Fixed single-chain CSP workflow for the shared PBMC3k baseline.

The representation uses X_scaled, RSP, loading-column L2 norm weighting,
and all 50 conditional-mean score columns. CSP's native H* posterior is
reported separately in the original prior coordinates and is never inferred
from the RSP-aligned representation.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import arviz as az
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CSP_JAX import GibbsCSP

if __package__:
    from .data_processing import _jsonable, _software_versions, load_baseline_input
    from .downstream import DownstreamConfig, run_downstream, write_top_markers
else:
    from data_processing import _jsonable, _software_versions, load_baseline_input
    from downstream import DownstreamConfig, run_downstream, write_top_markers

__all__ = [
    "CSPConfig",
    "CSPResult",
    "conditional_factor_mean",
    "run_csp",
    "run_csp_from_baseline",
    "save_csp_result",
    "save_csp_figures",
    "save_top_markers",
]

_F32 = jnp.float32
N_SCORE_COLUMNS = 50
_MARKER_CANDIDATES = (
    "IL7R", "CD79A", "MS4A1", "CD8A", "CD8B", "LYZ", "CD14",
    "GNLY", "NKG7", "FCGR3A", "MS4A7", "FCER1A", "CST3", "PPBP",
)


@dataclass(frozen=True)
class CSPConfig:
    H: int = 50
    alpha: float = 5.0
    a_theta: float = 2.0
    b_theta: float = 2.0
    theta_inf: float = 0.05
    a_sigma: float = 1.0
    b_sigma: float = 1.0
    n_iter: int = 20000
    burn_in: int = 10000
    thin: int = 10
    seed: int = 2026
    batch_size: int = 32
    n_neighbors: int = 10
    leiden_resolution: float = 0.4
    random_state: int = 0
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.H < 2 or self.thin < 1 or self.batch_size < 1:
            raise ValueError("H must be >= 2; thin and batch_size must be positive")
        for name in ("alpha", "a_theta", "b_theta", "theta_inf", "a_sigma", "b_sigma"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.a_theta <= 1:
            raise ValueError("a_theta must be > 1 for the planned CSP analysis")
        if not 0 <= self.burn_in < self.n_iter:
            raise ValueError("burn_in must satisfy 0 <= burn_in < n_iter")
        if (self.n_iter - self.burn_in) // self.thin < 2:
            raise ValueError("the run must retain at least two draws")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if self.n_neighbors < 1 or self.leiden_resolution <= 0:
            raise ValueError("n_neighbors must be positive and Leiden resolution > 0")


@dataclass
class CSPResult:
    adata: ad.AnnData
    estimator: GibbsCSP
    config: CSPConfig
    name: str
    diagnostics: dict
    batch_means: np.ndarray
    batch_indices: np.ndarray


@jax.jit
def _conditional_mean_batch(Xc, loadings, variances, transforms):
    def one_draw(args):
        B, sigma2, transform = args
        scale = jnp.sqrt(jnp.maximum(sigma2, jnp.finfo(_F32).tiny ** 0.5))
        C = B.T / scale
        precision = C @ C.T + jnp.eye(B.shape[1], dtype=_F32)
        rhs = C @ (Xc.T / scale[:, None])
        factor = jnp.linalg.cholesky(precision)
        conditional_mean = jsla.cho_solve((factor, True), rhs)
        return transform.T @ conditional_mean

    return jax.lax.map(one_draw, (loadings, variances, transforms))


def conditional_factor_mean(
    X,
    loadings,
    variances,
    offset,
    *,
    transforms=None,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average per-draw conditional means without storing all score draws."""
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(
        batch_size, (int, np.integer)
    ) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    loadings = np.asarray(loadings)
    variances = np.asarray(variances)
    if loadings.ndim != 3 or min(loadings.shape) < 1:
        raise ValueError("loadings must have nonempty shape (S, p, H)")
    S, p, H = loadings.shape
    if variances.shape != (S, p):
        raise ValueError(f"variances must have shape {(S, p)}")
    if not np.isfinite(loadings).all() or not np.isfinite(variances).all():
        raise ValueError("loadings and variances must be finite")
    if np.any(variances <= 0):
        raise ValueError("variances must be strictly positive")

    X64 = np.asarray(X, dtype=np.float64)
    offset64 = np.asarray(offset, dtype=np.float64)
    if X64.ndim != 2 or X64.shape[0] < 1 or X64.shape[1] != p:
        raise ValueError(f"X must have shape (n, {p}), got {X64.shape}")
    if offset64.shape != (p,) or not np.isfinite(offset64).all():
        raise ValueError(f"offset must be finite with shape {(p,)}")
    if not np.isfinite(X64).all():
        raise ValueError("X must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        Xc = (X64 - offset64).astype(np.float32)
    if not np.isfinite(Xc).all():
        raise ValueError("centering X overflowed the float32 coercion")

    if transforms is None:
        transforms = np.broadcast_to(np.eye(H, dtype=np.float32), (S, H, H))
    else:
        transforms = np.asarray(transforms)
        if transforms.shape != (S, H, H):
            raise ValueError(f"transforms must have shape {(S, H, H)}")
        if not np.isfinite(transforms).all():
            raise ValueError("transforms must be finite")

    n = X64.shape[0]
    n_batches = (S + int(batch_size) - 1) // int(batch_size)
    batch_means = np.empty((n_batches, n, H), dtype=np.float64)
    batch_indices = np.empty((n_batches, 2), dtype=np.int64)
    total = np.zeros((H, n), dtype=np.float64)
    Xc_j = jnp.asarray(Xc)
    for batch_index, start in enumerate(range(0, S, int(batch_size))):
        stop = min(start + int(batch_size), S)
        aligned = np.asarray(_conditional_mean_batch(
            Xc_j,
            jnp.asarray(loadings[start:stop]),
            jnp.asarray(variances[start:stop]),
            jnp.asarray(transforms[start:stop]),
        ))
        if not np.isfinite(aligned).all():
            raise FloatingPointError(
                f"conditional means became non-finite in batch {batch_index}"
            )
        batch_total = aligned.sum(axis=0, dtype=np.float64)
        total += batch_total
        batch_means[batch_index] = (batch_total / (stop - start)).T
        batch_indices[batch_index] = (start, stop)
    return (total / S).T, batch_means, batch_indices


def _diagnostic_jsonable(value):
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    if isinstance(value, dict):
        return {str(key): _diagnostic_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_jsonable(item) for item in value]
    return _jsonable(value)


def _rsp_summary(estimator: GibbsCSP, warning_messages: list[str]) -> dict:
    rsp = estimator.rsp_result_
    n_iter = int(rsp.n_iter)
    return {
        "status_code": int(rsp.status_code),
        "converged": bool(rsp.converged),
        "n_iter": n_iter,
        "stop_reason": str(rsp.diagnostics["stop_reason"]),
        "nonfinite_fields": list(rsp.diagnostics["nonfinite_fields"]),
        "loss_initial": float(rsp.loss_initial),
        "loss_final": float(rsp.loss_final),
        "loss_history": np.asarray(rsp.loss_history[:n_iter]).copy(),
        "loss_history_post": np.asarray(rsp.loss_history_post[:n_iter]).copy(),
        "varimax_n_iter": np.asarray(rsp.varimax_n_iter).copy(),
        "warnings": warning_messages,
    }


def _ess_summary(values: dict, n_draws: int) -> dict:
    return {
        key: {
            "min": float(np.min(array)),
            "median": float(np.median(array)),
            "mean": float(np.mean(array)),
            "max": float(np.max(array)),
            "n_draws": int(n_draws),
        }
        for key, array in values.items()
    }


def _score_half_stability(
    batch_means: np.ndarray,
    batch_indices: np.ndarray,
    final_scores: np.ndarray,
) -> dict:
    n_draws = int(batch_indices[-1, 1])
    split = n_draws // 2
    first = np.zeros(final_scores.shape, dtype=np.float64)
    second = np.zeros(final_scores.shape, dtype=np.float64)
    for mean, (start, stop) in zip(batch_means, batch_indices):
        count_first = max(0, min(int(stop), split) - int(start))
        count_second = int(stop - start) - count_first
        first += mean * count_first
        second += mean * count_second
    first /= split
    second /= (n_draws - split)
    difference = second - first
    denominator = np.maximum(np.abs(final_scores), np.finfo(np.float64).eps)
    relative = np.abs(difference) / denominator
    return {
        "split_draw": split,
        "absolute_difference_median": float(np.median(np.abs(difference))),
        "absolute_difference_max": float(np.max(np.abs(difference))),
        "relative_difference_median": float(np.median(relative)),
        "relative_difference_max": float(np.max(relative)),
    }


def _sampling_diagnostics(
    estimator: GibbsCSP,
    hvg_names: np.ndarray,
    batch_means: np.ndarray,
    batch_indices: np.ndarray,
    final_scores: np.ndarray,
) -> dict:
    B = estimator.B_samples_
    sigma2 = estimator.sigma2_samples_
    hs = estimator.H_star_samples_
    S = B.shape[0]
    bulk = estimator.ess(method="bulk")
    tail = estimator.ess(method="tail", prob=0.05)
    bulk_summary = _ess_summary(bulk, S)
    tail_summary = _ess_summary(tail, S)

    aligned = np.asarray(estimator.B_aligned_)
    mcse_B = np.asarray(az.mcse(aligned[None], method="mean"))
    mcse_sigma2 = np.asarray(az.mcse(sigma2[None], method="mean"))
    mcse_hs = float(np.asarray(az.mcse(hs[None], method="mean")))
    covariance_diag = np.einsum("spk,spk->sp", B, B) + sigma2
    mcse_covariance = np.asarray(az.mcse(covariance_diag[None], method="mean"))

    marker_indices = np.array(
        [i for i, name in enumerate(hvg_names) if name in _MARKER_CANDIDATES],
        dtype=np.int64,
    )
    marker_names = hvg_names[marker_indices]
    split = S // 2
    segment_covariance = (
        covariance_diag[split:].mean(axis=0) - covariance_diag[:split].mean(axis=0)
    )
    segment_sigma2 = sigma2[split:].mean(axis=0) - sigma2[:split].mean(axis=0)
    near_constant = np.std(B, axis=0) <= np.finfo(np.float32).eps
    min_bulk = min(
        bulk_summary["B"]["min"],
        bulk_summary["sigma2"]["min"],
        bulk_summary["H_star"]["min"],
    )
    h_star_summary = estimator.h_star_summary(level=0.95)
    h_star_summary["ci"] = np.asarray(h_star_summary["ci"], dtype=np.float64)
    h_star_saturated = bool(np.all(hs >= estimator.H - 1))
    return {
        "bulk_summary": bulk_summary,
        "tail_summary": tail_summary,
        "bulk_B": bulk["B"],
        "bulk_sigma2": bulk["sigma2"],
        "bulk_H_star": bulk["H_star"],
        "tail_B": tail["B"],
        "tail_sigma2": tail["sigma2"],
        "tail_H_star": tail["H_star"],
        "mcse_B_aligned": mcse_B,
        "mcse_sigma2": mcse_sigma2,
        "mcse_H_star": mcse_hs,
        "mcse_covariance_diag": mcse_covariance,
        "h_star_summary": h_star_summary,
        "h_star_at_upper_bound_fraction": float(np.mean(hs >= estimator.H - 1)),
        "h_star_saturated": h_star_saturated,
        "segment_h_star_mean_difference": float(hs[split:].mean() - hs[:split].mean()),
        "marker_indices": marker_indices,
        "marker_names": marker_names,
        "segment_covariance_difference": segment_covariance[marker_indices],
        "segment_sigma2_difference": segment_sigma2[marker_indices],
        "score_half_stability": _score_half_stability(
            batch_means, batch_indices, final_scores
        ),
        "near_constant_count": int(near_constant.sum()),
        "review_required": bool(
            not np.isfinite(min_bulk) or min_bulk < 400 or h_star_saturated
        ),
        "chain_coordinate_note": (
            "H* and same-index B differ by half a sweep; use marginal summaries, "
            "not per-draw joint interpretation. H* is not defined after RSP."
        ),
        "truncation_note": (
            "Every retained draw has H*=H-1; the dimension posterior and truncation "
            "are not validated." if h_star_saturated else
            "H* is not saturated at H-1 in every retained draw."
        ),
    }


def run_csp(
    adata: ad.AnnData,
    X: np.ndarray,
    hvg_names: np.ndarray,
    *,
    config: CSPConfig | None = None,
) -> CSPResult:
    config = config or CSPConfig()
    if config.H < N_SCORE_COLUMNS:
        raise ValueError(f"H must be at least {N_SCORE_COLUMNS} for the fixed representation")
    if X.shape != (adata.n_obs, len(hvg_names)):
        raise ValueError("X_scaled does not match the baseline cells and HVGs")

    estimator = GibbsCSP(
        H=config.H,
        alpha=config.alpha,
        a_theta=config.a_theta,
        b_theta=config.b_theta,
        theta_inf=config.theta_inf,
        a_sigma=config.a_sigma,
        b_sigma=config.b_sigma,
        center=False,
        n_iter=config.n_iter,
        burn_in=config.burn_in,
        thin=config.thin,
        seed=config.seed,
        verbose=config.verbose,
    )
    start = time.perf_counter()
    estimator.fit(X)
    fit_wall_sec = time.perf_counter() - start

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        estimator.align_factors()
        rsp_wall_sec = time.perf_counter() - start
    warning_messages = [str(item.message) for item in caught]
    rsp = _rsp_summary(estimator, warning_messages)
    rsp["applied"] = True
    transforms = estimator.rsp_result_.transforms

    start = time.perf_counter()
    scores, batch_means, batch_indices = conditional_factor_mean(
        X,
        estimator.B_samples_,
        estimator.sigma2_samples_,
        estimator.x_offset_,
        transforms=transforms,
        batch_size=config.batch_size,
    )
    score_wall_sec = time.perf_counter() - start
    S = estimator.B_samples_.shape[0]
    fit = {
        "n_iter": int(config.n_iter),
        "burn_in": int(config.burn_in),
        "thin": int(config.thin),
        "n_kept": int(S),
        "retained_sweeps": config.burn_in + config.thin * np.arange(1, S + 1),
        "seed": int(estimator.seed_),
        "center": False,
        "runtime_sec": float(estimator.runtime_sec_),
        "fit_wall_sec": float(fit_wall_sec),
        "rsp_wall_sec": float(rsp_wall_sec),
        "score_wall_sec": float(score_wall_sec),
        "backend": str(jax.default_backend()),
        "expected_pi_H": float(1 - (config.alpha / (1 + config.alpha)) ** config.H),
    }

    loading_column_norm = np.linalg.norm(estimator.B_hat_, axis=0)
    scores = scores * loading_column_norm
    batch_means = batch_means * loading_column_norm
    selected_columns = np.arange(N_SCORE_COLUMNS)
    scores = scores[:, selected_columns]
    batch_means = batch_means[:, :, selected_columns]
    if not np.isfinite(scores).all():
        raise FloatingPointError("final CSP scores contain non-finite values")

    active_mask = np.asarray(
        estimator.active_entries(level=0.95, interval="hpd", use_aligned=True)
    )
    singular_values = np.linalg.svd(scores, compute_uv=False)
    sampling = _sampling_diagnostics(
        estimator, hvg_names, batch_means, batch_indices, scores
    )

    name = "csp_scaled_norm_d50"
    representation = f"X_{name}"
    adata.obsm[representation] = scores
    run = run_downstream(
        adata,
        representation=representation,
        name=name,
        config=DownstreamConfig(
            n_pcs=N_SCORE_COLUMNS,
            n_neighbors=config.n_neighbors,
            leiden_resolution=config.leiden_resolution,
            random_state=config.random_state,
        ),
    )
    display_key = f"{run['leiden']}_display"
    adata.obs[display_key] = (
        (adata.obs[run["leiden"]].astype(int) + 1).astype(str).astype("category")
    )
    run["display_leiden"] = display_key

    comparison = {}
    if "leiden_scaled" in adata.obs:
        reference = adata.obs["leiden_scaled"].astype(str)
        current = adata.obs[run["leiden"]].astype(str)
        comparison["pca_scaled"] = {
            "ari": float(adjusted_rand_score(reference, current)),
            "nmi": float(adjusted_mutual_info_score(reference, current)),
        }

    run_complete = bool(rsp["status_code"] == 1)
    diagnostics = {
        "fit": fit,
        "sampling": sampling,
        "rsp": rsp,
        "active_mask_hpd": active_mask,
        "hpd_active_factor_count": int(active_mask.any(axis=0).sum()),
        "hpd_selected_active_factor_count": int(
            active_mask[:, selected_columns].any(axis=0).sum()
        ),
        "selected_columns": selected_columns,
        "score_singular_values": singular_values,
        "score_weighting": "loading_column_l2_norm",
        "loading_column_norm": loading_column_norm,
        "selected_loading_column_norm": loading_column_norm[selected_columns],
        "conditional_mean": {
            "batch_size": int(config.batch_size),
            "n_batches": int(batch_indices.shape[0]),
            "score_shape": list(scores.shape),
            "full_score_array_allocated": False,
            "aligned": True,
        },
        "comparison": comparison,
        "comparison_metric_note": "The nmi fields contain adjusted mutual information (AMI).",
        "run_complete": run_complete,
        "validation_status": (
            "review_required" if not run_complete or sampling["review_required"]
            else "complete"
        ),
    }
    adata.uns[name] = {
        "config_json": json.dumps(_jsonable(asdict(config)), sort_keys=True),
        "align": True,
        "weight_by_norm": True,
        "n_columns": N_SCORE_COLUMNS,
        "display_cluster_offset": 1,
        "display_label_note": (
            "Display labels equal raw zero-based cluster labels plus one; "
            "raw labels and marker files are unchanged."
        ),
        "input_matrix": "X_scaled",
        "hvg_names": np.asarray(hvg_names, dtype=str),
        "diagnostics": diagnostics,
        "run": run,
        "software": _software_versions(),
    }
    return CSPResult(
        adata, estimator, config, name, diagnostics, batch_means, batch_indices
    )


def run_csp_from_baseline(
    path: str | Path,
    *,
    config: CSPConfig | None = None,
) -> CSPResult:
    adata, X, hvg_names = load_baseline_input(path)
    return run_csp(adata, X, hvg_names, config=config)


def save_csp_result(
    result: CSPResult,
    output: str | Path,
) -> tuple[Path, Path]:
    output = Path(output)
    if output.suffix.lower() != ".h5ad":
        raise ValueError(f"output must end in .h5ad, got {output}")
    model_output = output.with_name(f"{output.stem}_model.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    estimator = result.estimator
    sampling = result.diagnostics["sampling"]
    metadata = result.adata.uns[result.name]
    diagnostics_json = json.dumps(
        _diagnostic_jsonable({
            "config": asdict(result.config),
            "representation": {
                "input_matrix": metadata["input_matrix"],
                "align": metadata["align"],
                "weight_by_norm": metadata["weight_by_norm"],
                "n_columns": metadata["n_columns"],
            },
            "diagnostics": result.diagnostics,
        }),
        sort_keys=True,
    )
    rsp = estimator.rsp_result_
    arrays = {
        "B_samples": estimator.B_samples_,
        "sigma2_samples": estimator.sigma2_samples_,
        "H_star_samples": estimator.H_star_samples_,
        "x_offset": estimator.x_offset_,
        "B_raw_mean": estimator.B_samples_.mean(axis=0),
        "B_posterior_mean": estimator.B_hat_,
        "B_aligned_mean": estimator.B_hat_,
        "transforms": rsp.transforms,
        "signs": rsp.signs,
        "permutations": rsp.permutations,
        "score_mean": result.adata.obsm[f"X_{result.name}"],
        "batch_means": result.batch_means,
        "batch_indices": result.batch_indices,
        "retained_sweeps": result.diagnostics["fit"]["retained_sweeps"],
        "active_mask_hpd": result.diagnostics["active_mask_hpd"],
        "selected_columns": result.diagnostics["selected_columns"],
        "selected_loading_column_norm": result.diagnostics["selected_loading_column_norm"],
        "loading_column_norm": result.diagnostics["loading_column_norm"],
        "score_singular_values": result.diagnostics["score_singular_values"],
        "bulk_B": sampling["bulk_B"],
        "bulk_sigma2": sampling["bulk_sigma2"],
        "bulk_H_star": sampling["bulk_H_star"],
        "tail_B": sampling["tail_B"],
        "tail_sigma2": sampling["tail_sigma2"],
        "tail_H_star": sampling["tail_H_star"],
        "mcse_B_aligned": sampling["mcse_B_aligned"],
        "mcse_sigma2": sampling["mcse_sigma2"],
        "mcse_covariance_diag": sampling["mcse_covariance_diag"],
        "D_last": estimator.raw_output_["D_last"],
        "V_last": estimator.raw_output_["V_last"],
        "eta_last": estimator.raw_output_["eta_last"],
        "H_star_last": estimator.raw_output_["H_star_last"],
        "hvg_names": metadata["hvg_names"],
        "cell_names": result.adata.obs_names.to_numpy(dtype=str),
        "diagnostics_json": diagnostics_json,
    }
    np.savez_compressed(model_output, **arrays)
    result.adata.uns[result.name]["model_npz"] = str(model_output.resolve())
    result.adata.write_h5ad(output, compression="gzip")
    return output, model_output


def save_top_markers(
    result: CSPResult, output_prefix: str | Path, n: int = 5
) -> tuple[list[dict], Path, Path]:
    marker_key = result.adata.uns[result.name]["run"]["markers"]
    return write_top_markers(result.adata, marker_key, "CSP", output_prefix, n)


def save_csp_figures(result: CSPResult, output_dir: str | Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("saving CSP figures requires matplotlib") from exc
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    name = result.name
    diagnostics = result.diagnostics

    singular = diagnostics["score_singular_values"]
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(np.arange(1, len(singular) + 1), singular, marker=".", linewidth=1)
    ax.set_xlabel("factor column")
    ax.set_ylabel("singular value")
    ax.set_title(f"Score spectrum: {name}")
    path = output_dir / f"score_spectrum_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    raw_norm = np.linalg.norm(result.estimator.B_samples_.mean(axis=0), axis=0)
    aligned_norm = np.linalg.norm(result.estimator.B_aligned_.mean(axis=0), axis=0)
    columns = np.arange(1, len(raw_norm) + 1)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(columns, raw_norm, marker=".", linewidth=1, label="Before RSP")
    ax.plot(columns, aligned_norm, marker=".", linewidth=1, label="After RSP")
    ax.set_xlabel("factor column")
    ax.set_ylabel("loading column L2 norm")
    ax.set_title("CSP")
    ax.legend()
    path = output_dir / f"loading_column_norms_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    run = result.adata.uns[name]["run"]
    fig = sc.pl.embedding(
        result.adata, basis=run["umap"],
        color=run.get("display_leiden", run["leiden"]),
        show=False, return_fig=True,
    )
    path = output_dir / f"umap_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    sampling = diagnostics["sampling"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].hist(sampling["bulk_B"].ravel(), bins=40)
    axes[0].set_title("Bulk ESS: loadings")
    axes[1].hist(sampling["bulk_sigma2"], bins=40)
    axes[1].set_title("Bulk ESS: noise variances")
    for ax in axes:
        ax.axvline(400, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("ESS")
        ax.set_ylabel("entries")
    path = output_dir / f"ess_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    hs = result.estimator.H_star_samples_
    sweeps = diagnostics["fit"]["retained_sweeps"]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(sweeps, hs, linewidth=0.8)
    ax.set_xlabel("sweep")
    ax.set_ylabel("H*")
    ax.set_title(f"CSP active-factor trace: {name}")
    path = output_dir / f"h_star_trace_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.hist(hs, bins=np.arange(hs.min(), hs.max() + 2) - 0.5, rwidth=0.8)
    ax.set_xlabel("H*")
    ax.set_ylabel("retained draws")
    ax.set_title(f"CSP active-factor posterior: {name}")
    path = output_dir / f"h_star_hist_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    marker_indices = sampling["marker_indices"]
    if len(marker_indices):
        j = int(marker_indices[0])
        B = result.estimator.B_samples_
        series = np.einsum("sk,sk->s", B[:, j, :], B[:, j, :]) + result.estimator.sigma2_samples_[:, j]
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        ax.plot(sweeps, series, linewidth=0.8)
        ax.set_xlabel("sweep")
        ax.set_ylabel(f"Sigma[{j}, {j}]")
        ax.set_title(f"Covariance trace: {sampling['marker_names'][0]}")
        path = output_dir / f"trace_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_path", nargs="?", type=Path,
        default=Path("PBMC/data/notebook_scaled_pca_baseline.h5ad"),
        help="shared PCA baseline H5AD with lognorm and the HVG mask",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()

    result = run_csp_from_baseline(args.data_path, config=CSPConfig())
    output = args.output or Path(
        f"PBMC/data/notebook_scaled_csp_norm_d50_r0p4_k9_n{result.config.n_iter}"
        f"_b{result.config.burn_in}_t{result.config.thin}_h{result.config.H}.h5ad"
    )
    h5ad_path, model_path = save_csp_result(result, output)
    fit = result.diagnostics["fit"]
    sampling = result.diagnostics["sampling"]
    cluster_key = result.adata.uns[result.name]["run"]["leiden"]
    print(
        f"{result.name}: kept={fit['n_kept']}, "
        f"clusters={result.adata.obs[cluster_key].nunique()}, "
        f"H*={sampling['h_star_summary']}"
    )
    print(f"RSP status={result.diagnostics['rsp']['status_code']}")
    print(
        "min bulk ESS="
        f"{min(sampling['bulk_summary'][key]['min'] for key in ('B', 'sigma2', 'H_star')):.3g}; "
        f"validation={result.diagnostics['validation_status']}"
    )
    print(f"Saved H5AD to {h5ad_path}; model to {model_path}")
    if args.figure_dir is not None:
        paths = save_csp_figures(result, args.figure_dir)
        print(f"Saved {len(paths)} figures to {args.figure_dir}")
    _, csv_path, md_path = save_top_markers(
        result, h5ad_path.with_name(f"{h5ad_path.stem}_top5_markers")
    )
    print(f"Saved top-5 markers to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()

"""Preprocess PBMC3k into the fixed ``X_scaled`` CSP model input.

The preprocessing mirrors the established LH_BFM PBMC workflow: covariate
regression, scaling, and clipping are restricted to 2000 highly variable
genes. The full AnnData retains ``lognorm`` for marker analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

__all__ = [
    "PreprocessConfig",
    "PreprocessedPBMC",
    "read_pbmc3k",
    "save_qc_figures",
    "preprocess_adata",
    "build_model_inputs",
    "preprocess_pbmc3k",
    "save_preprocessed",
    "load_baseline_input",
]


@dataclass(frozen=True)
class PreprocessConfig:
    min_genes: int = 200
    min_cells: int = 3
    max_genes: int = 2500
    max_pct_counts_mt: float = 5.0
    target_sum: float = 1e4
    n_top_genes: int = 2000
    hvg_min_mean: float = 0.0125
    hvg_max_mean: float = 3.0
    hvg_min_disp: float = 0.5
    regress_keys: tuple[str, ...] = ("total_counts", "pct_counts_mt")
    scale_max_value: float | None = 10.0
    expected_raw_shape: tuple[int, int] | None = (2700, 32738)
    expected_qc_shape: tuple[int, int] | None = (2638, 13714)
    qc_figure_dir: str | Path | None = None

    def __post_init__(self) -> None:
        if self.min_genes < 1 or self.min_cells < 1:
            raise ValueError("min_genes and min_cells must be positive")
        if self.max_genes <= self.min_genes:
            raise ValueError("max_genes must be greater than min_genes")
        if not 0 < self.max_pct_counts_mt <= 100:
            raise ValueError("max_pct_counts_mt must be in (0, 100]")
        if self.target_sum <= 0 or not np.isfinite(self.target_sum):
            raise ValueError("target_sum must be positive and finite")
        if self.n_top_genes < 1:
            raise ValueError("n_top_genes must be positive")
        if self.scale_max_value is not None and self.scale_max_value <= 0:
            raise ValueError("scale_max_value must be positive or None")


@dataclass
class PreprocessedPBMC:
    adata: ad.AnnData
    X_scaled: np.ndarray
    hvg_names: pd.Index

    @property
    def cell_names(self) -> pd.Index:
        return self.adata.obs_names

    @property
    def gene_names(self) -> pd.Index:
        return self.adata.var_names


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _software_versions() -> dict[str, str]:
    packages = (
        "scanpy", "anndata", "numpy", "pandas", "scipy", "scikit-misc",
        "numba", "scikit-learn", "h5py", "matplotlib", "seaborn",
        "igraph", "umap-learn", "jax", "jaxlib", "optax", "arviz",
        "csp-jax",
    )
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _file_checksum(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "size_bytes": path.stat().st_size}


def _record_stage(adata: ad.AnnData, stage: str, config: PreprocessConfig) -> None:
    metadata = adata.uns.setdefault("preprocessing", {
        "schema_version": 1,
        "source": {
            "format": "in_memory",
            "checksum_status": "unavailable: input not read by read_pbmc3k",
        },
    })
    metadata.setdefault("stages", {})[stage] = {
        "config_json": json.dumps(_jsonable(asdict(config)), sort_keys=True),
        "software": _software_versions(),
    }


def _check_expected_shape(
    actual: tuple[int, int], expected: tuple[int, int] | None, stage: str
) -> None:
    if expected is not None and tuple(actual) != tuple(expected):
        raise ValueError(
            f"unexpected {stage} shape: got {tuple(actual)}, expected "
            f"{tuple(expected)}. Check the input data, software versions, "
            "and operation order; do not adjust QC thresholds merely to "
            "match the tutorial dimensions. Set the corresponding expected "
            "shape to None to disable this check."
        )


def read_pbmc3k(
    data_path: str | Path,
    *,
    expected_shape: tuple[int, int] | None = (2700, 32738),
) -> ad.AnnData:
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"PBMC3k input does not exist: {data_path}")
    if data_path.suffix.lower() != ".h5ad":
        raise ValueError(f"expected a raw PBMC3k .h5ad file, got {data_path}")
    checksum = _file_checksum(data_path)
    adata = sc.read_h5ad(data_path)
    adata.var_names_make_unique()
    if not adata.obs_names.is_unique:
        raise ValueError("cell barcodes must be unique")
    _check_expected_shape(adata.shape, expected_shape, "raw PBMC3k")
    adata.uns["pbmc3k_source_path"] = str(data_path.resolve())
    adata.uns["pbmc3k_source_format"] = "h5ad"
    adata.uns["pbmc3k_source_dir"] = str(data_path.parent.resolve())
    adata.uns["preprocessing"] = {
        "schema_version": 1,
        "source": {
            "format": "h5ad",
            "path": str(data_path.resolve()),
            "hash_algorithm": "sha256",
            "files": {data_path.name: checksum},
            "software": _software_versions(),
        },
    }
    return adata


def _require_seurat_v3_dependency() -> None:
    try:
        import skmisc  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "highly_variable_genes(flavor='seurat_v3') requires scikit-misc. "
            "Install it in the PBMC analysis environment, for example with "
            "`python -m pip install scikit-misc`."
        ) from exc


def save_qc_figures(adata: ad.AnnData, output_dir: str | Path) -> tuple[Path, Path]:
    required = {"n_genes_by_counts", "total_counts", "pct_counts_mt"}
    missing = required.difference(adata.obs.columns)
    if missing:
        raise ValueError(f"missing QC metrics: {sorted(missing)}")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("saving QC figures requires matplotlib") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    violin_path = output_dir / "qc_metrics_violin.png"
    scatter_path = output_dir / "qc_scatter.png"

    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4,
        multi_panel=True,
        show=False,
    )
    fig = plt.gcf()
    fig.savefig(violin_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt", show=False, ax=axes[0])
    sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts", show=False, ax=axes[1])
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return violin_path, scatter_path


def preprocess_adata(
    adata: ad.AnnData, config: PreprocessConfig | None = None
) -> ad.AnnData:
    config = config or PreprocessConfig()
    if adata.n_obs < 1 or adata.n_vars < 1:
        raise ValueError("adata must contain at least one cell and one gene")

    sc.pp.filter_cells(adata, min_genes=config.min_genes)
    sc.pp.filter_genes(adata, min_cells=config.min_cells)

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )
    if config.qc_figure_dir is not None:
        save_qc_figures(adata, config.qc_figure_dir)

    keep = (
        (adata.obs["n_genes_by_counts"] > config.min_genes)
        & (adata.obs["n_genes_by_counts"] < config.max_genes)
        & (adata.obs["pct_counts_mt"] < config.max_pct_counts_mt)
    )
    adata = adata[keep].copy()
    _check_expected_shape(adata.shape, config.expected_qc_shape, "post-QC PBMC3k")
    if adata.n_obs == 0:
        raise ValueError("cell QC removed every observation")
    if adata.n_vars < config.n_top_genes:
        raise ValueError(
            f"n_top_genes={config.n_top_genes} exceeds the {adata.n_vars} "
            "genes remaining after QC"
        )

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=config.target_sum)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()

    _require_seurat_v3_dependency()
    sc.pp.highly_variable_genes(
        adata,
        layer="counts",
        n_top_genes=config.n_top_genes,
        min_mean=config.hvg_min_mean,
        max_mean=config.hvg_max_mean,
        min_disp=config.hvg_min_disp,
        flavor="seurat_v3",
    )
    n_hvg = int(adata.var["highly_variable"].sum())
    if n_hvg != config.n_top_genes:
        raise RuntimeError(
            f"expected {config.n_top_genes} highly variable genes, got {n_hvg}"
        )
    _record_stage(adata, "expression", config)
    return adata


def _dense_float64(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64, order="C")


def build_model_inputs(
    adata: ad.AnnData, config: PreprocessConfig | None = None
) -> tuple[np.ndarray, pd.Index]:
    config = config or PreprocessConfig()
    if "lognorm" not in adata.layers:
        raise ValueError("adata.layers['lognorm'] is required")
    if "highly_variable" not in adata.var:
        raise ValueError("run highly-variable-gene selection first")

    hvg_mask = adata.var["highly_variable"].to_numpy(dtype=bool)
    hvg_names = adata.var_names[hvg_mask].copy()
    if len(hvg_names) != config.n_top_genes:
        raise ValueError(
            f"the HVG mask contains {len(hvg_names)} genes, expected "
            f"{config.n_top_genes}"
        )
    missing_regressors = set(config.regress_keys).difference(adata.obs.columns)
    if missing_regressors:
        raise ValueError(f"missing regression covariates: {sorted(missing_regressors)}")

    model_adata = adata[:, hvg_names].copy()
    model_adata.layers["scaled"] = _dense_float64(model_adata.layers["lognorm"])
    if config.regress_keys:
        sc.pp.regress_out(
            model_adata,
            keys=list(config.regress_keys),
            layer="scaled",
        )
    sc.pp.scale(model_adata, layer="scaled", max_value=config.scale_max_value)
    X_scaled = _dense_float64(model_adata.layers["scaled"])

    expected_shape = (adata.n_obs, len(hvg_names))
    if X_scaled.shape != expected_shape:
        raise RuntimeError(
            f"X_scaled has shape {X_scaled.shape}, expected {expected_shape}"
        )
    if not np.isfinite(X_scaled).all():
        raise FloatingPointError("X_scaled contains non-finite values")
    _record_stage(adata, "model_inputs", config)
    return X_scaled, hvg_names


def preprocess_pbmc3k(
    data_path: str | Path,
    *,
    config: PreprocessConfig | None = None,
) -> PreprocessedPBMC:
    config = config or PreprocessConfig()
    adata = read_pbmc3k(data_path, expected_shape=config.expected_raw_shape)
    adata = preprocess_adata(adata, config)
    X_scaled, hvg_names = build_model_inputs(adata, config)
    return PreprocessedPBMC(adata, X_scaled, hvg_names)


def save_preprocessed(
    result: PreprocessedPBMC, output: str | Path
) -> tuple[Path, Path]:
    output = Path(output)
    if output.suffix.lower() != ".h5ad":
        raise ValueError(f"output must end in .h5ad, got {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = output.with_name(f"{output.stem}_model_inputs.npz")

    result.adata.write_h5ad(output, compression="gzip")
    np.savez_compressed(
        arrays_path,
        X_scaled=result.X_scaled,
        cell_names=result.cell_names.to_numpy(dtype=str),
        hvg_names=result.hvg_names.to_numpy(dtype=str),
        preprocessing_json=json.dumps(
            _jsonable(result.adata.uns.get("preprocessing", {})), sort_keys=True
        ),
    )
    return output, arrays_path


def load_baseline_input(
    path: str | Path,
    *,
    expected_n_cells: int | None = 2638,
    expected_n_hvg: int = 2000,
) -> tuple[ad.AnnData, np.ndarray, np.ndarray]:
    """Load a shared H5AD and reconstruct its fixed X_scaled model input."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PBMC baseline does not exist: {path}")
    adata = sc.read_h5ad(path)
    if "lognorm" not in adata.layers or "highly_variable" not in adata.var:
        raise ValueError("baseline must contain lognorm and the HVG mask")
    if expected_n_cells is not None and adata.n_obs != expected_n_cells:
        raise ValueError(f"expected {expected_n_cells} cells, got {adata.n_obs}")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("baseline cell and gene names must be unique")

    X, hvg_names = build_model_inputs(
        adata, PreprocessConfig(n_top_genes=expected_n_hvg)
    )
    if X.shape != (adata.n_obs, expected_n_hvg):
        raise RuntimeError(f"unexpected X_scaled shape: {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X_scaled contains non-finite values")
    return adata, X, hvg_names.to_numpy(dtype=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qc-figure-dir", type=Path)
    args = parser.parse_args()

    config = PreprocessConfig(qc_figure_dir=args.qc_figure_dir)
    result = preprocess_pbmc3k(args.data_path, config=config)
    print(
        f"Prepared {result.adata.n_obs} cells, {result.adata.n_vars} genes, "
        f"and {len(result.hvg_names)} highly variable genes."
    )
    print(f"X_scaled: {result.X_scaled.shape}")
    if args.output is not None:
        h5ad_path, arrays_path = save_preprocessed(result, args.output)
        print(f"Saved AnnData to {h5ad_path}")
        print(f"Saved model inputs to {arrays_path}")


if __name__ == "__main__":
    main()
